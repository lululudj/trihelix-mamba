"""Three-Chain DNA-Mamba3 (官方版): 基于 state-spaces/mamba 官方 Mamba3 的三链重构

A 版本 ThreeChainMamba3: 与 ThreeChainMamba2 同骨架，只把 Mamba2 → Mamba3 (官方 Triton 内核版)

与之前 mamba3_ref 版本的差异（本文件已切换到官方版）:
    1. import Mamba3 from mamba_ssm (官方 state-spaces/mamba, 含 Triton 内核)
       — 之前 from .mamba3_ref import Mamba3 是第三方纯 Python 移植, 无内核
    2. 删除 _ckpt_call 梯度检查点 — Triton 内核自带高效反向传播, 不再需要
    3. make_mamba3 加 chunk_size=64 (官方推荐: SISO 用 64, MIMO 用 64/rank)
    4. forward 速度恢复到 Mamba2 同量级 (1.6ms vs mamba3_ref 几秒, 快 1000x)

使用方式 (PYTHONPATH 开发模式, 不编译):
    export PYTHONPATH=/mnt/e/mamba_official:$PYTHONPATH
    # 然后正常跑 train.py --model three_chain_mamba3 ...
    # mamba_ssm 2.3.2.post1 (含 Mamba3) 会从 /mnt/e/mamba_official 加载
    # 系统级 mamba_ssm 2.2.4 (Mamba2 baseline) 不受影响

Mamba3 三链参数表 (与 Mamba2 版对齐, 仅去 d_conv):
    空间链: d_state=128, expand=1, headdim=32, nheads=8, 双向
    时间链: d_state=64,  expand=2, headdim=64, nheads=8, 因果
    因果链: d_state=32,  expand=2, headdim=64, nheads=8, 因果
    约束: d_inner (=expand*d_model) 必须被 headdim 整除
    注意: Mamba3 无 d_conv 参数 (用梯形离散化代替卷积)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba3  # 官方版 (含 Triton 内核)

from .common import (
    AnchorInit, Bind, Eagle, Fusion, M3SnapshotManager,
    balanced_ce_loss, SpaceAuxHead, CausalAuxHead, TimeAuxHead,
    JEPA_Predictor,
)


# ========== Mamba3 工具组件 ==========

def make_mamba3(d_model, d_state, expand, headdim, chunk_size=64):
    """构造官方 Mamba3 (梯形离散化 + RoPE + Triton 内核)。参数约束:
        - d_inner = expand * d_model 必须被 headdim 整除
        - 无 d_conv 参数 (Mamba3 用梯形离散化代替卷积)
        - chunk_size=64 官方推荐 SISO 模式用 64; MIMO 用 64/rank
        - is_mimo=False: SISO 模式, 只需 Triton 内核 (避免 TileLang 依赖)
    """
    return Mamba3(
        d_model=d_model,
        d_state=d_state,
        expand=expand,
        headdim=headdim,
        ngroups=1,
        is_mimo=False,      # SISO 模式 (MIMO 是 decode 优化, 训练用不上)
        chunk_size=chunk_size,
        is_outproj_norm=False,
    )


class BidirectionalMamba3(nn.Module):
    """双向 Mamba3: y = Mamba3(x) + flip(Mamba3(flip(x)))。

    Mamba3 默认因果(左→右)。双向扫描让空间链看到全局(cell 间无因果序)。
    共享参数以控制参数量(等价于一个 Mamba3 做两次 forward)。
    注意: 双向会让 forward 调 2 次 Mamba3, 速度 ×2。
    """

    def __init__(self, d_model, d_state, expand, headdim, chunk_size=64):
        super().__init__()
        self.mamba = make_mamba3(d_model, d_state, expand, headdim, chunk_size)

    def forward(self, x):
        # x: (B, L, d) — 直接调 Mamba3 (Triton 内核, 不需要 checkpoint)
        y_fwd = self.mamba(x)
        y_bwd = self.mamba(torch.flip(x, dims=[1]))
        return y_fwd + torch.flip(y_bwd, dims=[1])


# ========== A 版本核心: HeteroMamba3 (统一时空张量) ==========

class HeteroMamba3(nn.Module):
    """A 版本三链异构核心: 在统一时空张量上做 3 视角扫描 + 每层残差融合。
    与 HeteroMamba2 完全同构, 仅 Mamba2 → Mamba3 (官方 Triton 版)。

    输入:
        x: (B, T, N², d)  统一时空张量
        act_emb: (B, K, T, d)  动作嵌入(因果链专用输入)
    输出:
        x: (B, T, N², d)  演化后的统一张量
        act_emb: (B, K, T, d)  演化后的因果表示

    每层三链:
        空间链: reshape(B*T, N², d) → 行+列双向 Mamba3 → 注入 x
        时间链: reshape(B*N², T, d) → 因果 Mamba3 → 注入 x
        因果链: reshape(B*T, K, d) → 因果 Mamba3 → 聚合 K → 注入 x
    残差融合: x = Norm(x + x_s + x_t + c_inject)
    """

    def __init__(self, d_model=256, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # 空间链: 行 + 列 双向扫描(2 个独立 BidirectionalMamba3)
        # expand=1, headdim=32 (与 Mamba2 版对齐; Mamba3 无 stride 约束但保留对照公平)
        self.mamba_s_row = nn.ModuleList([
            BidirectionalMamba3(d_model, d_state=128, expand=1, headdim=32)
            for _ in range(n_layers)
        ])
        self.mamba_s_col = nn.ModuleList([
            BidirectionalMamba3(d_model, d_state=128, expand=1, headdim=32)
            for _ in range(n_layers)
        ])
        self.fusion_s = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(n_layers)
        ])

        # 时间链: 因果 Mamba3 (Mamba3 默认因果)
        self.mamba_t = nn.ModuleList([
            make_mamba3(d_model, d_state=64, expand=2, headdim=64)
            for _ in range(n_layers)
        ])

        # 因果链: 对 K 维因果扫描
        self.mamba_c = nn.ModuleList([
            make_mamba3(d_model, d_state=32, expand=2, headdim=64)
            for _ in range(n_layers)
        ])

        # 每层残差融合 LayerNorm
        self.norm_fuse = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        # 因果链注入投影(聚合 K 维后投影回 d)
        self.causal_inject = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])

    def forward(self, x, act_emb, collect_states=False):
        """
        x: (B, T, N², d)
        act_emb: (B, K, T, d)
        collect_states: 是否收集每层三链状态（供 .m3 存储和 JEPA 用，省显存时关）
        Returns: (x, act_emb, chain_states)
            chain_states: list of (h_s, h_t, h_c) per layer，或 None（当 collect_states=False）
                h_s: (B, T, N², d) 空间链输出（残差融合前）
                h_t: (B, T, N², d) 时间链输出（残差融合前）
                h_c: (B, K, T, d) 因果链 act_emb 更新后
        """
        B, T, N2, D = x.shape
        K = act_emb.shape[1]
        N = int(math.isqrt(N2))

        chain_states = [] if collect_states else None

        for i in range(self.n_layers):
            # --- 空间链: 行+列双向扫描 ---
            # 行优先: (B*T, N², d)
            x_row = x.reshape(B * T, N2, D)
            row_out = self.mamba_s_row[i](x_row)  # (B*T, N², d)
            # 列优先: 转置 N×N 后展平
            x_2d = x.reshape(B, T, N, N, D)
            x_col = x_2d.permute(0, 1, 3, 2, 4).reshape(B * T, N2, D)  # 列优先
            col_out = self.mamba_s_col[i](x_col)
            # 转回行优先顺序
            col_out = col_out.reshape(B, T, N, N, D).permute(0, 1, 3, 2, 4).reshape(B * T, N2, D)
            # 融合行+列
            x_s = self.fusion_s[i](torch.cat([row_out, col_out], dim=-1))  # (B*T, d)
            x_s = x_s.reshape(B, T, N2, D)

            # --- 时间链: 因果 Mamba3 ---
            # (B*N², T, d): 对每个 cell 沿时间因果扫描
            x_t = x.permute(0, 2, 1, 3).reshape(B * N2, T, D)  # (B*N², T, d)
            x_t = self.mamba_t[i](x_t)
            x_t = x_t.reshape(B, N2, T, D).permute(0, 2, 1, 3)  # (B, T, N², d)

            # --- 因果链: K 维因果扫描 ---
            # act_emb (B, K, T, d) → (B*T, K, d)
            x_c = act_emb.permute(0, 2, 1, 3).reshape(B * T, K, D)
            x_c = self.mamba_c[i](x_c)  # (B*T, K, d)
            # 更新 act_emb (残差)
            act_emb = x_c.reshape(B, T, K, D).permute(0, 2, 1, 3)  # (B, K, T, d)
            # 聚合 K 维 → 注入 x
            c_pool = x_c.mean(dim=1)  # (B*T, d)
            c_inject = self.causal_inject[i](c_pool)  # (B*T, d)
            c_inject = c_inject.reshape(B, T, 1, D)  # 广播到 N²

            # --- 收集每层三链状态（供 .m3 存储和 JEPA）---
            if collect_states:
                chain_states.append((x_s, x_t, act_emb))

            # --- 残差融合 ---
            x = self.norm_fuse[i](x + x_s + x_t + c_inject)

        return x, act_emb, chain_states


# ========== A 版本模型: ThreeChainMamba3 ==========

class ThreeChainMamba3(nn.Module):
    """A 版本: 统一时空张量三链 Mamba3 (官方 Triton 内核版)。

    与 ThreeChainMamba2 完全同构, 仅 Mamba2 → Mamba3。
    用于对照实验: 隔离 Mamba3 (梯形离散化 + RoPE, 官方 Triton) vs Mamba2 (SSD) 的纯效果。

    架构:
        AnchorInit2: x = cell_emb + act_mean + time_emb  → (B, T, N², d)
        HeteroMamba3 (n_layers): 每层三链残差融合
        输出头: Linear(d → C) → (B, T, N, N, C)

    因果性:
        - 时间维因果(Mamba3 默认): logits[t] 只依赖 S_0 + actions[:, :t+1]
        - 空间维双向(同时间步内 cell 并行, 无因果序)
        - agent 维因果(低 ID → 高 ID, 匹配冲突解决规则)
    """

    def __init__(self, cell_types=16, action_dim=5, d_model=256, n_layers=2,
                 max_T=256, enable_m3=False, enable_jepa=False, jepa_weight=0.3):
        super().__init__()
        self.cell_types = cell_types
        self.d_model = d_model
        self.n_layers = n_layers
        self.enable_m3 = enable_m3
        self.enable_jepa = enable_jepa
        self.jepa_weight = jepa_weight  # 接通 config: cfg["model"]["jepa_weight"]

        # embeddings
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        self.time_embed = nn.Embedding(max_T, d_model)

        # 三链核心 (Mamba3 官方版)
        self.mamba = HeteroMamba3(d_model, n_layers)

        # 输出头: Linear 直接投影(不用 attention, 解决问题2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, cell_types),
        )

        # 阶段 A: .m3 记忆 + JEPA 预言（开关控制，默认关闭保留 baseline）
        if enable_m3:
            self.m3 = M3SnapshotManager()
        if enable_jepa:
            self.jepa = JEPA_Predictor(d_state_total=d_model, n_future=3, hidden=64)
        self._last_chain_states = None  # 缓存最近一次 forward 的 chain_states，供 save_m3 用

    @property
    def mamba_s(self):
        """暴露空间链最后一层(eval.py 正交性钩子用)。"""
        return self.mamba.mamba_s_row[-1].mamba

    @property
    def mamba_c(self):
        """暴露因果链最后一层。"""
        return self.mamba.mamba_c[-1]

    @property
    def mamba_t(self):
        """暴露时间链最后一层。"""
        return self.mamba.mamba_t[-1]

    def forward(self, S_0, actions):
        """
        S_0: (B, N, N) long
        actions: (B, K, T) long
        Returns: logits (B, T, N, N, C), info dict
        """
        B, N, _ = S_0.shape
        K, T = actions.shape[1], actions.shape[2]
        D = self.d_model
        N2 = N * N

        # --- AnchorInit2: 统一时空张量 ---
        cell_emb = self.cell_embed(S_0.reshape(B, N2))  # (B, N², d)
        act_emb = self.action_embed(actions)  # (B, K, T, d)
        time_idx = torch.arange(T, device=S_0.device)
        time_emb = self.time_embed(time_idx)  # (T, d)

        # x: (B, T, N², d) = cell + act_mean(每步) + time_pos
        act_mean = act_emb.mean(dim=1)  # (B, T, d)
        x = (cell_emb.unsqueeze(1)                      # (B, 1, N², d)
             + act_mean.unsqueeze(2)                    # (B, T, 1, d)
             + time_emb.view(1, T, 1, D))              # (1, T, 1, d)

        # --- 三链演化（开关控制是否收集每层状态，省显存）---
        collect = self.enable_m3 or self.enable_jepa
        x, act_emb_out, chain_states = self.mamba(x, act_emb, collect_states=collect)

        # --- 输出头: Linear 直接投影 ---
        logits = self.head(x)  # (B, T, N², C)
        logits = logits.view(B, T, N, N, self.cell_types)

        info = {
            "h_last": x[:, -1],           # (B, N², d) 最后时间步
            "act_emb_out": act_emb_out,   # (B, K, T, d)
        }

        # --- 阶段 A 钩子：JEPA 预言（接空间链）---
        if self.enable_jepa and chain_states is not None:
            # 取最后一层空间链 h_s: (B, T, N², d) → N² 维 mean-pool → (B, T, d)
            h_s_last = chain_states[-1][0]  # (B, T, N², d)
            h_seq = h_s_last.mean(dim=2)    # (B, T, d)
            # JEPA 输入最后一步，target 用最后 3 步（余弦相似度 loss）
            jepa_pred = self.jepa(h_seq[:, -1])      # (B, 3, d)
            jepa_target = h_seq[:, -3:]               # (B, 3, d)
            info["jepa_pred"] = jepa_pred
            info["jepa_target"] = jepa_target

        # --- 阶段 A 钩子：.m3 记忆（缓存 chain_states 供 save_m3 用）---
        if self.enable_m3 and chain_states is not None:
            self._last_chain_states = [
                (s[0].detach(), s[1].detach(), s[2].detach()) for s in chain_states
            ]

        return logits, info

    def loss(self, logits, S_t, info, aux_weight=0.3, jepa_weight=None):
        """balanced_ce_loss + 轻量正则 + JEPA 潜空间预测 loss（阶段 A）。

        logits: (B, T, N, N, C)
        S_t: (B, T+1, N, N)
        info: dict，含 jepa_pred/jepa_target（当 enable_jepa=True 时）
        aux_weight: balanced_ce_loss 的辅助权重
        jepa_weight: JEPA loss 权重；None 时用 self.jepa_weight（来自 config），
                    传具体值则 override（向后兼容）
        """
        S_0 = S_t[:, 0]
        total, loss_info = balanced_ce_loss(logits, S_t, S_0, aux_weight=aux_weight)

        # 阶段 A: JEPA 潜空间预测 loss（余弦相似度）
        if "jepa_pred" in info and "jepa_target" in info:
            w = self.jepa_weight if jepa_weight is None else jepa_weight
            jepa_loss = self.jepa.compute_loss(info["jepa_pred"], info["jepa_target"])
            total = total + w * jepa_loss
            loss_info["jepa_loss"] = jepa_loss.item()

        return total, loss_info

    def save_m3(self, path, sample_id, N, K, T, step=0, scenario_type="random", batch_idx=0):
        """把最近一次 forward 缓存的三链状态存成 .m3 文件（仅 enable_m3=True 时可用）。

        从 self._last_chain_states（含 batch 维 B）切第 batch_idx 个样本，每层存:
            h_s: (N², d)  空间链最后时间步（树干）
            h_t: (T,  d)  时间链整段 N² 均值（树根）
            h_c: (K,  d)  因果链最后时间步所有 agent（枝叶）

        必须在 forward 之后、下一次 forward 之前调用（_last_chain_states 会被覆盖）。
        """
        if not self.enable_m3:
            raise RuntimeError("enable_m3=False，无法保存 .m3（forward 未缓存 chain_states）")
        if self._last_chain_states is None:
            raise RuntimeError("尚未 forward，_last_chain_states 为空，无法 save_m3")

        chain_states_sq = []
        for (h_s, h_t, h_c) in self._last_chain_states:
            # h_s: (B, T, N², d) → 最后时间步 → batch_idx → (N², d)
            s = h_s[batch_idx, -1]
            # h_t: (B, T, N², d) → N² 维 mean → batch_idx → (T, d)
            t = h_t[batch_idx].mean(dim=1)
            # h_c: (B, K, T, d) → 最后时间步 → batch_idx → (K, d)
            c = h_c[batch_idx, :, -1, :]
            chain_states_sq.append({"h_s": s, "h_t": t, "h_c": c})

        meta = {
            "N": int(N), "K": int(K), "T": int(T),
            "d_model": int(self.d_model),
            "n_layers": int(self.n_layers),
            "step": int(step), "sample_id": int(sample_id),
            "scenario_type": str(scenario_type),
        }
        self.m3.save_to_file(path, meta, chain_states_sq)
