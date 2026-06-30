"""Three-Chain DNA-Mamba2: 基于 Mamba2 (SSD) 的三链重构

A 版本 ThreeChainMamba2: 统一时空张量重构
    核心: 三链在统一张量 x:(B,T,N²,d) 上作为 3 个扫描视角，每层残差融合。
    解决旧架构 6 个问题:
        1. 空间链无时间维 → 统一张量含完整 T 维，空间链每步都有表示
        2. Fusion 假注意力 → 输出头用 Linear 直接投影，不用 cross-attn
        3. BasePair 全局 pooling → 不用 BasePair，每层残差融合（局部信息保留）
        4. h_c/h_t 信息冗余 → 因果链直接扫描 act_emb 的 K 维，不再 mean 派生
        5. Bind 强制 L_t 对齐 → 不用 Bind，三链在统一张量自然对齐
        6. loss 防作弊失败 → 沿用 balanced_ce_loss，强化变化 cell 权重

B 版本 ThreeChainMamba2Lite: 骨架保留换 Mamba2 (对照实验)
    保留 AnchorInit/Bind/Fusion/Eagle 骨架，只把 Mamba1 → Mamba2，
    用于隔离 "Mamba2 SSD vs Mamba1 SSM" 的纯效果。

三链 Mamba2 参数表 (经 _probe_mamba2.py 验证):
    空间链: d_state=128, d_conv=4, expand=1, headdim=32, nheads=8, 双向
    时间链: d_state=64,  d_conv=4, expand=2, headdim=64, nheads=8, 因果
    因果链: d_state=32,  d_conv=4, expand=2, headdim=64, nheads=8, 因果
    约束: d_conv∈{2,3,4}; 空间链 headdim 必须为 32 (causal_conv1d stride 对齐)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2

from .common import (
    AnchorInit, Bind, Eagle, Fusion, M3SnapshotManager,
    balanced_ce_loss, SpaceAuxHead, CausalAuxHead, TimeAuxHead,
    JEPA_Predictor,
)


# ========== Mamba2 工具组件 ==========

def make_mamba2(d_model, d_state, d_conv, expand, headdim):
    """构造 Mamba2 (SSD)。参数约束:
        - d_conv ∈ {2, 3, 4}
        - headdim 必须整除 d_model * expand
        - 空间链 (expand=1) 用 headdim=32 规避 causal_conv1d stride 对齐
    """
    return Mamba2(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
    )


class BidirectionalMamba2(nn.Module):
    """双向 Mamba2: y = Mamba2(x) + flip(Mamba2(flip(x)))。

    Mamba2 默认因果(左→右)。双向扫描让空间链看到全局(cell 间无因果序)。
    共享参数以控制参数量(等价于一个 Mamba2 做两次 forward)。
    """

    def __init__(self, d_model, d_state, d_conv, expand, headdim):
        super().__init__()
        self.mamba = make_mamba2(d_model, d_state, d_conv, expand, headdim)

    def forward(self, x):
        # x: (B, L, d)
        y_fwd = self.mamba(x)
        y_bwd = self.mamba(torch.flip(x, dims=[1]))
        return y_fwd + torch.flip(y_bwd, dims=[1])


# ========== A 版本核心: HeteroMamba2 (统一时空张量) ==========

class HeteroMamba2(nn.Module):
    """A 版本三链异构核心: 在统一时空张量上做 3 视角扫描 + 每层残差融合。

    输入:
        x: (B, T, N², d)  统一时空张量
        act_emb: (B, K, T, d)  动作嵌入(因果链专用输入)
    输出:
        x: (B, T, N², d)  演化后的统一张量
        act_emb: (B, K, T, d)  演化后的因果表示

    每层三链:
        空间链: reshape(B*T, N², d) → 行+列双向 Mamba2 → 注入 x
        时间链: reshape(B*N², T, d) → 因果 Mamba2 → 注入 x
        因果链: reshape(B*T, K, d) → 因果 Mamba2 → 聚合 K → 注入 x
    残差融合: x = Norm(x + x_s + x_t + c_inject)
    """

    def __init__(self, d_model=256, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # 空间链: 行 + 列 双向扫描(2 个独立 BidirectionalMamba2)
        self.mamba_s_row = nn.ModuleList([
            BidirectionalMamba2(d_model, d_state=128, d_conv=4, expand=1, headdim=32)
            for _ in range(n_layers)
        ])
        self.mamba_s_col = nn.ModuleList([
            BidirectionalMamba2(d_model, d_state=128, d_conv=4, expand=1, headdim=32)
            for _ in range(n_layers)
        ])
        self.fusion_s = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(n_layers)
        ])

        # 时间链: 因果 Mamba2 (Mamba2 默认因果)
        self.mamba_t = nn.ModuleList([
            make_mamba2(d_model, d_state=64, d_conv=4, expand=2, headdim=64)
            for _ in range(n_layers)
        ])

        # 因果链: 对 K 维因果扫描
        self.mamba_c = nn.ModuleList([
            make_mamba2(d_model, d_state=32, d_conv=4, expand=2, headdim=64)
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

    def forward(self, x, act_emb):
        """
        x: (B, T, N², d)
        act_emb: (B, K, T, d)
        Returns: (x, act_emb) 同形状
        """
        B, T, N2, D = x.shape
        K = act_emb.shape[1]
        N = int(math.isqrt(N2))

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

            # --- 时间链: 因果 Mamba2 ---
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

            # --- 残差融合 ---
            x = self.norm_fuse[i](x + x_s + x_t + c_inject)

        return x, act_emb


# ========== A 版本模型: ThreeChainMamba2 ==========

class ThreeChainMamba2(nn.Module):
    """A 版本: 统一时空张量三链 Mamba2。

    架构:
        AnchorInit2: x = cell_emb + act_mean + time_emb  → (B, T, N², d)
        HeteroMamba2 (n_layers): 每层三链残差融合
        输出头: Linear(d → C) → (B, T, N, N, C)

    因果性:
        - 时间维因果(Mamba2 默认): logits[t] 只依赖 S_0 + actions[:, :t+1]
        - 空间维双向(同时间步内 cell 并行, 无因果序)
        - agent 维因果(低 ID → 高 ID, 匹配冲突解决规则)
    """

    def __init__(self, cell_types=16, action_dim=5, d_model=256, n_layers=2,
                 max_T=256):
        super().__init__()
        self.cell_types = cell_types
        self.d_model = d_model
        self.n_layers = n_layers

        # embeddings
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        self.time_embed = nn.Embedding(max_T, d_model)

        # 三链核心
        self.mamba = HeteroMamba2(d_model, n_layers)

        # 输出头: Linear 直接投影(不用 attention, 解决问题2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, cell_types),
        )

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

        # --- 三链演化 ---
        x, act_emb_out = self.mamba(x, act_emb)

        # --- 输出头: Linear 直接投影 ---
        logits = self.head(x)  # (B, T, N², C)
        logits = logits.view(B, T, N, N, self.cell_types)

        info = {
            "h_last": x[:, -1],           # (B, N², d) 最后时间步
            "act_emb_out": act_emb_out,   # (B, K, T, d)
        }
        return logits, info

    def loss(self, logits, S_t, info, aux_weight=0.3):
        """balanced_ce_loss + 轻量正则。

        logits: (B, T, N, N, C)
        S_t: (B, T+1, N, N)
        """
        S_0 = S_t[:, 0]
        total, loss_info = balanced_ce_loss(logits, S_t, S_0, aux_weight=aux_weight)
        return total, loss_info


# ========== B 版本核心: HeteroMamba2Lite (骨架保留换 Mamba2) ==========

class HeteroMamba2Lite(nn.Module):
    """B 版本三链核心: 保留 HeteroMamba 骨架, Mamba1 → Mamba2。

    保留: 空间双向扫描 + 因果 agent 门控 + 时间多尺度 + TriHelixBasePairCoupling
    只换: 4 个 Mamba() → Mamba2()
    用于隔离 Mamba2 SSD vs Mamba1 SSM 的纯效果(控制变量)。
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        # === 空间链: 双向交叉扫描 (Mamba2) ===
        # headdim=32 规避 causal_conv1d stride 对齐 (expand=1)
        self.mamba_s_row = make_mamba2(d_model, d_state=128, d_conv=4, expand=1, headdim=32)
        self.mamba_s_col = make_mamba2(d_model, d_state=128, d_conv=4, expand=1, headdim=32)
        self.fusion_s = nn.Linear(d_model * 2, d_model)
        self.norm_s = nn.LayerNorm(d_model)

        # === 因果链: Agent 间门控 (Mamba2) ===
        self.mamba_c = make_mamba2(d_model, d_state=32, d_conv=4, expand=2, headdim=64)
        self.agent_gate = nn.Linear(d_model * 2, d_model)
        self.norm_c = nn.LayerNorm(d_model)

        # === 时间链: 多尺度膨胀卷积 (Mamba2) ===
        self.mamba_t = make_mamba2(d_model, d_state=64, d_conv=4, expand=2, headdim=64)
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 4]
        ])
        self.fusion_t = nn.Linear(d_model * 3, d_model)
        self.norm_t = nn.LayerNorm(d_model)

        # === 三螺旋碱基对耦合 (保留骨架) ===
        from .common import TriHelixBasePairCoupling
        self.basepair = TriHelixBasePairCoupling(d_model)

    def forward(self, h_s, h_c, h_t):
        """
        h_s: (B, N², d)  空间序列
        h_c: (B, K, d)    因果序列
        h_t: (B, T, d)    时间序列
        Returns: (h_s', h_c', h_t', bp_info)
        """
        B = h_s.shape[0]
        N2 = h_s.shape[1]
        N = int(N2 ** 0.5)
        D = self.d_model

        # --- 空间链: 双向交叉扫描 ---
        row_out = self.mamba_s_row(h_s)
        h_s_2d = h_s.view(B, N, N, D)
        col_seq = h_s_2d.permute(0, 2, 1, 3).reshape(B, N2, D)
        col_out = self.mamba_s_col(col_seq)
        h_s_out = self.fusion_s(torch.cat([row_out, col_out], dim=-1))
        h_s_out = self.norm_s(h_s + h_s_out)

        # --- 因果链: Agent 间门控 ---
        c_out = self.mamba_c(h_c)
        agent_sum = c_out.sum(dim=1, keepdim=True)
        K = h_c.shape[1]
        agent_mean = (agent_sum - c_out) / max(1, K - 1)
        gate = torch.sigmoid(self.agent_gate(torch.cat([c_out, agent_mean], dim=-1)))
        h_c_out = self.norm_c(h_c + gate * c_out + (1 - gate) * agent_mean)

        # --- 时间链: 多尺度膨胀卷积 ---
        t_out = self.mamba_t(h_t)
        t_conv = t_out.permute(0, 2, 1)
        multi_scale = [conv(t_conv) for conv in self.dilated_convs]
        t_fused = torch.cat(multi_scale, dim=1).permute(0, 2, 1)
        h_t_out = self.fusion_t(t_fused)
        h_t_out = self.norm_t(h_t + h_t_out)

        # --- 碱基对耦合 ---
        h_s_out, h_c_out, h_t_out, bp_info = self.basepair(h_s_out, h_c_out, h_t_out)

        return h_s_out, h_c_out, h_t_out, bp_info


# ========== B 版本模型: ThreeChainMamba2Lite ==========

class ThreeChainMamba2Lite(nn.Module):
    """B 版本: 保留 ThreeChain 骨架, HeteroMamba → HeteroMamba2Lite。

    保留: AnchorInit/Bind/Fusion/Eagle/JEPA/辅助头 全部骨架
    只换: HeteroMamba(Mamba1) → HeteroMamba2Lite(Mamba2)
    用于控制变量对照: 隔离 Mamba2 SSD 的纯效果。
    """

    def __init__(self, cell_types=16, action_dim=5, d_model=256, n_layers=2,
                 bind_heads=4, fusion_heads=4, N=8, max_T=100):
        super().__init__()
        self.cell_types = cell_types
        self.N = N
        self.anchor = AnchorInit(cell_types, action_dim, d_model)
        self.mamba = HeteroMamba2Lite(d_model)
        self.bind = Bind(d_model, bind_heads)
        self.eagle = Eagle()
        self.fusion = Fusion(d_model, cell_types, fusion_heads)
        self.m3 = M3SnapshotManager()
        self.space_head = SpaceAuxHead(d_model, cell_types, N)
        self.causal_head = CausalAuxHead(d_model, max_N=12)
        self.time_head = TimeAuxHead(d_model, max_T)
        self.jepa = JEPA_Predictor(d_state_total=d_model, n_future=3, hidden=64)

    @property
    def mamba_s(self):
        return self.mamba.mamba_s_row

    @property
    def mamba_c(self):
        return self.mamba.mamba_c

    @property
    def mamba_t(self):
        return self.mamba.mamba_t

    def forward(self, S_0, actions):
        B, N, _ = S_0.shape
        h_s, h_c, h_t = self.anchor(S_0, actions)
        h_s, h_c, h_t, bp_info = self.mamba(h_s, h_c, h_t)
        space_logits = self.space_head(h_s)
        causal_logits = self.causal_head(h_c, N)
        time_pred = self.time_head(h_t)
        h_bind = self.bind(h_s, h_c, h_t)
        h_last = h_bind[:, -1]
        jepa_pred = self.jepa(h_last)
        rollback_mask, drift = self.eagle(h_bind)
        self.m3.save(0, h_bind[:, -1])
        logits = self.fusion(h_bind, h_s)
        aux = {"space_logits": space_logits, "causal_logits": causal_logits,
               "time_pred": time_pred, "bp_info": bp_info,
               "jepa_pred": jepa_pred, "h_bind_last": h_bind[:, -3:]}
        return logits, {"rollback_mask": rollback_mask, "drift": drift, "aux": aux}

    def loss(self, logits, S_t, eagle_info, aux_weight=0.3,
             space_weight=0.5, causal_weight=0.5, time_weight=0.3):
        S_0 = S_t[:, 0]
        total, info = balanced_ce_loss(logits, S_t, S_0, aux_weight=aux_weight)
        aux = eagle_info.get("aux", {})
        if "space_logits" in aux and aux["space_logits"] is not None:
            target_final = S_t[:, -1]
            space_loss = F.cross_entropy(
                aux["space_logits"].permute(0, 3, 1, 2), target_final, reduction="mean")
            total = total + space_weight * space_loss
            info["space_loss"] = space_loss.item()
        if "causal_logits" in aux and aux["causal_logits"] is not None:
            target_change = (S_t[:, -1] != S_0).long()
            causal_loss = F.cross_entropy(
                aux["causal_logits"].permute(0, 3, 1, 2), target_change, reduction="mean")
            total = total + causal_weight * causal_loss
            info["causal_loss"] = causal_loss.item()
        if "jepa_pred" in aux and "h_bind_last" in aux:
            jepa_loss = self.jepa.compute_loss(aux["jepa_pred"], aux["h_bind_last"])
            total = total + 0.3 * jepa_loss
            info["jepa_loss"] = jepa_loss.item()
        if "time_pred" in aux and aux["time_pred"] is not None:
            B2, T2, _ = aux["time_pred"].shape
            time_target = torch.linspace(0, 1, T2, device=aux["time_pred"].device)
            time_target = time_target.unsqueeze(0).unsqueeze(-1).expand(B2, T2, 1)
            time_loss = F.mse_loss(aux["time_pred"], time_target)
            total = total + time_weight * time_loss
            info["time_loss"] = time_loss.item()
        return total, info
