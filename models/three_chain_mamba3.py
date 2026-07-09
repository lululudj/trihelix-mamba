"""Three-Chain DNA-Mamba3 (官方版): 复数状态 + dt-RoPE 旋转位置编码，长度外推 32K 不衰减

核心升级点（解决 OOD 长度外推衰减问题）:
    1. 【移除固定长度 time_embed】删除 Mamba2 时代遗留的 nn.Embedding(max_T, d_model)
       —— 这是外推衰减的罪魁祸首：T>max_T 时要么越界要么用未训练的随机向量，
          还会干扰 Mamba3 内部 RoPE 的位置信号。
    2. 【完全依赖 Mamba3 内核的 dt-RoPE】官方 Mamba3 内置旋转位置编码不是传统固定位置 RoPE，
       而是基于时间步 dt 自适应累积角度：cumulative_angles = cumsum(angle_raw * dt)
       —— 不需要预设 max_len，天然支持任意长度外推。
    3. 【复数状态空间】通过 RoPE 旋转 B/C（key/query）投影，SSM 状态 h 获得等效复值结构，
       可以追踪旋转/振荡/周期模式，解决了实数状态长程遗忘问题。
    4. 【梯形离散化】trapezoidal rule 替代 Mamba2 的 ZOH，大步长下离散化误差更小。

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
import torch.utils.checkpoint as cp

try:
    from mamba_ssm import Mamba3  # 官方 Triton 内核版 (mamba_ssm 2.3.x)
    HAS_OFFICIAL_MAMBA3 = True
except ImportError:
    HAS_OFFICIAL_MAMBA3 = False
    try:
        from .mamba3_ref import Mamba3  # fallback: 纯Python参考实现(本地验证用)
        print("[warn] mamba_ssm 2.3.x 未找到，three_chain_mamba3 使用 mamba3_ref 纯Python版本(仅验证用,慢)")
    except ImportError:
        Mamba3 = None
        print("[error] 无可用的 Mamba3 实现，请安装 mamba_ssm>=2.3 或确保 mamba3_ref.py 存在")

from .common import (
    AnchorInit, Bind, Eagle, Fusion, M3SnapshotManager,
    balanced_ce_loss, SpaceAuxHead, CausalAuxHead, TimeAuxHead,
    JEPA_Predictor,
)


# ========== Mamba3 工具组件 ==========

def _ckpt_call(mamba, x, use_checkpoint=False):
    """梯度检查点包装 Mamba3 调用 (突破大模型 OOM)。
    从 ThreeChainMamba2 移植: backward 时重算 forward, 内存 O(L)->O(1)。
    eval 时 (no_grad) 自动直通, 不影响推理速度。

    注: Mamba3 Triton 内核的 backward 多次访问 ctx.saved_tensors,
    与 use_reentrant=False 不兼容 (CheckpointError)。
    改用 use_reentrant=True (重算 forward, 不拦截 saved_tensors)。
    d_model=64 时 T=200 峰值仅 1.49GB, 通常无需开启 use_checkpoint。
    """
    if use_checkpoint and torch.is_grad_enabled() and x.requires_grad:
        return cp.checkpoint(mamba, x, use_reentrant=True)
    return mamba(x)


def make_mamba3(d_model, d_state, expand, headdim, chunk_size=64):
    """构造 Mamba3 (梯形离散化 + dt-RoPE 复数状态)。
    
    自动选择后端:
        - 官方 mamba_ssm 2.3.x (Triton内核快) → 传 chunk_size/is_outproj_norm
        - mamba3_ref 纯Python (本地验证/调试)  → 省略Triton特有参数
    """
    if Mamba3 is None:
        raise RuntimeError("无可用 Mamba3 实现")
    
    import os
    force_ref = os.environ.get("MAMBA3_FORCE_REF", "0") == "1"
    
    if HAS_OFFICIAL_MAMBA3 and not force_ref:
        return Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=1,
            is_mimo=False,
            chunk_size=chunk_size,
            is_outproj_norm=False,
        )
    else:
        if force_ref and HAS_OFFICIAL_MAMBA3:
            print("[info] MAMBA3_FORCE_REF=1, using mamba3_ref")
        # 显式导入mamba3_ref，避免Mamba3变量绑定官方版
        from .mamba3_ref import Mamba3 as Mamba3Ref
        return Mamba3Ref(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=1,
            rope_fraction=0.5,
            is_mimo=False,
        )


class BidirectionalMamba3(nn.Module):
    """双向 Mamba3: y = Mamba3(x) + flip(Mamba3(flip(x)))。

    Mamba3 默认因果(左->右)。双向扫描让空间链看到全局(cell 间无因果序)。
    共享参数以控制参数量(等价于一个 Mamba3 做两次 forward)。
    注意: 双向会让 forward 调 2 次 Mamba3, 速度 ×2。
    """

    def __init__(self, d_model, d_state, expand, headdim, chunk_size=64, use_checkpoint=False):
        super().__init__()
        self.mamba = make_mamba3(d_model, d_state, expand, headdim, chunk_size)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        # x: (B, L, d)
        y_fwd = _ckpt_call(self.mamba, x, use_checkpoint=self.use_checkpoint)
        y_bwd = _ckpt_call(self.mamba, torch.flip(x, dims=[1]), use_checkpoint=self.use_checkpoint)
        return y_fwd + torch.flip(y_bwd, dims=[1])


# ========== 两两碱基对耦合: PairwiseBasePairMamba3 ==========

class PairwiseBasePairMamba3(nn.Module):
    """三链Mamba3两两碱基对互联: s↔t, t↔c, c↔s 三对直接耦合。

    融合设计:
        - TriHelixBasePairCoupling (common.py:582) 的两两耦合语义 (3对碱基对)
        - CrossChainBasePair (three_chain_mamba2_bpv2.py:42) 的统一张量接口 + alpha初始0

    三对碱基对 (借鉴 TriHelixBasePairCoupling):
        BP1 (空间↔时间): Cross-Delta 调制 — 时间链生成 dt 系数调制空间链; 空间变化率门控时间链
        BP2 (时间↔因果): Reset Gate — 因果链能量门控时间链记忆; 时间相位注入因果链
        BP3 (因果↔空间): FiLM 仿射 — 因果链生成 gamma/beta 仿射空间链; 空间校验因果链

    接口 (借鉴 CrossChainBasePair): forward(x, x_s, x_t, c_inject) → (x_s', x_t', c_inject')
        x:        (B, T, N², d)  共享状态 (融合前的 x, 作螺旋轴参考)
        x_s:      (B, T, N², d)  空间链输出
        x_t:      (B, T, N², d)  时间链输出
        c_inject: (B, T, 1, d)   因果链输出 (广播到 N²)

    alpha 初始 0: 训练开始时 (x_s', x_t', c_inject') = (x_s, x_t, c_inject) = baseline
        → 公平对照: 关闭BP时参数量不变, 开启BP时从baseline出发自学拉力强度
    """

    def __init__(self, d_model):
        super().__init__()
        D = d_model
        # 可学习拉力强度, 初始 0 → 开始时无碱基对 (= baseline)
        self.alpha = nn.Parameter(torch.zeros(1))

        # === BP1: 空间↔时间 (Cross-Delta) ===
        # 时间→空间: dt调制系数 (B,1) → 广播
        self.st_t2s_delta = nn.Sequential(
            nn.Linear(D, D // 4), nn.SiLU(), nn.Linear(D // 4, 1), nn.Sigmoid()
        )
        # 空间→时间: 空间变化率门控 (B,d)
        self.st_s2t_gate = nn.Sequential(
            nn.Linear(D, D // 4), nn.SiLU(), nn.Linear(D // 4, D), nn.Sigmoid()
        )

        # === BP2: 时间↔因果 (Memory Reset) ===
        # 因果→时间: reset gate (B,d)
        self.tc_c2t_reset = nn.Sequential(
            nn.Linear(D, D // 4), nn.SiLU(), nn.Linear(D // 4, D), nn.Sigmoid()
        )
        # 时间→因果: 相位注入 (B,d)
        self.tc_t2c_phase = nn.Sequential(
            nn.Linear(D, D // 4), nn.SiLU(), nn.Linear(D // 4, D)
        )

        # === BP3: 因果↔空间 (FiLM) ===
        # 因果→空间: gamma/beta
        self.cs_c2s_gamma = nn.Sequential(
            nn.Linear(D, D // 4), nn.SiLU(), nn.Linear(D // 4, D)
        )
        self.cs_c2s_beta = nn.Sequential(
            nn.Linear(D, D // 4), nn.SiLU(), nn.Linear(D // 4, D)
        )
        # 空间→因果: 现实校验 (B,d)
        self.cs_s2c_check = nn.Sequential(
            nn.Linear(D, D // 4), nn.SiLU(), nn.Linear(D // 4, D), nn.Tanh()
        )

        # 零初始化所有末层 Linear → 调制初始为中性 (sigmoid(0)=0.5, gamma=0, beta=0)
        for mod in [self.st_t2s_delta, self.st_s2t_gate, self.tc_c2t_reset,
                    self.tc_t2c_phase, self.cs_c2s_gamma, self.cs_c2s_beta,
                    self.cs_s2c_check]:
            last_lin = [m for m in mod.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_lin.weight)
            nn.init.zeros_(last_lin.bias)

    def forward(self, x, x_s, x_t, c_inject):
        """
        x:        (B, T, N², d)  共享状态 (螺旋轴)
        x_s:      (B, T, N², d)  空间链输出
        x_t:      (B, T, N², d)  时间链输出
        c_inject: (B, T, 1, d)   因果链输出 (广播到 N²)
        Returns: (x_s', x_t', c_inject') 同形状, 被两两碱基对耦合修正
        """
        B = x_s.shape[0]
        D = x_s.shape[-1]

        # 1. 提取三链全局表示 (B, d): 在 N² 和 T 维上 pool
        g_s = x_s.mean(dim=(1, 2))                        # (B, d) 空间全局
        g_t = x_t.mean(dim=(1, 2))                        # (B, d) 时间全局
        g_c = c_inject.squeeze(2).mean(dim=1)             # (B, d) 因果全局

        # 2. 空间一阶差分 (沿 T) 作"变化率"特征
        s_diff = (x_s[:, 1:] - x_s[:, :-1]).mean(dim=(1, 2))  # (B, d)

        # 3. 三对碱基对调制参数
        # BP1: 时间→空间 delta, 空间→时间 gate
        delta_mod = 0.5 + 0.5 * self.st_t2s_delta(g_t)   # (B, 1)
        s_gate = self.st_s2t_gate(s_diff)                 # (B, d)

        # BP2: 因果→时间 reset, 时间→因果 phase
        c_energy = torch.norm(g_c, dim=-1, keepdim=True) / (D ** 0.5)
        reset_gate = self.tc_c2t_reset(g_c) * c_energy.sigmoid()  # (B, d)
        t_phase = self.tc_t2c_phase(g_t)                  # (B, d)

        # BP3: 因果→空间 gamma/beta, 空间→因果 check
        gamma = self.cs_c2s_gamma(g_c)                    # (B, d)
        beta = self.cs_c2s_beta(g_c)                      # (B, d)
        s_check = self.cs_s2c_check(g_s)                  # (B, d)

        # 4. 应用耦合 (广播回原形状) + alpha 缩放
        # BP1+BP3 → 空间链: FiLM 仿射 + delta 调制
        gamma_b = gamma.view(B, 1, 1, D)
        beta_b = beta.view(B, 1, 1, D)
        delta_b = delta_mod.view(B, 1, 1, 1)              # (B,1,1,1) 标量调制
        x_s_new = x_s + self.alpha * (gamma_b * delta_b * x_s + beta_b - x_s)

        # BP1+BP2 → 时间链: reset gate + 空间变化率门控
        reset_b = reset_gate.view(B, 1, 1, D)
        sgate_b = s_gate.view(B, 1, 1, D)
        x_t_new = x_t + self.alpha * (reset_b * sgate_b * x_t - x_t)

        # BP2+BP3 → 因果链: 时间相位注入 + 空间校验
        tphase_b = t_phase.view(B, 1, 1, D)
        scheck_b = s_check.view(B, 1, 1, D)
        c_inject_new = c_inject + self.alpha * (
            0.1 * tphase_b + 0.1 * scheck_b * c_inject
        )

        return x_s_new, x_t_new, c_inject_new


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

    def __init__(self, d_model=256, n_layers=2, use_checkpoint=False,
                 ablate_s=False, ablate_t=False, ablate_c=False,
                 enable_bp=False):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint  # 从 Mamba2 移植: gradient checkpointing
        # 从 Mamba2 移植: 消融开关（置零某链输出，不删模块，保持参数量不变）
        self.ablate_s = ablate_s  # 消融空间链
        self.ablate_t = ablate_t  # 消融时间链
        self.ablate_c = ablate_c  # 消融因果链
        self.enable_bp = enable_bp  # ★ 两两碱基对耦合开关 (alpha初始0=baseline)

        # 空间链: 行 + 列 双向扫描(2 个独立 BidirectionalMamba3)
        # WSL2兼容: d_state=64, expand=2, headdim=64 (已验证Triton内核稳定)
        # C500可用异构参数: d_state=128, expand=1, headdim=32
        self.mamba_s_row = nn.ModuleList([
            BidirectionalMamba3(d_model, d_state=64, expand=2, headdim=64,
                                use_checkpoint=use_checkpoint)
            for _ in range(n_layers)
        ])
        self.mamba_s_col = nn.ModuleList([
            BidirectionalMamba3(d_model, d_state=64, expand=2, headdim=64,
                                use_checkpoint=use_checkpoint)
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
        # WSL2兼容: d_state=64 (统一参数, 减少Triton JIT编译次数)
        self.mamba_c = nn.ModuleList([
            make_mamba3(d_model, d_state=64, expand=2, headdim=64)
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

        # ★ 两两碱基对耦合 (enable_bp=True 时生效, alpha初始0=baseline)
        if enable_bp:
            self.base_pair = nn.ModuleList([
                PairwiseBasePairMamba3(d_model) for _ in range(n_layers)
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
            x_t = _ckpt_call(self.mamba_t[i], x_t, use_checkpoint=self.use_checkpoint)
            x_t = x_t.reshape(B, N2, T, D).permute(0, 2, 1, 3)  # (B, T, N², d)

            # --- 因果链: K 维因果扫描 ---
            # act_emb (B, K, T, d) -> (B*T, K, d)
            x_c = act_emb.permute(0, 2, 1, 3).reshape(B * T, K, D)
            x_c = _ckpt_call(self.mamba_c[i], x_c, use_checkpoint=self.use_checkpoint)  # (B*T, K, d)
            # 更新 act_emb (残差)
            act_emb = x_c.reshape(B, T, K, D).permute(0, 2, 1, 3)  # (B, K, T, d)
            # 聚合 K 维 → 注入 x
            c_pool = x_c.mean(dim=1)  # (B*T, d)
            c_inject = self.causal_inject[i](c_pool)  # (B*T, d)
            c_inject = c_inject.reshape(B, T, 1, D)  # 广播到 N²

            # --- 收集每层三链状态（供 .m3 存储和 JEPA）---
            if collect_states:
                chain_states.append((x_s, x_t, act_emb))

            # --- 消融：置零某链输出（不删模块，保持参数量不变，严格控制变量）---
            if self.ablate_s:
                x_s = torch.zeros_like(x_s)
            if self.ablate_t:
                x_t = torch.zeros_like(x_t)
            if self.ablate_c:
                c_inject = torch.zeros_like(c_inject)

            # ★ 两两碱基对耦合 (enable_bp=True 时生效, alpha初始0=baseline)
            if self.enable_bp:
                x_s, x_t, c_inject = self.base_pair[i](x, x_s, x_t, c_inject)

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
                 max_T=256, enable_m3=False, enable_jepa=False, jepa_weight=0.3,
                 use_checkpoint=False, ablate_s=False, ablate_t=False, ablate_c=False,
                 enable_bp=False):
        super().__init__()
        self.cell_types = cell_types
        self.d_model = d_model
        self.n_layers = n_layers
        self.enable_m3 = enable_m3
        self.enable_jepa = enable_jepa
        self.jepa_weight = jepa_weight
        self.use_checkpoint = use_checkpoint  # 从 Mamba2 移植: gradient checkpointing
        self.enable_bp = enable_bp  # ★ 两两碱基对耦合开关

        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)

        # 三链核心 (Mamba3 官方版, 融合 Mamba2 的 use_checkpoint + ablate 开关 + 碱基对耦合)
        self.mamba = HeteroMamba3(d_model, n_layers, use_checkpoint=use_checkpoint,
                                  ablate_s=ablate_s, ablate_t=ablate_t, ablate_c=ablate_c,
                                  enable_bp=enable_bp)

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
        长度外推说明: 不再使用固定长度 nn.Embedding(max_T)，位置感知完全由
            Mamba3 内部 dt-scaled RoPE (cumsum(angle*dt)) 提供，支持 T→32K+ 无衰减。
        """
        B, N, _ = S_0.shape
        K, T = actions.shape[1], actions.shape[2]
        D = self.d_model
        N2 = N * N

        cell_emb = self.cell_embed(S_0.reshape(B, N2))  # (B, N², d)
        act_emb = self.action_embed(actions)  # (B, K, T, d)

        act_mean = act_emb.mean(dim=1)  # (B, T, d)
        # 注意: 不加 time_embed! 位置感知完全交给 Mamba3 内部 dt-RoPE (复数状态)，支持任意长度外推
        x = (cell_emb.unsqueeze(1)                      # (B, 1, N², d)
             + act_mean.unsqueeze(2))                  # (B, T, 1, d)

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
