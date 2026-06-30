"""ThreeChainMamba2 + 跨链碱基对 v2: 3链之间加横档防散架

对照目的:
    验证"跨链碱基对"(用户直觉: 3链旋转没碱基对会散架)能否提升 OOD 长程外推。
    baseline = ThreeChainMamba2 (无碱基对)
    treatment = ThreeChainMamba2BPv2 (跨链碱基对)

与 v1 (自反分身, 已失败) 的核心区别:
    v1 (失败): 碱基对加在单链内部 (ComplementGate 通道互补 + AntiSymmetricHead)
              → 给一条链戴枷锁, 损害 OOD -5.6%
    v2 (本文件): 碱基对加在 3 链之间 (CrossChainBasePair 拉力)
              → 横档把 3 链绑一起转, 防止飘散

设计原理 (用户直觉翻译):
    "3链是个旋转的, 没有碱基对这3链转起来就散架了"
    → 3 链在长序列中各自演化, 逐渐失去同步 (散架)
    → 碱基对 = 每条链向共享状态 x (螺旋轴) 的弹力拉力
    → 拉力门控 + 可学习强度 alpha (初始0 = baseline, 训练中逐渐增强)

    类比 DNA:
        x = 螺旋中心轴 (3链共享状态)
        3 链输出 = 3 根螺旋臂 (空间/时间/因果)
        CrossChainBasePair = 臂到轴的弹力 (碱基对横档)
        alpha = 弹力强度 (学习)
        gate = 哪些维度需要拉 (选择性同步)

参数约束: 同 three_chain_mamba2.py (d_conv∈{2,3,4}, headdim∈{32,64})
"""
import math

import torch
import torch.nn as nn

from .common import balanced_ce_loss
from .three_chain_mamba2 import (
    make_mamba2, BidirectionalMamba2,
)


# ========== 跨链碱基对: 3链向共享状态靠拢 ==========

class CrossChainBasePair(nn.Module):
    """跨链碱基对: 每层融合前, 3链输出向共享状态 x 拉拢, 防止长序列中各自飘散。

    forward(x, x_s, x_t, c_inject) → (x_s', x_t', c_inject'):
        x_s'   = x_s   + alpha * gate(x_s)   * (x        - x_s)      # 空间链向轴靠拢
        x_t'   = x_t   + alpha * gate(x_t)   * (x        - x_t)      # 时间链向轴靠拢
        c_inj' = c_inj + alpha * gate(c_inj) * (x.mean   - c_inj)     # 因果链向轴均值靠拢

    初始化: alpha=0 (恒等变换 = baseline), gate 权重=0 (sigmoid(0)=0.5 中性)
    → 训练开始时模型行为与 baseline 完全一致, 公平对照
    → 训练中 alpha 自学拉力强度, gate 自学哪些维度需要同步
    """

    def __init__(self, d_model):
        super().__init__()
        # 可学习拉力强度, 初始 0 → 开始时无碱基对 (= baseline)
        self.alpha = nn.Parameter(torch.zeros(1))
        # 共享门控 (3 链共用一个 gate, 省参数)
        self.gate = nn.Linear(d_model, d_model)
        # 零初始化 → sigmoid(0) = 0.5 (中性起点)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, x_s, x_t, c_inject):
        """
        x:        (B, T, N², d)  共享状态 (螺旋轴)
        x_s:      (B, T, N², d)  空间链输出
        x_t:      (B, T, N², d)  时间链输出
        c_inject: (B, T, 1, d)   因果链输出 (广播到 N²)

        Returns: (x_s', x_t', c_inject') 同形状, 被碱基对拉力修正
        """
        # 空间链: 向 x 靠拢
        g_s = torch.sigmoid(self.gate(x_s))                 # (B,T,N²,d) 哪些维度需要同步
        x_s = x_s + self.alpha * g_s * (x - x_s)            # 弹簧拉力

        # 时间链: 向 x 靠拢
        g_t = torch.sigmoid(self.gate(x_t))
        x_t = x_t + self.alpha * g_t * (x - x_t)

        # 因果链: 向 x 的 cell 均值靠拢 (c_inject 是广播的, 用 x 的均值匹配)
        x_mean = x.mean(dim=2, keepdim=True)                # (B,T,1,d) 螺旋轴的因果视角
        g_c = torch.sigmoid(self.gate(c_inject))            # (B,T,1,d)
        c_inject = c_inject + self.alpha * g_c * (x_mean - c_inject)

        return x_s, x_t, c_inject


# ========== A 版本核心 + 跨链碱基对: HeteroMamba2BPv2 ==========

class HeteroMamba2BPv2(nn.Module):
    """A 版本三链异构核心 + 跨链碱基对: 每层 3 链扫描后、融合前加横档。

    与 HeteroMamba2 的唯一区别:
        每层融合前多一步: x_s, x_t, c_inject = base_pair[i](x, x_s, x_t, c_inject)
    其余完全一致 (公平对照, 单变量实验)。
    """

    def __init__(self, d_model=256, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # 空间链: 行 + 列 双向扫描 (同 HeteroMamba2)
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

        # 时间链: 因果 Mamba2 (同 HeteroMamba2)
        self.mamba_t = nn.ModuleList([
            make_mamba2(d_model, d_state=64, d_conv=4, expand=2, headdim=64)
            for _ in range(n_layers)
        ])

        # 因果链: K 维因果扫描 (同 HeteroMamba2)
        self.mamba_c = nn.ModuleList([
            make_mamba2(d_model, d_state=32, d_conv=4, expand=2, headdim=64)
            for _ in range(n_layers)
        ])

        # 每层残差融合 LayerNorm (同 HeteroMamba2)
        self.norm_fuse = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.causal_inject = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])

        # ★ 新增: 每层一个跨链碱基对 (唯一变量)
        self.base_pair = nn.ModuleList([
            CrossChainBasePair(d_model) for _ in range(n_layers)
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
            # --- 空间链: 行+列双向扫描 (同 baseline) ---
            x_row = x.reshape(B * T, N2, D)
            row_out = self.mamba_s_row[i](x_row)
            x_2d = x.reshape(B, T, N, N, D)
            x_col = x_2d.permute(0, 1, 3, 2, 4).reshape(B * T, N2, D)
            col_out = self.mamba_s_col[i](x_col)
            col_out = col_out.reshape(B, T, N, N, D).permute(0, 1, 3, 2, 4).reshape(B * T, N2, D)
            x_s = self.fusion_s[i](torch.cat([row_out, col_out], dim=-1))
            x_s = x_s.reshape(B, T, N2, D)

            # --- 时间链: 因果 Mamba2 (同 baseline) ---
            x_t = x.permute(0, 2, 1, 3).reshape(B * N2, T, D)
            x_t = self.mamba_t[i](x_t)
            x_t = x_t.reshape(B, N2, T, D).permute(0, 2, 1, 3)

            # --- 因果链: K 维因果扫描 (同 baseline) ---
            x_c = act_emb.permute(0, 2, 1, 3).reshape(B * T, K, D)
            x_c = self.mamba_c[i](x_c)
            act_emb = x_c.reshape(B, T, K, D).permute(0, 2, 1, 3)
            c_pool = x_c.mean(dim=1)
            c_inject = self.causal_inject[i](c_pool)
            c_inject = c_inject.reshape(B, T, 1, D)

            # ★ 跨链碱基对: 3 链输出融合前加横档 (唯一新增, baseline 无此步)
            x_s, x_t, c_inject = self.base_pair[i](x, x_s, x_t, c_inject)

            # --- 残差融合 (同 baseline) ---
            x = self.norm_fuse[i](x + x_s + x_t + c_inject)

        return x, act_emb


# ========== A 版本模型 + 跨链碱基对: ThreeChainMamba2BPv2 ==========

class ThreeChainMamba2BPv2(nn.Module):
    """A 版本 + 跨链碱基对 v2: 统一时空张量三链 Mamba2 + 3链间横档。

    架构与 ThreeChainMamba2 完全一致, 唯一区别:
        HeteroMamba2 → HeteroMamba2BPv2 (每层多一个 CrossChainBasePair)

    因果性: 同 ThreeChainMamba2 (时间因果 + 空间双向 + agent 维因果)
    """

    def __init__(self, cell_types=16, action_dim=5, d_model=256, n_layers=2,
                 max_T=256):
        super().__init__()
        self.cell_types = cell_types
        self.d_model = d_model
        self.n_layers = n_layers

        # embeddings (同 ThreeChainMamba2)
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        self.time_embed = nn.Embedding(max_T, d_model)

        # ★ 三链核心 + 跨链碱基对 (唯一变量)
        self.mamba = HeteroMamba2BPv2(d_model, n_layers)

        # 输出头 (同 ThreeChainMamba2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, cell_types),
        )

    @property
    def mamba_s(self):
        return self.mamba.mamba_s_row[-1].mamba

    @property
    def mamba_c(self):
        return self.mamba.mamba_c[-1]

    @property
    def mamba_t(self):
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

        # --- AnchorInit2: 统一时空张量 (同 baseline) ---
        cell_emb = self.cell_embed(S_0.reshape(B, N2))
        act_emb = self.action_embed(actions)
        time_idx = torch.arange(T, device=S_0.device)
        time_emb = self.time_embed(time_idx)

        act_mean = act_emb.mean(dim=1)
        x = (cell_emb.unsqueeze(1)
             + act_mean.unsqueeze(2)
             + time_emb.view(1, T, 1, D))

        # --- 三链演化 + 跨链碱基对 ---
        x, act_emb_out = self.mamba(x, act_emb)

        # --- 输出头 (同 baseline) ---
        logits = self.head(x)
        logits = logits.view(B, T, N, N, self.cell_types)

        info = {
            "h_last": x[:, -1],
            "act_emb_out": act_emb_out,
            "bp_alpha": [bp.alpha.item() for bp in self.mamba.base_pair],  # 记录拉力强度
        }
        return logits, info

    def loss(self, logits, S_t, info, aux_weight=0.3):
        """同 baseline: balanced_ce_loss, 无额外正则 (公平对照)"""
        S_0 = S_t[:, 0]
        total, loss_info = balanced_ce_loss(logits, S_t, S_0, aux_weight=aux_weight)
        return total, loss_info
