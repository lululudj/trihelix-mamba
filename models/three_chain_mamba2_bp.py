"""ThreeChainMamba2 + 碱基对(自反分身)变体: 对照实验处理组

对照目的:
    验证"自反分身"配对思路能否进一步治好退化、提升真实能力。
    baseline = ThreeChainMamba2 (无配对)
    treatment = ThreeChainMamba2BP (加自反分身)

3 个零件 (从结构 + 输出 + loss 三层破坏全0对称):

1. ComplementGate (结构层, 每层残差后):
    通道两两配对, sigmoid(g) 拆 (g_A, g_B), 强制 g_B = 1 - g_A。
    若 g_A→0 (想关通道), g_B→1 (互补通道强制开)。
    总有一半通道开着, 全0捷径被破坏。
    类比: DNA 碱基 A-T 互补, 一条链关闭, 互补链必开。

2. AntiSymmetricHead (输出层):
    logits 拆 8+8, 强制 logits_B = -logits_A (反对称)。
    全0在反对称空间里是不稳定鞍点, 模型会被推离全0。
    类比: DNA 双链反向互补, 两条链一正一反。

3. diversity_reg (loss 层):
    鼓励预测分布熵高 (多样), 惩罚全0(低熵)。
    类比: 给"摆烂学生"扣分, "认真预测"加分。

参数约束: 同 three_chain_mamba2.py (d_conv∈{2,3,4}, headdim∈{32,64})
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import balanced_ce_loss
from .three_chain_mamba2 import (
    make_mamba2, BidirectionalMamba2,
)


# ========== 零件 1: 互补门控 (结构层) ==========

class ComplementGate(nn.Module):
    """互补门控: 通道两两配对, A+B=1, 破坏全0对称。

    forward: x → x * g_paired
        g = sigmoid(proj(x))           ∈ [0,1]
        g_A, g_B = g[..., :half], g[..., half:]
        g_B = 1 - g_A                  (互补约束)
        g_paired = cat(g_A, g_B)
        return x * g_paired

    防退化原理: 若 g_A→0, g_B→1, 总有一半通道开着。
    """
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.half = d_model // 2
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        g = torch.sigmoid(self.proj(x))               # (..., d) ∈ [0,1]
        g_A = g[..., :self.half]
        g_B = 1.0 - g_A                                # 互补约束
        g_paired = torch.cat([g_A, g_B], dim=-1)      # (..., d)
        return x * g_paired


# ========== 零件 2: 反对称输出头 (输出层) ==========

class AntiSymmetricHead(nn.Module):
    """反对称输出头: logits 拆 half+half, B = -A, 全0变鞍点。

    forward: x → logits
        logits_A = proj(x)         (..., half)
        logits_B = -logits_A        (反对称约束)
        return cat(logits_A, logits_B, dim=-1)   (..., cell_types)

    防退化原理: 全0时 logits_A=logits_B=0, 但反对称空间里这是鞍点
                (任意小扰动都会被放大), 模型被推离全0。
    """
    def __init__(self, d_model, cell_types):
        super().__init__()
        assert cell_types % 2 == 0, f"cell_types 必须偶数, 得到 {cell_types}"
        self.half = cell_types // 2
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model // 2)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_model // 2, self.half)   # 只投影一半

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.proj1(x))
        logits_A = self.proj2(x)                    # (..., half)
        logits_B = -logits_A                         # 反对称约束
        return torch.cat([logits_A, logits_B], dim=-1)   # (..., cell_types)


# ========== 核心带配对的三链: HeteroMamba2BP ==========

class HeteroMamba2BP(nn.Module):
    """带互补门控的三链核心: 每层残差后加 ComplementGate。

    其余结构与 HeteroMamba2 完全一致 (控制变量),
    仅在每层残差融合后多加一个 ComplementGate。
    """
    def __init__(self, d_model=256, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # 空间链: 行 + 列 双向扫描
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

        # 时间链
        self.mamba_t = nn.ModuleList([
            make_mamba2(d_model, d_state=64, d_conv=4, expand=2, headdim=64)
            for _ in range(n_layers)
        ])

        # 因果链
        self.mamba_c = nn.ModuleList([
            make_mamba2(d_model, d_state=32, d_conv=4, expand=2, headdim=64)
            for _ in range(n_layers)
        ])

        self.norm_fuse = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.causal_inject = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])

        # === 新增: 每层互补门控 ===
        self.complement_gate = nn.ModuleList([
            ComplementGate(d_model) for _ in range(n_layers)
        ])

    def forward(self, x, act_emb):
        B, T, N2, D = x.shape
        K = act_emb.shape[1]
        N = int(math.isqrt(N2))

        for i in range(self.n_layers):
            # --- 空间链 ---
            x_row = x.reshape(B * T, N2, D)
            row_out = self.mamba_s_row[i](x_row)
            x_2d = x.reshape(B, T, N, N, D)
            x_col = x_2d.permute(0, 1, 3, 2, 4).reshape(B * T, N2, D)
            col_out = self.mamba_s_col[i](x_col)
            col_out = col_out.reshape(B, T, N, N, D).permute(0, 1, 3, 2, 4).reshape(B * T, N2, D)
            x_s = self.fusion_s[i](torch.cat([row_out, col_out], dim=-1))
            x_s = x_s.reshape(B, T, N2, D)

            # --- 时间链 ---
            x_t = x.permute(0, 2, 1, 3).reshape(B * N2, T, D)
            x_t = self.mamba_t[i](x_t)
            x_t = x_t.reshape(B, N2, T, D).permute(0, 2, 1, 3)

            # --- 因果链 ---
            x_c = act_emb.permute(0, 2, 1, 3).reshape(B * T, K, D)
            x_c = self.mamba_c[i](x_c)
            act_emb = x_c.reshape(B, T, K, D).permute(0, 2, 1, 3)
            c_pool = x_c.mean(dim=1)
            c_inject = self.causal_inject[i](c_pool)
            c_inject = c_inject.reshape(B, T, 1, D)

            # --- 残差融合 ---
            x = self.norm_fuse[i](x + x_s + x_t + c_inject)
            # === 新增: 互补门控 (自反分身) ===
            x = self.complement_gate[i](x)

        return x, act_emb


# ========== 模型: ThreeChainMamba2BP ==========

class ThreeChainMamba2BP(nn.Module):
    """带碱基对(自反分身)的三链 Mamba2。

    相比 ThreeChainMamba2 加 3 零件:
        1. 每层 ComplementGate (互补门控)
        2. AntiSymmetricHead (反对称输出头)
        3. diversity_reg (多样性正则, 在 loss 中)

    用于对照实验: 验证"自反分身"配对思路的价值。
    """
    def __init__(self, cell_types=16, action_dim=5, d_model=256, n_layers=2,
                 max_T=256, div_weight=0.1):
        super().__init__()
        self.cell_types = cell_types
        self.d_model = d_model
        self.n_layers = n_layers
        self.div_weight = div_weight

        # embeddings
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        self.time_embed = nn.Embedding(max_T, d_model)

        # 三链核心 (带互补门控)
        self.mamba = HeteroMamba2BP(d_model, n_layers)

        # 反对称输出头 (替代原 Linear head)
        self.head = AntiSymmetricHead(d_model, cell_types)

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
        B, N, _ = S_0.shape
        K, T = actions.shape[1], actions.shape[2]
        D = self.d_model
        N2 = N * N

        # AnchorInit2 (同 baseline)
        cell_emb = self.cell_embed(S_0.reshape(B, N2))
        act_emb = self.action_embed(actions)
        time_idx = torch.arange(T, device=S_0.device)
        time_emb = self.time_embed(time_idx)

        act_mean = act_emb.mean(dim=1)
        x = (cell_emb.unsqueeze(1)
             + act_mean.unsqueeze(2)
             + time_emb.view(1, T, 1, D))

        # 三链演化 (带互补门控)
        x, act_emb_out = self.mamba(x, act_emb)

        # 反对称输出头
        logits = self.head(x)  # (B, T, N², C)
        logits = logits.view(B, T, N, N, self.cell_types)

        info = {
            "h_last": x[:, -1],
            "act_emb_out": act_emb_out,
        }
        return logits, info

    def loss(self, logits, S_t, info, aux_weight=0.3):
        """balanced_ce_loss + 多样性正则。

        多样性正则: 鼓励预测分布熵高, 惩罚全0(低熵)。
        """
        S_0 = S_t[:, 0]
        total, loss_info = balanced_ce_loss(logits, S_t, S_0, aux_weight=aux_weight)

        # 零件 3: 多样性正则 (惩罚预测熵过低)
        pred_dist = torch.softmax(logits, dim=-1)
        entropy = -(pred_dist * torch.log(pred_dist + 1e-8)).sum(dim=-1).mean()
        div_loss = -entropy  # 负号: 熵越大奖励, 越小惩罚
        total = total + self.div_weight * div_loss
        loss_info["div_loss"] = div_loss.item()
        loss_info["entropy"] = entropy.item()

        return total, loss_info
