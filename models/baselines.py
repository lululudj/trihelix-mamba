"""6 个基线模型（spec §3）

所有模型参数量对齐到 ThreeChain 的 ±5%。
开发阶段集中到一个文件，交付时按 spec §11 拆分到独立文件。

基线对照假设：
    SingleChain         三链 vs 单链（同参数预算）
    ConcatMamba         正交分解 vs 拼接（同骨干）
    Transformer         Mamba 长程 vs Transformer 参数效率
    GNN                 空间归纳偏置对照
    ThreeChainNoEagle   鹰眼校验增益
    ThreeChainNoBind    Bind 算子增益（替换为拼接）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (AnchorInit, BasePairCoupling, CrossAttention, Fusion,
                     _GRUSeq, make_ssm, balanced_ce_loss)
from .three_chain import ThreeChain

# 纯 Mamba3 单链 baseline (延迟 import 避免循环依赖)
def _make_mamba3_seq(d_model, n_layers):
    """用 Mamba3 堆叠 n_layers 层 (官方Triton优先, 回退mamba3_ref)。"""
    from .three_chain_mamba3 import make_mamba3, Mamba3 as _M3
    if _M3 is None:
        raise RuntimeError("Mamba3 不可用, 无法构造 SingleChainMamba3")
    import torch.nn as nn
    return nn.Sequential(*[
        make_mamba3(d_model, d_state=64, expand=2, headdim=64)
        for _ in range(n_layers)
    ])


def _base_loss(logits, S_t, aux_weight=0.3):
    """平衡奖惩损失：变化 cell 加权 ×5，不变 cell ×1。S_0 从 S_t 提取。"""
    S_0 = S_t[:, 0]
    total, info = balanced_ce_loss(logits, S_t, S_0, aux_weight=aux_weight)
    return total, info


class SingleChain(nn.Module):
    """单链 Mamba：拼接 (S_0, actions) 送单条 Mamba。验证三链 vs 单链。"""

    def __init__(self, cell_types=12, action_dim=5, d_model=256, n_layers=9, **kw):
        super().__init__()
        self.cell_types = cell_types
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        # n_layers 加倍以匹配三链总参数
        self.mamba = make_ssm(d_model, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fusion = Fusion(d_model, cell_types, n_heads=4)

    def forward(self, S_0, actions):
        B, N, _ = S_0.shape
        K, T = actions.shape[1], actions.shape[2]
        cell_emb = self.cell_embed(S_0.reshape(B, N * N))  # (B, N², d)
        act_emb = self.action_embed(actions).mean(dim=1)   # (B, T, d)
        h = torch.cat([cell_emb, act_emb], dim=1)
        h = self.norm(self.mamba(h))
        h_bind = h[:, N * N:]   # (B, T, d)
        h_s = h[:, :N * N]      # (B, N², d)
        logits = self.fusion(h_bind, h_s)
        return logits, {"rollback_mask": None, "drift": None}

    def loss(self, logits, S_t, aux_weight=0.3):
        return _base_loss(logits, S_t, aux_weight)


class SingleChainMamba3(nn.Module):
    """单链 Mamba3：拼接 (S_0, actions) 送单条 Mamba3。验证三链 vs 单链 Mamba3。

    与 SingleChain 唯一区别：底层 SSM 从 Mamba1 -> Mamba3 (复数状态+dt-RoPE)。
    用于隔离 Mamba3 内核 vs Mamba1 的纯效果 (控制变量)。
    """

    def __init__(self, cell_types=16, action_dim=5, d_model=256, n_layers=3, **kw):
        super().__init__()
        self.cell_types = cell_types
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        self.mamba = _make_mamba3_seq(d_model, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fusion = Fusion(d_model, cell_types, n_heads=4)

    def forward(self, S_0, actions):
        B, N, _ = S_0.shape
        K, T = actions.shape[1], actions.shape[2]
        cell_emb = self.cell_embed(S_0.reshape(B, N * N))  # (B, N², d)
        act_emb = self.action_embed(actions).mean(dim=1)   # (B, T, d)
        h = torch.cat([cell_emb, act_emb], dim=1)
        h = self.norm(self.mamba(h))
        h_bind = h[:, N * N:]   # (B, T, d)
        h_s = h[:, :N * N]      # (B, N², d)
        logits = self.fusion(h_bind, h_s)
        return logits, {"rollback_mask": None, "drift": None}

    def loss(self, logits, S_t, aux_weight=0.3):
        return _base_loss(logits, S_t, aux_weight)


class ConcatMamba(nn.Module):
    """拼接 Mamba：三轴 embedding 拼接后送单条 Mamba。验证正交分解 vs 拼接。"""

    def __init__(self, cell_types=12, action_dim=5, d_model=256, n_layers=9, **kw):
        super().__init__()
        self.cell_types = cell_types
        self.anchor = AnchorInit(cell_types, action_dim, d_model)
        self.mamba = make_ssm(d_model, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fusion = Fusion(d_model, cell_types, n_heads=4)

    def forward(self, S_0, actions):
        h_s, h_c, h_t = self.anchor(S_0, actions)
        h = torch.cat([h_s, h_c, h_t], dim=1)  # (B, N²+K+T, d)
        h = self.norm(self.mamba(h))
        N2 = h_s.shape[1]
        K = h_c.shape[1]
        h_s_new = h[:, :N2]
        h_t_new = h[:, N2 + K:]
        logits = self.fusion(h_t_new, h_s_new)
        return logits, {"rollback_mask": None, "drift": None}

    def loss(self, logits, S_t, aux_weight=0.3):
        return _base_loss(logits, S_t, aux_weight)


class TransformerBaseline(nn.Module):
    """标准 Transformer encoder 基线。验证 Mamba 长程 vs Transformer 参数效率。"""

    def __init__(self, cell_types=12, action_dim=5, d_model=256, n_layers=5, n_heads=4, **kw):
        super().__init__()
        self.cell_types = cell_types
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fusion = Fusion(d_model, cell_types, n_heads=4)

    def forward(self, S_0, actions):
        B, N, _ = S_0.shape
        K, T = actions.shape[1], actions.shape[2]
        cell_emb = self.cell_embed(S_0.reshape(B, N * N))
        act_emb = self.action_embed(actions).mean(dim=1)
        h = torch.cat([cell_emb, act_emb], dim=1)
        h = self.norm(self.transformer(h))
        h_bind = h[:, N * N:]
        h_s = h[:, :N * N]
        logits = self.fusion(h_bind, h_s)
        return logits, {"rollback_mask": None, "drift": None}

    def loss(self, logits, S_t, aux_weight=0.3):
        return _base_loss(logits, S_t, aux_weight)


class GNNBaseline(nn.Module):
    """GNN 基线：3 层消息传递（用 Conv2d 实现，避免 torch_geometric 依赖）。"""

    def __init__(self, cell_types=12, action_dim=5, d_model=256, n_layers=5, n_heads=4, **kw):
        super().__init__()
        self.cell_types = cell_types
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        self.gnn_layers = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.temporal = _GRUSeq(d_model)
        self.fusion = Fusion(d_model, cell_types, n_heads=4)

    def forward(self, S_0, actions):
        B, N, _ = S_0.shape
        K, T = actions.shape[1], actions.shape[2]
        cell_emb = self.cell_embed(S_0)               # (B, N, N, d)
        h = cell_emb.permute(0, 3, 1, 2)             # (B, d, N, N)
        for layer in self.gnn_layers:
            h = F.relu(layer(h))
        h_s = h.permute(0, 2, 3, 1).reshape(B, N * N, -1)  # (B, N², d)
        act_emb = self.action_embed(actions).mean(dim=1)     # (B, T, d)
        h_t = self.norm(self.temporal(act_emb))
        logits = self.fusion(h_t, h_s)
        return logits, {"rollback_mask": None, "drift": None}

    def loss(self, logits, S_t, aux_weight=0.3):
        return _base_loss(logits, S_t, aux_weight)


class ThreeChainNoEagle(ThreeChain):
    """ThreeChain 移除 Eagle 校验。验证鹰眼校验增益。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eagle.set_tau(1e9)  # 始终 +∞，不触发回滚

    def forward(self, S_0, actions):
        logits, info = super().forward(S_0, actions)
        # 覆盖 rollback_mask 全 False（保留 drift 用于诊断）
        if info["rollback_mask"] is not None:
            info["rollback_mask"] = torch.zeros_like(info["rollback_mask"])
        return logits, info


class _ConcatBind(nn.Module):
    """Bind 的简化版：三轴对齐后拼接 + Linear。验证 Bind 算子增益。"""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.align_s = CrossAttention(d_model, n_heads)
        self.align_c = CrossAttention(d_model, n_heads)
        self.proj = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_s, h_c, h_t):
        h_s_a = self.align_s(h_t, h_s)
        h_c_a = self.align_c(h_t, h_c)
        cat = torch.cat([h_s_a, h_c_a, h_t], dim=-1)  # (B, L_t, 3d)
        out = self.proj(cat)
        return self.norm(out)


class ThreeChainNoBind(ThreeChain):
    """ThreeChain 的 Bind 替换为 concat + Linear。验证 Bind 算子增益。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.bind.W_s.in_features
        self.bind = _ConcatBind(d_model)


class ThreeChainBP(ThreeChain):
    """ThreeChain + 碱基对耦合（DNA Base Pair Coupling，v2 移除门控）。

    在三链 Mamba 演化后、Bind 之前插入 BasePairCoupling，
    让三链（空间/因果/时间）两两 cross-attention 互联。

    设计动机：DNA 双螺旋中碱基对氢键让两条链既独立又耦合，
    本模块让三链在演化后通过"碱基对"交换信息，预期在长程/OOD 任务上更稳定。

    v1 用 tanh 门控（初始 0）实现弱耦合，但门控死锁从未打开（max|gate|=0.0007），
    导致 BP 层空操作、OOD 结果与 ThreeChain 完全一致——实验无效。
    v2 移除门控，标准 transformer 残差（固定权重 1.0），强制 BP 层生效，
    才能真正检验"碱基对是否提升长程外推"。

    与 ThreeChain 的唯一区别：多一层 BasePairCoupling（v2 无门控）。
    """

    def __init__(self, *args, bp_heads=4, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.bind.W_s.in_features
        self.bp = BasePairCoupling(d_model, n_heads=bp_heads)

    def forward(self, S_0, actions):
        # 1. Anchor_Init
        h_s, h_c, h_t = self.anchor(S_0, actions)
        # 2. 三链并行演化
        h_s = self.mamba_s(h_s)
        h_c = self.mamba_c(h_c)
        h_t = self.mamba_t(h_t)
        # 2.5 碱基对耦合（v2：移除门控，标准残差）
        h_s, h_c, h_t = self.bp(h_s, h_c, h_t)
        # 3. Bind 融合
        h_bind = self.bind(h_s, h_c, h_t)
        # 4. Eagle 校验
        rollback_mask, drift = self.eagle(h_bind)
        # 5. .m3 快照
        self.m3.save(0, h_bind[:, -1])
        # 6. Fusion
        logits = self.fusion(h_bind, h_s)
        return logits, {"rollback_mask": rollback_mask, "drift": drift}
