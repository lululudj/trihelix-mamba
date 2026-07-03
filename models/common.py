"""三链 DNA-Mamba 共享组件

按 spec §2 实现：
- AnchorInit: 统一初始锚定单元（共享投影 → 三路分流）
- Bind: 碱基对绑定算子（门控分支 + CrossAttn 分支）
- Eagle: 鹰眼校验模块（连续两步差超阈值 → 触发回滚标志）
- Fusion: 融合单元，把 Bind 输出投影到 cell 状态分布
- M3SnapshotManager: .m3 快照管理器（占位实现）
- balanced_ce_loss: 平衡奖惩交叉熵（变化 cell 加权，答对奖励答错惩罚）
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("[warn] mamba_ssm 不可用，用 GRU fallback（仅供开发测试，最终结果需用真实 Mamba）")


class _GRUSeq(nn.Module):
    """GRU 包装：只返回 output，不返回 h_n（与 Mamba 接口对齐）。"""

    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


def make_ssm(d_model, n_layers):
    """返回 n_layers 层序列模块，按环境选择 Mamba 或 GRU fallback。"""
    if HAS_MAMBA:
        return nn.Sequential(*[
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
    return nn.Sequential(*[_GRUSeq(d_model) for _ in range(n_layers)])


class CrossAttention(nn.Module):
    """跨轴注意力：Q 来自一个轴，K/V 来自另一个轴。"""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        query: (B, Lq, d)
        key_value: (B, Lkv, d)
        Returns: (B, Lq, d)
        """
        B, Lq, D = query.shape
        Lkv = key_value.shape[1]
        q = self.q(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(key_value).view(B, Lkv, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(key_value).view(B, Lkv, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out(out)


class AnchorInit(nn.Module):
    """统一初始碱基对锚定单元（spec §2.1）。

    所有输入先经过共享锚定层（shared_proj）统一拆解，再分流三路。
    贴合架构图「初始碱基对统一拆分三轴状态」的定义。

    输出 (h_s, h_c, h_t)：
        h_s: (B, N², d)   空间序列（每个 cell 一个向量）
        h_c: (B, K, d)    因果序列（每智能体一个向量，聚合其动作序列）
        h_t: (B, T, d)    时间序列（每步聚合所有 K 的动作）
    """

    def __init__(self, cell_types, action_dim, d_model):
        super().__init__()
        self.cell_embed = nn.Embedding(cell_types, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        # 共享锚定层：三路输入统一经过同一投影再分流
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_c = nn.Linear(d_model, d_model)
        self.proj_t = nn.Linear(d_model, d_model)
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

    def forward(self, S_0, actions):
        """
        S_0: (B, N, N) long
        actions: (B, K, T) long
        """
        B, N, _ = S_0.shape
        K, T = actions.shape[1], actions.shape[2]
        # 空间轴：每个 cell 的 embedding → 共享锚定
        cell_emb = self.cell_embed(S_0.reshape(B, N * N))  # (B, N², d)
        h_s = self.proj_s(self.norm_s(cell_emb))
        # 因果轴：每智能体的动作序列按时间平均 → 共享锚定
        act_emb = self.action_embed(actions)  # (B, K, T, d)
        h_c = self.proj_c(self.norm_c(act_emb.mean(dim=2)))  # (B, K, d)
        # 时间轴：每步所有智能体动作的聚合 → 共享锚定
        act_emb_t = self.action_embed(actions).mean(dim=1)  # (B, T, d)
        h_t = self.proj_t(self.norm_t(act_emb_t))
        return h_s, h_c, h_t


class Bind(nn.Module):
    """碱基对绑定算子（spec §2.2）。

    Bind(h_s, h_c, h_t) = (h_s ⊙ σ(W_s(h_s)) + h_c ⊙ σ(W_c(h_c)) + h_t ⊙ σ(W_t(h_t)))
                       + CrossAttn(h_t, concat(h_s, h_c))

    其中 h_s / h_c 先通过 cross-attention 对齐到 h_t 的序列长度 L_t。
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.W_s = nn.Linear(d_model, d_model)
        self.W_c = nn.Linear(d_model, d_model)
        self.W_t = nn.Linear(d_model, d_model)
        self.align_s = CrossAttention(d_model, n_heads)  # 对齐 h_s → L_t
        self.align_c = CrossAttention(d_model, n_heads)  # 对齐 h_c → L_t
        self.cross_attn = CrossAttention(d_model, n_heads)  # Q=h_t, K/V=concat(h_s, h_c)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_s, h_c, h_t):
        # 对齐到 L_t 长度
        h_s_a = self.align_s(h_t, h_s)  # (B, L_t, d)
        h_c_a = self.align_c(h_t, h_c)
        # 门控分支
        gated = (h_s_a * torch.sigmoid(self.W_s(h_s_a))
                 + h_c_a * torch.sigmoid(self.W_c(h_c_a))
                 + h_t * torch.sigmoid(self.W_t(h_t)))
        # CrossAttn 分支
        kv = torch.cat([h_s, h_c], dim=1)
        cross = self.cross_attn(h_t, kv)
        return self.norm(gated + cross)


class Eagle(nn.Module):
    """鹰眼校验模块（spec §2.3）。

    计算连续两步 Bind 输出的 L2 差，超阈值 τ 则触发回滚标志。
    训练期 τ 用课程学习从 +∞ 退火到目标值。

    注意（实验边界说明）：
        本预实验仅在显存内缓存校验通过的初始隐向量模拟快照回滚,
        完整外置硬盘 .m3 存储方案为后续拓展工程设计。
    """

    def __init__(self):
        super().__init__()
        self.tau = 1e9  # 初始 +∞（关闭校验）

    def set_tau(self, tau):
        self.tau = float(tau)

    def forward(self, h_seq):
        """
        h_seq: (B, T, d)
        Returns:
            rollback_mask: (B, T) bool，第 0 步恒 False
            drift: (B, T-1) float，连续两步差的 L2 范数
        """
        B, T, D = h_seq.shape
        if T < 2:
            zero = torch.zeros(B, T, dtype=torch.bool, device=h_seq.device)
            return zero, torch.zeros(B, 0, device=h_seq.device)
        diff = h_seq[:, 1:] - h_seq[:, :-1]  # (B, T-1, d)
        drift = diff.norm(dim=-1)  # (B, T-1)
        rollback = drift > self.tau
        # 对齐到 T 长度（第 0 步无前序）
        full = torch.cat([torch.zeros_like(rollback[:, :1]), rollback], dim=1)
        return full, drift


class Fusion(nn.Module):
    """融合单元：每步用空间位置查询 Bind 输出，重建网格预测。

    输入:
        h_bind: (B, T, d)   Bind 输出
        h_s: (B, N², d)     空间轴最后状态
    输出: (B, T, N, N, C) logits
    """

    def __init__(self, d_model, cell_types, n_heads):
        super().__init__()
        self.cell_types = cell_types
        self.cross = CrossAttention(d_model, n_heads)  # Q=空间, K/V=Bind 单步
        self.proj = nn.Linear(d_model, cell_types)

    def forward(self, h_bind, h_s):
        B, T, D = h_bind.shape
        N2 = h_s.shape[1]
        N = int(math.isqrt(N2))
        all_logits = []
        for t in range(T):
            h_step = h_bind[:, t:t + 1]  # (B, 1, d)
            out = self.cross(h_s, h_step)  # (B, N², d)
            logits = self.proj(out)  # (B, N², C)
            all_logits.append(logits)
        logits = torch.stack(all_logits, dim=1)  # (B, T, N², C)
        return logits.view(B, T, N, N, self.cell_types)


class BasePairCoupling(nn.Module):
    """DNA 碱基对启发的三链两两互联模块（v2：移除门控）。

    类比 DNA 双螺旋的氢键：两条链既独立又耦合。
    三链（空间 h_s / 因果 h_c / 时间 h_t）两两形成 3 对碱基对（s↔c, s↔t, c↔t），
    通过 cross-attention 双向交换信息。

    v1 用 tanh 可学习门控（初始 0）实现"弱耦合起步"，但实验发现门控从未打开
    （max|gate|=0.0007，门控死锁：∂loss/∂gate ≈ 0，无压力打开），
    导致 BP 层完全空操作、OOD 结果与 ThreeChain 完全一致。
    v2 移除门控，改用标准 transformer 残差风格（固定权重 1.0 + LayerNorm），
    强制 BP 层生效。失去"可学习弱耦合"的优雅性，但换来确定的耦合强度，
    才能真正检验"碱基对耦合是否提升长程外推"。

    序列长度可不同（N²/K/T），cross-attention 天然支持。
    """

    def __init__(self, d_model, n_heads=4, dropout=0.0):
        super().__init__()
        # 6 个单向 cross-attention（3 对 × 双向）
        # s 接收：from c, from t
        self.s_from_c = CrossAttention(d_model, n_heads, dropout)
        self.s_from_t = CrossAttention(d_model, n_heads, dropout)
        # c 接收：from s, from t
        self.c_from_s = CrossAttention(d_model, n_heads, dropout)
        self.c_from_t = CrossAttention(d_model, n_heads, dropout)
        # t 接收：from s, from c
        self.t_from_s = CrossAttention(d_model, n_heads, dropout)
        self.t_from_c = CrossAttention(d_model, n_heads, dropout)
        # LayerNorm 稳定耦合后表示（标准 transformer 残差风格）
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

    def forward(self, h_s, h_c, h_t):
        """
        h_s: (B, N², d)   空间轴
        h_c: (B, K, d)    因果轴
        h_t: (B, T, d)    时间轴
        Returns: 耦合后的 (h_s', h_c', h_t')，形状不变
        """
        # 标准 transformer 残差：h_new = Norm(h + BP(h))
        # 两路 cross-attention 输出求和后注入（类似多头合并）
        bp_s = self.s_from_c(h_s, h_c) + self.s_from_t(h_s, h_t)
        bp_c = self.c_from_s(h_c, h_s) + self.c_from_t(h_c, h_t)
        bp_t = self.t_from_s(h_t, h_s) + self.t_from_c(h_t, h_c)
        h_s_new = self.norm_s(h_s + bp_s)
        h_c_new = self.norm_c(h_c + bp_c)
        h_t_new = self.norm_t(h_t + bp_t)
        return h_s_new, h_c_new, h_t_new


class M3SnapshotManager:
    """ .m3 快照管理器（阶段 A 升级版：内存 dict + 真文件存储）。

    内存模式（save/load/latest_step）：训练期 in-graph dict 模拟快照回滚，
        B 版 ThreeChainMamba2Lite 仍在用，保留不动。
    文件模式（save_to_file/load_from_file）：阶段 A 新增，把每层三链状态
        存到硬盘供阶段 B 外脑（外置 transformer）读取。

    .m3 文件格式（魔数 M3SNP01 = 版本 1）：
        [Magic 8B]    b"M3SNP01\\n"
        [HeaderLen 4B] little-endian uint32，JSON 头的字节长度
        [Header JSON] utf-8 字节，含 N/K/T/d_model/n_layers/step/sample_id/scenario_type
        [Raw bytes]   每层三链状态拼接（float32）：
            layer_0: h_s (N²,d) + h_t (T,d) + h_c (K,d)
            layer_1: ...
    体积约 350KB/快照（d=256, n_layers=2, N=8, T=100, K=8）。
    """

    MAGIC = b"M3SNP01\n"
    VERSION = 1

    def __init__(self):
        self.snapshots = {}

    # ---- 内存模式（B 版 Lite 兼容，不动）----
    def save(self, step, h_state):
        self.snapshots[step] = h_state.detach().clone()

    def load(self, step):
        return self.snapshots[step]

    def latest_step(self):
        return max(self.snapshots.keys()) if self.snapshots else None

    # ---- 文件模式（阶段 A 新增）----
    @staticmethod
    def save_to_file(path, meta, chain_states):
        """写 .m3 文件（原子写）。

        Args:
            path: 输出路径（str 或 Path）
            meta: dict，必须含 N/K/T/d_model/n_layers；可选 step/sample_id/scenario_type
            chain_states: list[dict]，每层 {"h_s": (N²,d), "h_t": (T,d), "h_c": (K,d)}
                          张量会被 detach+cpu+float32+contiguous
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")

        with open(tmp, "wb") as f:
            f.write(M3SnapshotManager.MAGIC)
            header_json = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            f.write(len(header_json).to_bytes(4, "little"))
            f.write(header_json)
            for layer in chain_states:
                for key in ("h_s", "h_t", "h_c"):
                    t = layer[key].detach().cpu().contiguous().float()
                    f.write(t.numpy().tobytes())
        tmp.replace(path)

    @staticmethod
    def load_from_file(path):
        """读 .m3 文件。

        Returns:
            (meta: dict, chain_states: list[dict])
            chain_states[i] = {"h_s": (N²,d) float32, "h_t": (T,d), "h_c": (K,d)}
        """
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != M3SnapshotManager.MAGIC:
                raise ValueError(f"不是 .m3 文件或版本不兼容：magic={magic!r}")
            header_len = int.from_bytes(f.read(4), "little")
            meta = json.loads(f.read(header_len).decode("utf-8"))
            d = int(meta["d_model"])
            N2 = int(meta["N"]) ** 2
            T = int(meta["T"])
            K = int(meta["K"])
            n_layers = int(meta["n_layers"])
            chain_states = []
            for _ in range(n_layers):
                h_s = torch.frombuffer(f.read(N2 * d * 4), dtype=torch.float32).reshape(N2, d).clone()
                h_t = torch.frombuffer(f.read(T * d * 4), dtype=torch.float32).reshape(T, d).clone()
                h_c = torch.frombuffer(f.read(K * d * 4), dtype=torch.float32).reshape(K, d).clone()
                chain_states.append({"h_s": h_s, "h_t": h_t, "h_c": h_c})
        return meta, chain_states


# ========== 损失函数 ==========

def balanced_ce_loss(logits, S_t, S_0, aux_weight=0.3, change_weight=20.0,
                     empty_class_id=0, empty_class_weight=0.1,
                     unchanged_weight=1.0):
    r"""平衡奖惩交叉熵损失（v5：温和防作弊）。

    考试目标：让模型为拿 100 分而努力，而非靠"全猜不变/全猜空格"作弊。

    奖惩机制（v5 调整）：
        1. 变化 cell × change_weight（20）：答对 → loss≈0（奖励）
                                          答错 → 梯度放大 20 倍（督促修正）
        2. 不变 cell × unchanged_weight（1.0）：默认权重，让模型学到"保持 S_0"表示
            （v4 的 0.01 过于激进，导致空间编码崩溃）
        3. cell type 0（空格）× empty_class_weight（0.1）：全局降权
            防止模型学"输出 0"作为退化策略（无论变化/不变都降权）

    参数:
        logits: (B, T, N, N, C)
        S_t: (B, T+1, N, N) long  完整轨迹（含 S_0）
        S_0: (B, N, N) long       初始状态
        aux_weight: 辅助损失权重
        change_weight: 变化 cell 的权重倍数（v5: 20）
        empty_class_id: 空格 cell 的类别 ID（默认 0）
        empty_class_weight: 空格 cell 的降权系数（v5: 0.1）
        unchanged_weight: 不变 cell 的权重（v5: 1.0，恢复默认）
    返回:
        total_loss, info_dict
    """
    target = S_t[:, 1:]  # (B, T, N, N)
    B, T, N, _, C = logits.shape

    # 变化掩码：S_t[t] != S_0
    S_0_exp = S_0.unsqueeze(1)  # (B, 1, N, N)
    changed = (target != S_0_exp).float()  # (B, T, N, N)
    # 变化=20, 不变=1.0（v5 关键：让模型在不变 cell 上学到"保持"表示）
    change_w = changed * change_weight + (1 - changed) * unchanged_weight

    # 类别权重：空格全局降权，防止"全输出 0"局部最优
    class_w = torch.ones(C, device=logits.device)
    class_w[empty_class_id] = empty_class_weight

    # 逐 cell 交叉熵（带 class weight）
    ce = F.cross_entropy(
        logits.reshape(-1, C), target.reshape(-1),
        reduction='none', weight=class_w
    ).view(B, T, N, N)

    # 加权平均（变化加权 × class 加权）
    w_final = change_w[:, -1]
    loss_final = (ce[:, -1] * w_final).sum() / w_final.sum().clamp(min=1)

    if T > 1:
        w_aux = change_w[:, :-1]
        loss_aux = (ce[:, :-1] * w_aux).sum() / w_aux.sum().clamp(min=1)
    else:
        loss_aux = torch.zeros(1, device=logits.device, dtype=loss_final.dtype)

    # 变化 cell 准确率（监控指标，不影响梯度）
    changed_mask = changed.bool()
    if changed_mask.any():
        pred = logits.argmax(dim=-1)
        changed_acc = (pred[changed_mask] == target[changed_mask]).float().mean().item()
    else:
        changed_acc = 1.0

    total = loss_final + aux_weight * loss_aux
    return total, {
        "loss_final": loss_final.item(),
        "loss_aux": loss_aux.item(),
        "changed_acc": changed_acc,
    }


# ========== 辅助预测头（强制三链分工）==========

class SpaceAuxHead(nn.Module):
    def __init__(self, d_model, cell_types, N=None):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, cell_types),
        )
    
    def forward(self, h_s):
        B, N2, D = h_s.shape
        N = int(N2 ** 0.5)
        logits = self.head(h_s)
        return logits.view(B, N, N, -1)


class CausalAuxHead(nn.Module):
    def __init__(self, d_model, max_N=12):
        super().__init__()
        self.max_N2 = max_N * max_N
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.max_N2 * 2),
        )
    
    def forward(self, h_c, N):
        B = h_c.shape[0]
        pooled = h_c.mean(dim=1)
        logits = self.head(pooled)  # (B, max_N2*2)
        N2 = N * N
        logits = logits[:, :N2 * 2].view(B, N, N, 2)
        return logits


class TimeAuxHead(nn.Module):
    def __init__(self, d_model, max_T=100):
        super().__init__()
        self.max_T = max_T
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(self, h_t):
        return self.head(h_t)


# ========== HeteroMamba: 三链异构核心 (v4) ==========

class HeteroMamba(nn.Module):
    """三链异构Mamba: 每条链独立的核心超参数+外围扫描策略
    
    空间链: d_state=64 d_conv=7 expand=1 + 双向交叉扫描
    因果链: d_state=8  d_conv=3 expand=4 + Agent间门控
    时间链: d_state=32 d_conv=5 expand=2 + 多尺度膨胀卷积
    """
    
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        # === 空间链: 大状态+宽卷积+双向扫描 ===
        self.mamba_s_row = Mamba(d_model=d_model, d_state=64, d_conv=4, expand=1)  # d_conv max=4 per causal_conv1d limit
        self.mamba_s_col = Mamba(d_model=d_model, d_state=64, d_conv=4, expand=1)
        self.fusion_s = nn.Linear(d_model * 2, d_model)
        self.norm_s = nn.LayerNorm(d_model)
        
        # === 因果链: 小状态+高扩展+Agent门控 ===
        self.mamba_c = Mamba(d_model=d_model, d_state=8, d_conv=3, expand=4)
        self.agent_gate = nn.Linear(d_model * 2, d_model)
        self.norm_c = nn.LayerNorm(d_model)
        
        # === 时间链: 中状态+中卷积+多尺度 ===
        self.mamba_t = Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2)  # d_conv max=4
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 4]
        ])
        self.fusion_t = nn.Linear(d_model * 3, d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # === v5: 三螺旋碱基对耦合 ===
        self.basepair = TriHelixBasePairCoupling(d_model)
        
        # v6: CTM多频振荡器(三条链各一个频段)
        self.ctm_s = CTM_Oscillator(d_model, "delta")   # 空间=慢波
        self.ctm_c = CTM_Oscillator(d_model, "theta")    # 因果=海马节律
        self.ctm_t = CTM_Oscillator(d_model, "gamma")   # 时间=快波
    
    def forward(self, h_s, h_c, h_t):
        """
        h_s: (B, N^2, d)  空间序列
        h_c: (B, K, d)    因果序列
        h_t: (B, T, d)    时间序列
        Returns: (h_s', h_c', h_t') 形状不变
        """
        B = h_s.shape[0]
        N2 = h_s.shape[1]
        N = int(N2 ** 0.5)
        D = self.d_model
        
        # --- 空间链: 双向交叉扫描 ---
        # 行优先扫描
        row_out = self.mamba_s_row(h_s)  # (B, N^2, d)
        # 列优先扫描: 转置2D网格再展平
        h_s_2d = h_s.view(B, N, N, D)  # (B, N, N, d)
        col_seq = h_s_2d.permute(0, 2, 1, 3).reshape(B, N2, D)  # 列优先展平
        col_out = self.mamba_s_col(col_seq)
        # 融合行+列
        h_s_out = self.fusion_s(torch.cat([row_out, col_out], dim=-1))
        h_s_out = self.norm_s(h_s + h_s_out)
        h_s_out = h_s_out + 0.1 * self.ctm_s(h_s_out)  # CTM Delta 残差
        
        # --- 因果链: Agent间门控 ---
        c_out = self.mamba_c(h_c)  # (B, K, d)
        # 其他agent均值(排除自身)
        agent_sum = c_out.sum(dim=1, keepdim=True)  # (B, 1, d)
        K = h_c.shape[1]
        agent_mean = (agent_sum - c_out) / max(1, K - 1)  # (B, K, d)
        # 门控融合
        gate = torch.sigmoid(self.agent_gate(torch.cat([c_out, agent_mean], dim=-1)))
        h_c_out = self.norm_c(h_c + gate * c_out + (1 - gate) * agent_mean)
        h_c_out = h_c_out + 0.1 * self.ctm_c(h_c_out)  # CTM Theta
        
        # --- 时间链: 多尺度膨胀卷积 ---
        t_out = self.mamba_t(h_t)  # (B, T, d)
        t_conv = t_out.permute(0, 2, 1)  # (B, d, T)
        multi_scale = [conv(t_conv) for conv in self.dilated_convs]  # 3 x (B, d, T)
        t_fused = torch.cat(multi_scale, dim=1).permute(0, 2, 1)  # (B, T, 3d)
        h_t_out = self.fusion_t(t_fused)  # (B, T, d)
        h_t_out = self.norm_t(h_t + h_t_out)
        h_t_out = h_t_out + 0.1 * self.ctm_t(h_t_out)  # CTM Gamma 残差
        
        # --- v5: 碱基对耦合 (三链相位同步) ---
        h_s_out, h_c_out, h_t_out, bp_info = self.basepair(h_s_out, h_c_out, h_t_out)
        
        return h_s_out, h_c_out, h_t_out, bp_info


# ========== TriHelixBasePairCoupling: 三螺旋碱基对耦合 (v5) ==========

class TriHelixBasePairCoupling(nn.Module):
    """三螺旋碱基对两两耦合模块。
    
    三条链(空间/因果/时间)通过局部参数调制实现相位同步,
    不使用任何Attention, 保持O(L)线性复杂度。
    
    三对碱基对:
    1. 空间-时间: Cross-Delta调制 (频率锚定)
    2. 时间-因果: Reset Gate (记忆重置)  
    3. 因果-空间: FiLM仿射变换 (逻辑具身)
    """
    
    def __init__(self, d_model=256):
        super().__init__()
        D = d_model
        
        # === 碱基对1: 空间 <-> 时间 (Cross-Delta) ===
        # 时间->空间: h_time生成dt调制系数
        self.st_t2s_delta = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, 1),
            nn.Sigmoid()
        )
        # 空间->时间: 空间变化率门控时间更新
        self.st_s2t_gate = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
            nn.Sigmoid()
        )
        
        # === 碱基对2: 时间 <-> 因果 (Memory Reset) ===
        # 因果->时间: 因果能量生成Reset Gate
        self.tc_c2t_reset = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
            nn.Sigmoid()
        )
        # 时间->因果: 时间相位拼接作为位置先验
        self.tc_t2c_phase = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
        )
        
        # === 碱基对3: 因果 <-> 空间 (FiLM) ===
        # 因果->空间: 生成gamma和beta做仿射变换
        self.cs_c2s_gamma = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
        )
        self.cs_c2s_beta = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
        )
        # 空间->因果: 空间坐标作为现实校验器
        self.cs_s2c_check = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
            nn.Tanh()
        )
        
        # 归一化层
        self.norm_s = nn.LayerNorm(D)
        self.norm_c = nn.LayerNorm(D)
        self.norm_t = nn.LayerNorm(D)
    
    def forward(self, h_s, h_c, h_t):
        """
        h_s: (B, L_s, d) 空间序列
        h_c: (B, L_c, d) 因果序列  
        h_t: (B, L_t, d) 时间序列
        Returns: (h_s', h_c', h_t') 同形状
        """
        # 池化到全局表示 (B, d)
        g_s = h_s.mean(dim=1)  # 空间全局
        g_c = h_c.mean(dim=1)  # 因果全局
        g_t = h_t.mean(dim=1)  # 时间全局
        
        # === 碱基对1: 空间 <-> 时间 ===
        # 时间调制空间(Cross-Delta): delta_mod = 0.5 + 0.5*sigmoid
        delta_mod = 0.5 + 0.5 * self.st_t2s_delta(g_t)  # (B, 1)
        # 空间变化率门控时间
        s_diff = torch.diff(h_s, dim=1).mean(dim=1)  # (B, d) 空间一阶差分
        s_gate = self.st_s2t_gate(s_diff)  # (B, d)
        
        # === 碱基对2: 时间 <-> 因果 ===
        # 因果能量→Reset Gate (能量越高重置越多)
        c_energy = torch.norm(g_c, dim=-1, keepdim=True) / (g_c.shape[-1] ** 0.5)
        reset_gate = self.tc_c2t_reset(g_c) * c_energy.sigmoid()  # (B, d)
        # 时间相位→因果位置先验
        t_phase = self.tc_t2c_phase(g_t)  # (B, d)
        
        # === 碱基对3: 因果 <-> 空间 ===
        # 因果→FiLM参数
        gamma = self.cs_c2s_gamma(g_c)  # (B, d)
        beta = self.cs_c2s_beta(g_c)    # (B, d)
        # 空间→现实校验
        s_check = self.cs_s2c_check(g_s)  # (B, d)
        
        # === 应用耦合 ===
        # 空间链: delta调制 + FiLM仿射
        h_s_out = h_s * gamma.unsqueeze(1) + beta.unsqueeze(1)
        h_s_out = self.norm_s(h_s_out)
        
        # 因果链: 时间相位注入 + 现实校验抑制
        h_c_out = h_c + t_phase.unsqueeze(1) * 0.1  # 轻量注入
        h_c_out = h_c_out * (1.0 + 0.1 * s_check.unsqueeze(1))  # 空间校验
        h_c_out = self.norm_c(h_c_out)
        
        # 时间链: Reset门控 + 空间差分门控
        h_t_out = h_t * reset_gate.unsqueeze(1)  # 因果重置
        h_t_out = h_t_out * s_gate.unsqueeze(1)   # 空间变化门控
        h_t_out = self.norm_t(h_t_out)
        
        return h_s_out, h_c_out, h_t_out, {
            "delta_mod_mean": delta_mod.mean().item(),
            "reset_gate_mean": reset_gate.mean().item(),
            "s_gate_mean": s_gate.mean().item(),
        }


# ========== CTM: 多频神经振荡内核 (v6) ==========

class CTM_Oscillator(nn.Module):
    """连续时间流形振荡器: 在每个SSM隐状态维度上叠加可学习的多频振荡。
    
    三条链分配不同频段(仿脑波):
      空间链: Delta (0.5-4 Hz) - 慢波, 长程空间依赖
      因果链: Theta (4-8 Hz) - 海马节律, 因果序列绑定
      时间链: Gamma (30-80 Hz) - 快波, 局部时间同步
    
    数学: h_new = h * exp(i*freq*dt) * exp(-damp*dt)
    实际用实数旋转近似: h_new = h * cos(phase) * decay
    """
    
    def __init__(self, d_state, freq_band="theta"):
        super().__init__()
        self.d_state = d_state
        if freq_band == "delta":
            base_freq = 2.0
        elif freq_band == "theta":
            base_freq = 6.0
        elif freq_band == "gamma":
            base_freq = 50.0
        else:
            base_freq = 10.0
        
        self.log_freq = nn.Parameter(torch.full((d_state,), math.log(base_freq)))
        self.log_damp = nn.Parameter(torch.full((d_state,), -1.0))
    
    def forward(self, h):
        """h: (B, L, d_state) 或 (B, d_state)"""
        freq = torch.exp(self.log_freq).to(h.device)
        damp = torch.exp(self.log_damp).to(h.device)
        dt = 0.01  # 归一化时间步
        phase = freq * dt
        decay = torch.exp(-damp * dt)
        h_mod = h * torch.cos(phase) * decay
        return h_mod


# ========== JEPA: 潜空间世界预测引擎 (v6) ==========

class JEPA_Predictor(nn.Module):
    """JEPA潜空间预测器: 在隐状态空间做未来N步推演。
    
    参考LeWorldModel设计:
    - 输入当前融合隐状态
    - GRU递归预测未来隐状态
    - stop-gradient防表征崩塌
    - 余弦相似度损失(非MSE, 保持方向一致性)
    """
    
    def __init__(self, d_state_total, n_future=3, hidden=64):
        super().__init__()
        self.n_future = n_future
        self.predictor = nn.GRUCell(d_state_total, hidden)
        self.proj_out = nn.Linear(hidden, d_state_total)
    
    def forward(self, h_t):
        """h_t: (B, d) 当前融合隐状态
        Returns: (B, n_future, d) 未来隐状态预测"""
        preds = []
        h = torch.zeros(h_t.shape[0], self.predictor.hidden_size, device=h_t.device)
        x = h_t
        for _ in range(self.n_future):
            h = self.predictor(x, h)
            preds.append(self.proj_out(h))
            x = preds[-1]
        return torch.stack(preds, dim=1)
    
    def compute_loss(self, h_pred, h_true):
        """余弦相似度损失: 鼓励方向一致而非逐元素匹配"""
        h_pred_norm = F.normalize(h_pred, dim=-1)
        h_true_norm = F.normalize(h_true, dim=-1)
        cos_sim = (h_pred_norm * h_true_norm).sum(dim=-1)
        return (1.0 - cos_sim).mean()
