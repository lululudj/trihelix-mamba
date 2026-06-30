import sys; sys.stdout.reconfigure(encoding="utf-8")

new_model = """\"\"\"Three-Chain DNA-Mamba v4.0: HeteroMamba核心

v4改动: 用HeteroMamba替换三路vanilla Mamba
- 空间链: d_state=64 d_conv=7 expand=1 + 双向扫描
- 因果链: d_state=8  d_conv=3 expand=4 + Agent门控
- 时间链: d_state=32 d_conv=5 expand=2 + 多尺度膨胀
\"\"\"
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    AnchorInit, Bind, Eagle, Fusion, M3SnapshotManager,
    balanced_ce_loss, SpaceAuxHead, CausalAuxHead, TimeAuxHead,
    HeteroMamba,
)


class ThreeChain(nn.Module):

    def __init__(self, cell_types=12, action_dim=5, d_model=256, n_layers=2,
                 bind_heads=4, fusion_heads=4, N=8, max_T=100):
        super().__init__()
        self.cell_types = cell_types
        self.N = N
        self.anchor = AnchorInit(cell_types, action_dim, d_model)
        # v4: 异构Mamba替换三路vanilla Mamba
        self.mamba = HeteroMamba(d_model)
        self.bind = Bind(d_model, bind_heads)
        self.eagle = Eagle()
        self.fusion = Fusion(d_model, cell_types, fusion_heads)
        self.m3 = M3SnapshotManager()
        self.space_head = SpaceAuxHead(d_model, cell_types, N)
        self.causal_head = CausalAuxHead(d_model, N)
        self.time_head = TimeAuxHead(d_model, max_T)

    def forward(self, S_0, actions):
        h_s, h_c, h_t = self.anchor(S_0, actions)
        # v4: 异构Mamba三链并行演化
        h_s, h_c, h_t = self.mamba(h_s, h_c, h_t)
        space_logits = self.space_head(h_s)
        causal_logits = self.causal_head(h_c)
        time_pred = self.time_head(h_t)
        h_bind = self.bind(h_s, h_c, h_t)
        rollback_mask, drift = self.eagle(h_bind)
        self.m3.save(0, h_bind[:, -1])
        logits = self.fusion(h_bind, h_s)
        aux = {"space_logits": space_logits, "causal_logits": causal_logits, "time_pred": time_pred}
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
        if "time_pred" in aux and aux["time_pred"] is not None:
            B2, T2, _ = aux["time_pred"].shape
            time_target = torch.linspace(0, 1, T2, device=aux["time_pred"].device)
            time_target = time_target.unsqueeze(0).unsqueeze(-1).expand(B2, T2, 1)
            time_loss = F.mse_loss(aux["time_pred"], time_target)
            total = total + time_weight * time_loss
            info["time_loss"] = time_loss.item()
        return total, info
"""

with open("models/three_chain.py", "w", encoding="utf-8") as f:
    f.write(new_model)
print("OK: three_chain.py v4 written")
