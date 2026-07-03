"""高级 OOD 评估: 分解 changed_acc 的 shortcut.
指标:
1. changed_acc (原, 对照)
2. enter_acc: 进入 cell (S_0==0, target!=0) 上 pred==target 比例 (需预测对 agent ID)
3. leave_acc: 离开 cell (S_0!=0, target==0) 上 pred==0 比例 (预测空就对)
4. agent_id_acc: target∈[1,cell_types) 的变化 cell 上 pred==target (排除"预测空"shortcut)
5. position_iou: 预测非零 cell 集合 vs 真实非零 cell 集合的 IoU (位置准确率, 不看 agent ID)
6. n_active_cells: 每步平均非零 cell 数 (看变化量)

关键判别:
- 若 shuffle 的 agent_id_acc ≈ 随机(1/12≈0.083) 而 SSM 显著高 → SSM 学到真实 agent 身份
- 若 shuffle 的 leave_acc ≈ 1 而 enter_acc ≈ 0 → shuffle 只会"预测空"
- position_iou 排除 agent ID, 纯测位置预测能力
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))          # scripts_sdd/ (for eval_ood_advanced import)
sys.path.insert(0, str(Path(__file__).parent.parent))   # 项目根 (for data/utils)
from data.dataset import GridWorldDataset, GroupedByNSampler, collate_fn
from torch.utils.data import DataLoader
from utils import set_seed, get_device, load_config, build_model


def parse_args():
    p = argparse.ArgumentParser(description="高级 OOD 评估 (分解 shortcut)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", default="./data/ood_T150_sdd")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="model", help="标签名 (SSM_10k / shuffle 等)")
    return p.parse_args()


@torch.no_grad()
def evaluate_advanced(model, loader, device, cell_types):
    """计算分解指标."""
    model.eval()
    # 累积器: 每个 t 的指标
    metrics_per_t = defaultdict(lambda: {
        "n_cells": 0, "n_correct": 0,           # 原 changed_acc
        "n_enter": 0, "n_enter_correct": 0,     # 进入 (S_0==0, target!=0)
        "n_leave": 0, "n_leave_correct": 0,      # 离开 (S_0!=0, target==0)
        "n_agent_target": 0, "n_agent_correct": 0,  # agent_id_acc (target∈[1,cell_types))
        "n_pos_intersect": 0, "n_pos_union": 0,  # position IoU
        "n_active_pred": 0, "n_active_target": 0,  # 非零 cell 数
    })
    n_samples = 0

    for batch in loader:
        S_0 = batch["S_0"].to(device)        # (B, N, N)
        actions = batch["actions"].to(device)  # (B, K, T)
        S_t = batch["S_t"].to(device)          # (B, T+1, N, N)
        logits, info = model(S_0, actions)     # (B, T, N, N, C)
        pred = logits.argmax(dim=-1)           # (B, T, N, N)
        target = S_t[:, 1:]                     # (B, T, N, N)
        B, T, N, _ = target.shape
        n_samples += B

        for t in range(T):
            m = metrics_per_t[t + 1]
            pred_t = pred[:, t]      # (B, N, N)
            tgt_t = target[:, t]     # (B, N, N)
            s0 = S_0                  # (B, N, N)

            changed = (tgt_t != s0)   # (B, N, N)
            correct = (pred_t == tgt_t)

            # 原 changed_acc
            m["n_cells"] += changed.sum().item()
            m["n_correct"] += (correct & changed).sum().item()

            # 进入 cell (S_0==0, target!=0)
            enter = (s0 == 0) & (tgt_t != 0)
            m["n_enter"] += enter.sum().item()
            m["n_enter_correct"] += (correct & enter).sum().item()

            # 离开 cell (S_0!=0, target==0)
            leave = (s0 != 0) & (tgt_t == 0)
            m["n_leave"] += leave.sum().item()
            m["n_leave_correct"] += (correct & leave).sum().item()

            # agent_id_acc: target∈[1, cell_types) 的变化 cell
            agent_tgt = changed & (tgt_t >= 1) & (tgt_t < cell_types)
            m["n_agent_target"] += agent_tgt.sum().item()
            m["n_agent_correct"] += (correct & agent_tgt).sum().item()

            # position IoU: 非零 cell 集合 (不看 agent ID)
            pred_nonzero = (pred_t != 0)
            tgt_nonzero = (tgt_t != 0)
            inter = (pred_nonzero & tgt_nonzero).sum().item()
            union = (pred_nonzero | tgt_nonzero).sum().item()
            m["n_pos_intersect"] += inter
            m["n_pos_union"] += union
            m["n_active_pred"] += pred_nonzero.sum().item()
            m["n_active_target"] += tgt_nonzero.sum().item()

    # 汇总曲线
    max_T = max(metrics_per_t.keys())
    out = {"n_samples": n_samples, "max_T": max_T, "curves": {}}
    for t in range(1, max_T + 1):
        m = metrics_per_t[t]
        ch_acc = m["n_correct"] / max(1, m["n_cells"])
        enter_acc = m["n_enter_correct"] / max(1, m["n_enter"])
        leave_acc = m["n_leave_correct"] / max(1, m["n_leave"])
        agent_acc = m["n_agent_correct"] / max(1, m["n_agent_target"])
        pos_iou = m["n_pos_intersect"] / max(1, m["n_pos_union"])
        out["curves"][str(t)] = {
            "changed_acc": ch_acc,
            "enter_acc": enter_acc,
            "leave_acc": leave_acc,
            "agent_id_acc": agent_acc,
            "position_iou": pos_iou,
            "n_changed": m["n_cells"],
            "n_enter": m["n_enter"],
            "n_leave": m["n_leave"],
            "n_agent_tgt": m["n_agent_target"],
            "n_active_pred": m["n_active_pred"] / max(1, n_samples),
            "n_active_target": m["n_active_target"] / max(1, n_samples),
        }
    # 汇总 final (t=150)
    if str(max_T) in out["curves"]:
        out["final"] = out["curves"][str(max_T)]
    # t=100 vs t=150 decay (agent_id_acc)
    c = out["curves"]
    if "100" in c and str(max_T) in c:
        a100 = c["100"]["agent_id_acc"]
        a150 = c[str(max_T)]["agent_id_acc"]
        out["agent_id_decay_pct"] = (a150 - a100) / a100 if a100 > 0 else 0
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    cell_types = cfg.get("task", {}).get("cell_types", 16)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = state.get("model_name", "unknown")
    print(f"=== 高级 OOD 评估 [{args.label}] {model_name} ===")
    print(f"checkpoint: {args.checkpoint} step={state.get('step')}")
    print(f"cell_types={cell_types}, 随机基线 agent_id_acc=1/{cell_types-1}="
          f"{1/(cell_types-1):.4f}")

    model = build_model(model_name, cfg).to(device)
    model.load_state_dict(state["model"])

    ds = GridWorldDataset(args.data_root)
    sampler = GroupedByNSampler(ds.Ns, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed, Ts=ds.Ts, Ks=ds.Ks)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    print(f"OOD 样本数: {len(ds)}, max_T={max(ds.Ts)}")

    metrics = evaluate_advanced(model, loader, device, cell_types)
    metrics["model"] = model_name
    metrics["label"] = args.label
    metrics["cell_types"] = cell_types
    metrics["random_baseline_agent"] = 1 / (cell_types - 1)

    out = args.out or str(Path(args.checkpoint).parent / "ood_metrics_advanced.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print(f"\n=== [{args.label}] 关键指标 (t=100 vs t=150) ===")
    c = metrics["curves"]
    for t in [50, 100, 120, 150]:
        if str(t) in c:
            m = c[str(t)]
            print(f"  t={t:3d}: ch_acc={m['changed_acc']:.4f} | "
                  f"enter={m['enter_acc']:.4f} leave={m['leave_acc']:.4f} | "
                  f"agent_id={m['agent_id_acc']:.4f} "
                  f"(rand={1/(cell_types-1):.4f}) | "
                  f"pos_iou={m['position_iou']:.4f} | "
                  f"n_enter={m['n_enter']} n_leave={m['n_leave']}")
    print(f"\n  agent_id decay (100→{metrics['max_T']}): "
          f"{metrics.get('agent_id_decay_pct', 0)*100:+.1f}%")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
