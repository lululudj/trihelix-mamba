"""OOD 长程外推评估：T=150（训练时 T=100）。

评估三模型在超出训练长度 50% 的序列上的外推能力。
输出完整 t=1..150 的 changed_acc 曲线 + 外推衰减指标。
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import GridWorldDataset, GroupedByNSampler, collate_fn
from torch.utils.data import DataLoader
from utils import set_seed, get_device, load_config, build_model


def parse_args():
    p = argparse.ArgumentParser(description="OOD 长程外推评估")
    p.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_root", default="./data/ood_T150", help="OOD 数据目录（直接含 .npz）")
    p.add_argument("--out", default=None, help="输出 metrics json 路径")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def evaluate_ood(model, loader, device):
    """评估 OOD 长程外推，返回完整 t=1..T 的曲线。"""
    model.eval()
    final_accs = []
    step_accs = defaultdict(list)
    ch_final_accs = []
    ch_step_accs = defaultdict(list)
    max_T = 0

    for batch in loader:
        S_0 = batch["S_0"].to(device)
        actions = batch["actions"].to(device)
        S_t = batch["S_t"].to(device)
        logits, info = model(S_0, actions)
        pred = logits.argmax(dim=-1)
        target = S_t[:, 1:]
        correct = (pred == target).float()
        acc_per_step = correct.mean(dim=(0, 2, 3))
        T = acc_per_step.shape[0]
        max_T = max(max_T, T)
        final_accs.append(float(acc_per_step[-1]))
        for t in range(1, T + 1):
            step_accs[t].append(float(acc_per_step[t - 1]))

        changed = (target != S_0.unsqueeze(1))
        if changed.any():
            ch_correct = (correct * changed.float()).sum(dim=(0, 2, 3)) / \
                         changed.float().sum(dim=(0, 2, 3)).clamp(min=1)
            ch_final_accs.append(float(ch_correct[-1]))
            for t in range(1, T + 1):
                ch_step_accs[t].append(float(ch_correct[t - 1]))

    acc_curve = {str(t): sum(step_accs[t]) / len(step_accs[t])
                 for t in range(1, max_T + 1) if step_accs[t]}
    ch_curve = {str(t): sum(ch_step_accs[t]) / len(ch_step_accs[t])
                for t in range(1, max_T + 1) if ch_step_accs[t]}

    return {
        "acc_final": sum(final_accs) / max(1, len(final_accs)),
        "acc_curve": acc_curve,
        "changed_acc": sum(ch_final_accs) / max(1, len(ch_final_accs)),
        "changed_acc_curve": ch_curve,
        "n_samples": len(final_accs),
        "max_T": max_T,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = state.get("model_name", "unknown")
    print(f"=== OOD 长程外推评估 {model_name} ===")
    print(f"checkpoint: {args.checkpoint} step={state.get('step')}")
    print(f"OOD 数据: {args.data_root} (T=150, 训练时 T=100)")

    model = build_model(model_name, cfg).to(device)
    model.load_state_dict(state["model"])

    # 直接从目录加载（不分 train/val/test）
    ds = GridWorldDataset(args.data_root)
    sampler = GroupedByNSampler(ds.Ns, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed, Ts=ds.Ts, Ks=ds.Ks)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)

    print(f"OOD 样本数: {len(ds)}, max_T={max(ds.Ts)}")
    metrics = evaluate_ood(model, loader, device)
    metrics["model"] = model_name
    metrics["train_T"] = 100
    metrics["ood_T"] = 150

    # 关键 OOD 指标：t=100 vs t=150 的衰减
    ch_curve = metrics["changed_acc_curve"]
    if "100" in ch_curve and "150" in ch_curve:
        acc_100 = ch_curve["100"]
        acc_150 = ch_curve["150"]
        metrics["ood_decay_100_to_150"] = acc_150 - acc_100
        metrics["ood_decay_pct"] = (acc_150 - acc_100) / acc_100 if acc_100 > 0 else 0

    out = args.out or str(Path(args.checkpoint).parent / "ood_metrics.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== OOD 结果 ({model_name}) ===")
    print(f"acc_final (t=150): {metrics['acc_final']:.4f}")
    print(f"changed_acc (t=150): {metrics['changed_acc']:.4f}")
    if "ood_decay_100_to_150" in metrics:
        print(f"OOD 衰减 (t=100→150): {metrics['ood_decay_100_to_150']:+.4f} "
              f"({metrics['ood_decay_pct']*100:+.1f}%)")
    # 关键节点曲线
    for t in [50, 100, 120, 150]:
        k = str(t)
        if k in ch_curve:
            print(f"  changed_acc@t={t}: {ch_curve[k]:.4f}")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
