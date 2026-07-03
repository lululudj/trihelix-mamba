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
    p.add_argument("--save_m3", default=None,
                   help="保存 OOD 样本三链状态为 .m3 文件到此目录（需模型 enable_m3=True）")
    p.add_argument("--extend_max_T", type=int, default=None,
                   help="极限测试: 强制扩展模型的 time_embed 到此长度 (用于评估 T>训练时max_T). "
                        "旧位置保留 checkpoint 权重, 新位置用均值填充")
    return p.parse_args()


@torch.no_grad()
def evaluate_ood(model, loader, device, save_m3_dir=None, step=0):
    """评估 OOD 长程外推，返回完整 t=1..T 的曲线。

    save_m3_dir: 若非 None，则对每个样本调 model.save_m3 存 .m3 快照（需 enable_m3=True）。
    step: checkpoint 的训练步数，写进 .m3 meta。
    """
    model.eval()
    final_accs = []
    step_accs = defaultdict(list)
    ch_final_accs = []
    ch_step_accs = defaultdict(list)
    max_T = 0
    sample_counter = 0  # 全局样本计数，用于 .m3 文件命名 sample_XXXX.m3

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

        # 阶段 A：按需把每个样本的三链状态存成 .m3（必须在下一次 forward 前调用）
        if save_m3_dir is not None:
            if not getattr(model, "enable_m3", False):
                raise RuntimeError(
                    "--save_m3 需模型 enable_m3=True，请用 configs/matched_mamba2_m3.yaml "
                    "训练的 checkpoint，并配 --config configs/matched_mamba2_m3.yaml")
            N, K, Tm = batch["N"], batch["K"], batch["T"]
            scen = batch.get("scenario_type", "unknown")
            B = S_0.shape[0]
            for b in range(B):
                path = save_m3_dir / f"sample_{sample_counter:04d}.m3"
                model.save_m3(path, sample_id=sample_counter, N=N, K=K, T=Tm,
                             step=step, scenario_type=scen, batch_idx=b)
                sample_counter += 1

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
        "n_m3_saved": sample_counter if save_m3_dir is not None else 0,
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

    # 极限测试: 如指定 --extend_max_T, 覆盖 config 的 max_T
    if args.extend_max_T is not None:
        cfg.setdefault("model", {})["max_T"] = args.extend_max_T
        print(f"[极限测试] 强制 max_T={args.extend_max_T} (扩展 time_embed)")

    model = build_model(model_name, cfg).to(device)
    ckpt_sd = state["model"]

    # 极限测试: 如果 time_embed size 不匹配 (checkpoint 用 max_T=256, 新模型用更大值)
    te_ckpt = ckpt_sd.get("time_embed.weight")
    te_new = model.state_dict().get("time_embed.weight")
    if te_ckpt is not None and te_new is not None and te_ckpt.shape[0] < te_new.shape[0]:
        old_T, d = te_ckpt.shape
        new_T = te_new.shape[0]
        print(f"  扩展 time_embed: {old_T} → {new_T} (旧位置保留, 新位置用均值填充)")
        # 旧位置保留 checkpoint 权重
        te_new[:old_T] = te_ckpt
        # 新位置用旧权重的均值填充 (比随机更稳定)
        te_new[old_T:] = te_ckpt.mean(dim=0, keepdim=True)
        # 更新 checkpoint 的 time_embed (避免 load_state_dict 报错)
        ckpt_sd["time_embed.weight"] = te_new
        # 用 strict=False 加载 (其他参数应匹配)
        missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
        if missing:
            print(f"  [警告] missing keys: {missing}")
        if unexpected:
            print(f"  [警告] unexpected keys: {unexpected}")
    else:
        model.load_state_dict(ckpt_sd)

    # 阶段 A：按需保存 .m3 快照目录
    save_m3_dir = Path(args.save_m3) if args.save_m3 else None
    if save_m3_dir:
        save_m3_dir.mkdir(parents=True, exist_ok=True)
        print(f"将保存 .m3 快照到: {save_m3_dir}")
        if not getattr(model, "enable_m3", False):
            print("[警告] --save_m3 已指定但模型 enable_m3=False，将无法保存。"
                  "请配 --config configs/matched_mamba2_m3.yaml 并用其训练的 checkpoint。")

    # 直接从目录加载（不分 train/val/test）
    ds = GridWorldDataset(args.data_root)
    sampler = GroupedByNSampler(ds.Ns, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed, Ts=ds.Ts, Ks=ds.Ks)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)

    print(f"OOD 样本数: {len(ds)}, max_T={max(ds.Ts)}")
    step = state.get("step", 0)
    metrics = evaluate_ood(model, loader, device, save_m3_dir=save_m3_dir, step=step)
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
    if save_m3_dir and metrics.get("n_m3_saved", 0) > 0:
        n = metrics["n_m3_saved"]
        sizes = [f.stat().st_size for f in save_m3_dir.glob("*.m3")]
        avg_kb = (sum(sizes) / len(sizes) / 1024) if sizes else 0
        total_mb = (sum(sizes) / 1024 / 1024) if sizes else 0
        print(f"已保存 {n} 个 .m3 快照到 {save_m3_dir} "
              f"(平均 {avg_kb:.0f}KB/文件, 共 {total_mb:.1f}MB)")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
