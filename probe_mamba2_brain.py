"""Mamba2 大脑模型诊断: 训练 100 步 + 立即诊断是否退化成"全猜0"。

旧 ThreeChain 5 种子全部退化(非零预测 0/29492, OOD 100% 全猜0)。
本脚本验证新 ThreeChainMamba2 / Lite 是否突破退化。

非退化门:
    1. 训练长度内(t<=100) 非零预测占比 < 0.90 (不能全猜0)
    2. 训练长度内 变化cell acc > 0.30 (比随机 1/16=0.0625 强)
    3. OOD 段(t>100) 非零预测数 > 非零目标数的 5% (不能 OOD 全退化)

用法:
    python probe_mamba2_brain.py --model three_chain_mamba2 --max_steps 100
    python probe_mamba2_brain.py --model three_chain_mamba2_lite --max_steps 100
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import make_loaders, GridWorldDataset, GroupedByNSampler, collate_fn
from utils import set_seed, get_device, load_config, build_model, cosine_lr


def train_steps(model, train_loader, opt, max_steps, device, lr_base, warmup, aux_weight):
    """训练 max_steps 步, 返回最后的 loss。"""
    model.train()
    step = 0
    train_iter = iter(train_loader)
    losses = []
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        S_0 = batch["S_0"].to(device)
        actions = batch["actions"].to(device)
        S_t = batch["S_t"].to(device)

        lr = cosine_lr(step, max_steps, lr_base, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        logits, info = model(S_0, actions)
        try:
            loss, loss_info = model.loss(logits, S_t, info, aux_weight=aux_weight)
        except TypeError:
            loss, loss_info = model.loss(logits, S_t, aux_weight=aux_weight)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        step += 1
        if step % 20 == 0 or step == 1:
            print(f"    step {step:4d}/{max_steps}  loss={loss.item():.4f}  "
                  f"final={loss_info['loss_final']:.4f}  ch_acc={loss_info.get('changed_acc', 0):.4f}")
    return losses


@torch.no_grad()
def diagnose(model, loader, device, split_t=100):
    """诊断模型是否退化成"全猜0"。

    split_t: 训练长度(<=split_t 训练域, >split_t OOD 外推)
    """
    model.eval()
    # 全量收集
    all_pred = []
    all_target = []
    all_S0 = []
    all_T = []

    for batch in loader:
        S_0 = batch["S_0"].to(device)
        actions = batch["actions"].to(device)
        S_t = batch["S_t"].to(device)
        logits, _ = model(S_0, actions)
        pred = logits.argmax(dim=-1)  # (B, T, N, N)
        target = S_t[:, 1:]           # (B, T, N, N)
        all_pred.append(pred.cpu())
        all_target.append(target.cpu())
        all_S0.append(S_0.cpu())
        all_T.append(target.shape[1])

    pred = torch.cat(all_pred, dim=0)        # (N, T, n, n)
    target = torch.cat(all_target, dim=0)
    S_0 = torch.cat(all_S0, dim=0)
    max_T = pred.shape[1]

    results = {}
    for tag, t_range in [("训练长度内(t<={})".format(split_t), (1, split_t)),
                         ("OOD外推(t>{}".format(split_t), (split_t + 1, max_T))]:
        t_lo, t_hi = t_range
        if t_lo > max_T:
            continue
        t_hi = min(t_hi, max_T)
        mask = slice(t_lo - 1, t_hi)  # pred 的 t 从 0 开始
        p = pred[:, mask]
        tgt = target[:, mask]
        s0 = S_0

        n = p.numel()
        zero_ratio = (p == 0).float().mean().item()
        n_nonzero_pred = (p != 0).sum().item()
        n_nonzero_tgt = (tgt != 0).sum().item()
        acc = (p == tgt).float().mean().item()

        changed = (tgt != s0.unsqueeze(1))
        if changed.any():
            changed_acc = (p[changed] == tgt[changed]).float().mean().item()
        else:
            changed_acc = 1.0

        # 预测分布前6类
        pred_dist = torch.bincount(p.flatten(), minlength=16)[:6].tolist()
        tgt_dist = torch.bincount(tgt.flatten(), minlength=16)[:6].tolist()

        is_collapsed = zero_ratio > 0.95 or n_nonzero_pred == 0
        results[tag] = {
            "n": n,
            "zero_ratio": zero_ratio,
            "n_nonzero_pred": n_nonzero_pred,
            "n_nonzero_tgt": n_nonzero_tgt,
            "acc": acc,
            "changed_acc": changed_acc,
            "pred_dist": pred_dist,
            "tgt_dist": tgt_dist,
            "collapsed": is_collapsed,
        }
    return results


def print_diag(results, model_name):
    print(f"\n{'='*60}")
    print(f"  {model_name} 退化诊断")
    print(f"{'='*60}")
    for tag, r in results.items():
        flag = "⚠️ 退化(全猜0)" if r["collapsed"] else "✅ 正常"
        print(f"\n  [{tag}] n={r['n']:,}  {flag}")
        print(f"    预测为0占比 : {r['zero_ratio']:.4f}")
        print(f"    全局 acc    : {r['acc']:.4f}")
        print(f"    变化cell acc: {r['changed_acc']:.4f}  ← 真实能力")
        print(f"    非零预测数  : {r['n_nonzero_pred']:,} / 非零目标数 {r['n_nonzero_tgt']:,}")
        print(f"    预测分布(前6类): {r['pred_dist']}")
        print(f"    目标分布(前6类): {r['tgt_dist']}")


def judge_gate(results):
    """非退化门判定。"""
    gate_pass = True
    reasons = []
    for tag, r in results.items():
        if "训练长度内" in tag:
            if r["zero_ratio"] >= 0.90:
                gate_pass = False
                reasons.append(f"{tag}: 非零预测占比 {1-r['zero_ratio']:.2%} < 10% (全猜0)")
            if r["changed_acc"] < 0.30:
                gate_pass = False
                reasons.append(f"{tag}: 变化cell acc {r['changed_acc']:.4f} < 0.30 (无真实能力)")
        if "OOD" in tag:
            ratio = r["n_nonzero_pred"] / max(1, r["n_nonzero_tgt"])
            if ratio < 0.05:
                gate_pass = False
                reasons.append(f"{tag}: 非零预测/目标 {ratio:.2%} < 5% (OOD 全退化)")
    return gate_pass, reasons


def main():
    p = argparse.ArgumentParser(description="Mamba2 大脑模型退化诊断")
    p.add_argument("--model", required=True,
                   choices=["three_chain_mamba2", "three_chain_mamba2_lite",
                            "three_chain_mamba2_bp", "three_chain"])
    p.add_argument("--config", default="configs/matched_mamba2.yaml")
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ood_root", default="./data/ood_T150")
    p.add_argument("--data_root", default="./data_split")
    args = p.parse_args()

    # three_chain 用 default config
    if args.model == "three_chain":
        args.config = "configs/default.yaml"

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()

    print(f"=== {args.model} 退化诊断  seed={args.seed}  steps={args.max_steps} ===")
    print(f"device: {device}")

    # 构造模型
    model = build_model(args.model, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    # 训练数据
    loaders = make_loaders(args.data_root, batch_size=args.batch_size, seed=args.seed)
    train_loader = loaders["train"]
    print(f"train batches: {len(train_loader)}")

    # 优化器
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    aux_weight = cfg["train"]["loss_aux_weight"]

    # 训练
    print(f"\n--- 训练 {args.max_steps} 步 ---")
    train_steps(model, train_loader, opt, args.max_steps, device,
                cfg["train"]["lr"], cfg["train"]["warmup_steps"], aux_weight)

    # OOD 诊断
    print(f"\n--- OOD 诊断 (T=150, 训练 T=100) ---")
    ds = GridWorldDataset(args.ood_root)
    sampler = GroupedByNSampler(ds.Ns, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed, Ts=ds.Ts, Ks=ds.Ks)
    ood_loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    print(f"OOD 样本数: {len(ds)}, max_T={max(ds.Ts)}")

    results = diagnose(model, ood_loader, device, split_t=100)
    print_diag(results, args.model)

    # 非退化门
    gate_pass, reasons = judge_gate(results)
    print(f"\n{'='*60}")
    print(f"  非退化门: {'✅ PASS' if gate_pass else '❌ FAIL'}")
    print(f"{'='*60}")
    for r in reasons:
        print(f"    - {r}")
    if gate_pass:
        print("\n  ✅ 模型未退化成全猜0, 可以进入正式 benchmark")
    else:
        print("\n  ❌ 模型仍退化, 需进一步调整架构或损失")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
