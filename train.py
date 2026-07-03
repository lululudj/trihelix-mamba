"""训练脚本（spec §6）

用法：
    # 快速验证（100 步）
    python train.py --model three_chain --seed 0 --max_steps 100

    # 正式训练
    python train.py --model three_chain --seed 0
    python train.py --model single_chain --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

# 把当前目录加入 sys.path 以便 import models / data / utils
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import make_loaders
from utils import (
    set_seed, get_device, count_params, load_config, build_model,
    cosine_lr, eagle_tau_schedule, save_checkpoint, load_checkpoint,
    Logger, cell_accuracy,
)


MODELS = ["three_chain", "three_chain_mamba2", "three_chain_mamba2_hta",
          "three_chain_mamba2_lite", "three_chain_mamba2_bp", "three_chain_mamba2_bpv2", "three_chain_mamba3",
          "single_chain", "concat_mamba", "transformer",
          "gnn", "three_chain_no_eagle", "three_chain_no_bind", "three_chain_bp"]


def parse_args():
    p = argparse.ArgumentParser(description="三链验证实验 - 训练")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--data_root", default="./data_split")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--resume", default=None)
    p.add_argument("--shuffle_labels", action="store_true",
                   help="随机标签 sanity check: 训练集 S_t 在组内打乱 (验证 decay 指标有效性)")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """在 val/test 集评估，返回指标 dict（含变化 cell 准确率）。"""
    model.eval()
    accs, drifts, rollbacks = [], [], []
    changed_accs = []
    n = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        S_0 = batch["S_0"].to(device)
        actions = batch["actions"].to(device)
        S_t = batch["S_t"].to(device)
        logits, info = model(S_0, actions)
        acc_final, _ = cell_accuracy(logits, S_t)
        accs.append(acc_final)

        # 变化 cell 准确率（核心区分指标）
        pred = logits.argmax(dim=-1)       # (B, T, N, N)
        target = S_t[:, 1:]                # (B, T, N, N)
        changed = (target != S_0.unsqueeze(1))
        if changed.any():
            ch_acc = (pred[changed] == target[changed]).float().mean().item()
            changed_accs.append(ch_acc)

        # 基线模型返回 None（无 drift/rollback），跳过记录
        if info.get("drift") is not None and info["drift"].numel() > 0:
            drifts.append(float(info["drift"].mean()))
        if info.get("rollback_mask") is not None:
            rollbacks.append(float(info["rollback_mask"].float().mean()))
        n += 1

    out = {"acc_final": sum(accs) / max(1, n)}
    if changed_accs:
        out["changed_acc"] = sum(changed_accs) / len(changed_accs)
    if drifts:
        out["drift_mean"] = sum(drifts) / len(drifts)
    if rollbacks:
        out["rollback_rate"] = sum(rollbacks) / len(rollbacks)
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    set_seed(args.seed)
    device = get_device()

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"results/run_{args.model}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "log.jsonl")

    print(f"=== 训练 {args.model} seed={args.seed} ===")
    print(f"device: {device}")
    print(f"out_dir: {out_dir}")

    # 模型
    model = build_model(args.model, cfg).to(device)
    n_params = count_params(model)
    print(f"params: {n_params / 1e6:.2f}M")

    # 数据
    loaders = make_loaders(args.data_root, batch_size=cfg["train"]["batch_size"],
                           seed=args.seed, shuffle_labels=args.shuffle_labels)
    if "train" not in loaders:
        print("[ERROR] 未找到训练数据。请先执行数据划分：")
        print('  python -c "import sys; sys.path.insert(0,\'.\'); '
              'from data.dataset import split_dataset; '
              'split_dataset(\'./data_cache\', \'./data_split\')"')
        sys.exit(1)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    print(f"loaders: train={len(train_loader)} val={len(val_loader) if val_loader else 0}")

    # 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    # 训练超参
    max_steps = cfg["train"]["max_steps"]
    warmup = cfg["train"]["warmup_steps"]
    lr_base = cfg["train"]["lr"]
    grad_clip = cfg["train"]["grad_clip"]
    aux_weight = cfg["train"]["loss_aux_weight"]

    # Eagle τ 退火
    eagle_cfg = cfg["eagle"]
    has_eagle = hasattr(model, "eagle")

    # 恢复
    start_step = 0
    best_val = -1.0
    if args.resume:
        state = load_checkpoint(args.resume, model, opt, map_location=device)
        start_step = state.get("step", 0)
        best_val = state.get("best_val", -1.0)
        print(f"从 {args.resume} 恢复，step={start_step} best_val={best_val:.4f}")

    # 训练循环
    model.train()
    step = start_step
    t0 = time.time()
    train_iter = iter(train_loader)
    print("training loop start")

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        S_0 = batch["S_0"].to(device)
        actions = batch["actions"].to(device)
        S_t = batch["S_t"].to(device)

        # LR cosine 退火
        lr = cosine_lr(step, max_steps, lr_base, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        # Eagle τ 课程学习退火
        tau = None
        if has_eagle:
            tau = eagle_tau_schedule(
                step, eagle_cfg["tau_init"], eagle_cfg["tau_target"],
                eagle_cfg["tau_warmup_steps"], eagle_cfg["tau_anneal_steps"])
            model.eagle.set_tau(tau)

        # forward
        logits, info = model(S_0, actions)
        try:
            loss, loss_info = model.loss(logits, S_t, info, aux_weight=aux_weight)
        except TypeError:
            loss, loss_info = model.loss(logits, S_t, aux_weight=aux_weight)

        # backward
        opt.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        step += 1

        # 日志
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            record = {
                "step": step,
                "loss": loss_info["loss_final"] + aux_weight * loss_info["loss_aux"],
                "loss_final": loss_info["loss_final"],
                "loss_aux": loss_info["loss_aux"],
                "lr": lr,
                "elapsed": round(elapsed, 1),
            }
            # 变化 cell 准确率（核心区分指标）
            if "changed_acc" in loss_info:
                record["changed_acc"] = loss_info["changed_acc"]
            if tau is not None:
                record["tau"] = tau
            if info.get("drift") is not None and info["drift"].numel() > 0:
                record["drift"] = info["drift"].mean().item()
            if info.get("rollback_mask") is not None:
                record["rollback_rate"] = info["rollback_mask"].float().mean().item()
            logger.log(record)
            tau_str = f" tau={tau:.2f}" if tau is not None else ""
            ch_str = f" ch_acc={loss_info.get('changed_acc', 0):.3f}" if "changed_acc" in loss_info else ""
            print(f"[step {step}/{max_steps}] loss={record['loss']:.4f} "
                  f"final={record['loss_final']:.4f} lr={lr:.2e}{tau_str}{ch_str} "
                  f"elapsed={elapsed:.0f}s")

        # val 评估
        if val_loader and (step % args.eval_every == 0 or step == max_steps):
            metrics = evaluate(model, val_loader, device)
            model.train()
            logger.log({"event": "val", "step": step, **metrics})
            ch_str = f" ch_acc={metrics['changed_acc']:.4f}" if "changed_acc" in metrics else ""
            print(f"  [val step {step}] acc={metrics['acc_final']:.4f}{ch_str}")
            if metrics["acc_final"] > best_val:
                best_val = metrics["acc_final"]
                save_checkpoint({
                    "step": step, "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_val": best_val, "model_name": args.model,
                }, out_dir / "best.pt")
                print(f"  [val] 新最佳 {best_val:.4f}，保存 best.pt")

        # 定期保存
        if step % args.save_every == 0 or step == max_steps:
            save_checkpoint({
                "step": step, "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "best_val": best_val, "model_name": args.model,
            }, out_dir / "final.pt")

    # 最终 summary
    summary = {
        "model": args.model, "seed": args.seed,
        "params": n_params, "max_steps": max_steps,
        "best_val": best_val, "elapsed": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== 完成 best_val={best_val:.4f} ===")


if __name__ == "__main__":
    main()
