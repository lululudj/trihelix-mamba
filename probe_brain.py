"""probe_brain.py — 把 ThreeChain（三链 DNA-Mamba）大脑模型单独拎出来诊断

回答上一轮悬而未决的核心问题：
    模型在 OOD 上 acc=0.8729 是不是真的退化成了"全猜 0（空格）"？
    真实能力（只看变化 cell）到底多少？5 个种子是不是都退化成同一个模式？

用法（WSL，因为依赖 mamba_ssm）：
    cd /mnt/e/three_chain_v3
    python3 probe_brain.py
    # 只跑单种子快速验证：
    python3 probe_brain.py --seeds 42
    # 换 OOD 数据集：
    python3 probe_brain.py --ood data/ood_T200_noise
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import GridWorldDataset, GroupedByNSampler, collate_fn
from utils import set_seed, get_device, load_config, build_model


CKPT_DIR = Path("results_wsl/benchmark_multiseed")
DEFAULT_CFG = "configs/default.yaml"
OOD_DATA = "data/ood_T150"
SEEDS = [42, 123, 456, 789, 1024]
CELL_TYPES = 16  # default.yaml: 空格(0) + 智能体(1-12) + 物品(13-15)


def parse_args():
    p = argparse.ArgumentParser(description="ThreeChain 大脑模型独立诊断")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                   help="逗号分隔的种子列表")
    p.add_argument("--config", default=DEFAULT_CFG)
    p.add_argument("--ood", default=OOD_DATA, help="OOD 数据目录")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def diagnose(model, loader, device, label, max_batches=None):
    """对单个模型做完整诊断。

    返回 dict，含预测分布、各类准确率、是否退化判定。
    """
    model.eval()
    # 累积全部预测与目标（按训练长度内/外分段，看是否外推失败）
    pred_in, pred_out = [], []      # t<=100 / t>100 的预测
    target_in, target_out = [], []
    S0_in_list, S0_out_list = [], []   # 初始状态，按 split 对齐（用于判"变化 cell"）
    n_batch = 0
    for batch in loader:
        S_0 = batch["S_0"].to(device)
        actions = batch["actions"].to(device)
        S_t = batch["S_t"].to(device)
        logits, _ = model(S_0, actions)        # (B, T, N, N, C)
        pred = logits.argmax(dim=-1)           # (B, T, N, N)
        target = S_t[:, 1:]                    # (B, T, N, N)
        T = pred.shape[1]
        # 训练长度 T=100 为界，分段统计外推行为
        split = min(100, T)
        pred_in.append(pred[:, :split].reshape(-1).cpu())
        target_in.append(target[:, :split].reshape(-1).cpu())
        S0_in_list.append(S_0.unsqueeze(1).expand(-1, split, -1, -1).reshape(-1).cpu())
        if T > split:
            out_len = T - split
            pred_out.append(pred[:, split:].reshape(-1).cpu())
            target_out.append(target[:, split:].reshape(-1).cpu())
            S0_out_list.append(S_0.unsqueeze(1).expand(-1, out_len, -1, -1).reshape(-1).cpu())
        n_batch += 1
        if max_batches and n_batch >= max_batches:
            break

    pred_in = torch.cat(pred_in)
    target_in = torch.cat(target_in)
    S0_in = torch.cat(S0_in_list)
    pred_out = torch.cat(pred_out) if pred_out else None
    target_out = torch.cat(target_out) if target_out else None
    S0_out = torch.cat(S0_out_list) if S0_out_list else None

    def stats(pred, target, S0, tag):
        n = pred.numel()
        pred_dist = torch.bincount(pred, minlength=CELL_TYPES).tolist()
        target_dist = torch.bincount(target, minlength=CELL_TYPES).tolist()
        zero_pred_ratio = (pred == 0).float().mean().item()
        acc_global = (pred == target).float().mean().item()
        # 变化 cell（target != S_0）才是真实任务，排除"全猜 0 捡空格"
        changed = (target != S0)
        n_changed = changed.sum().item()
        if n_changed > 0:
            acc_changed = (pred[changed] == target[changed]).float().mean().item()
        else:
            acc_changed = float("nan")
        # 不变 cell（target == S_0）：全猜 0 在空格上会拿高分
        unchanged = ~changed
        acc_unchanged = (pred[unchanged] == target[unchanged]).float().mean().item()
        # 退化判定：>95% 预测为 0
        is_collapsed = zero_pred_ratio > 0.95
        return {
            "tag": tag,
            "n": n,
            "pred_dist": pred_dist,
            "target_dist": target_dist,
            "zero_pred_ratio": zero_pred_ratio,
            "acc_global": acc_global,
            "acc_changed": acc_changed,
            "acc_unchanged": acc_unchanged,
            "n_changed": n_changed,
            "is_collapsed": is_collapsed,
        }

    s_in = stats(pred_in, target_in, S0_in, "训练长度内 t<=100")
    s_out = stats(pred_out, target_out, S0_out, "OOD 外推 t>100") \
        if pred_out is not None else None
    return {"label": label, "in": s_in, "out": s_out}


def print_report(result):
    print(f"\n{'='*70}")
    print(f"  {result['label']}")
    print(f"{'='*70}")
    for s in [result["in"], result["out"]]:
        if s is None:
            continue
        flag = "  ⚠️ 退化(全猜0)" if s["is_collapsed"] else "  ✓ 非退化"
        print(f"\n  [{s['tag']}] n={s['n']:,}{flag}")
        print(f"    预测为0占比 : {s['zero_pred_ratio']:.4f}")
        print(f"    全局 acc    : {s['acc_global']:.4f}  (被空格污染)")
        print(f"    变化cell acc: {s['acc_changed']:.4f}  ← 真实能力")
        print(f"    不变cell acc: {s['acc_unchanged']:.4f}")
        # 简洁分布：只看非0类是否被预测出来
        pd = s["pred_dist"]
        td = s["target_dist"]
        nz_pred = sum(pd[1:])
        nz_target = sum(td[1:])
        print(f"    非零预测数  : {nz_pred:,} / 非零目标数 {nz_target:,}")
        print(f"    预测分布(前6类): {pd[:6]}")
        print(f"    目标分布(前6类): {td[:6]}")


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = get_device()
    cfg = load_config(args.config)
    set_seed(args.seed)

    print(f"=== ThreeChain 大脑模型独立诊断 ===")
    print(f"config: {args.config}  d_model={cfg['model']['d_model']}")
    print(f"OOD: {args.ood}  device: {device}")
    print(f"种子: {seeds}")

    # 加载 OOD 数据（不分 train/val/test，直接读目录）
    ds = GridWorldDataset(args.ood)
    sampler = GroupedByNSampler(ds.Ns, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed,
                                Ts=ds.Ts, Ks=ds.Ks)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn,
                        num_workers=0)
    print(f"OOD 样本数: {len(ds)}, batch 数: {len(sampler)}")

    results = []
    for seed in seeds:
        ckpt = CKPT_DIR / f"three_chain_seed{seed}" / "best.pt"
        if not ckpt.exists():
            print(f"\n[跳过] {ckpt} 不存在")
            continue
        state = torch.load(ckpt, map_location=device, weights_only=False)
        try:
            model = build_model(state["model_name"], cfg).to(device)
            model.load_state_dict(state["model"])
        except Exception as e:
            print(f"\n[加载失败] seed{seed}: {type(e).__name__}: {e}")
            # 尝试另一个 config
            print(f"  尝试 configs/matched_three.yaml ...")
            cfg2 = load_config("configs/matched_three.yaml")
            model = build_model(state["model_name"], cfg2).to(device)
            model.load_state_dict(state["model"])
        label = f"seed{seed}  (step={state.get('step')}, best_val={state.get('best_val'):.4f})"
        result = diagnose(model, loader, device, label)
        print_report(result)
        results.append(result)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 汇总：5 种子是否都退化成同一模式
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  5 种子汇总（OOD 外推段）")
        print(f"{'='*70}")
        print(f"{'seed':<8}{'全局acc':<10}{'变化acc':<10}{'全猜0占比':<12}{'退化?'}")
        for r in results:
            s = r["out"] or r["in"]
            print(f"{r['label'].split()[0]:<8}{s['acc_global']:<10.4f}"
                  f"{s['acc_changed']:<10.4f}{s['zero_pred_ratio']:<12.4f}"
                  f"{'是' if s['is_collapsed'] else '否'}")
        # 关键判定
        collapsed_all = all((r["out"] or r["in"])["is_collapsed"] for r in results)
        accs = [(r["out"] or r["in"])["acc_changed"] for r in results]
        acc_std = float(np.std(accs))
        print(f"\n  结论：{'全部退化成全猜0' if collapsed_all else '未全部退化'}")
        print(f"  变化cell acc 均值={np.mean(accs):.4f}  std={acc_std:.4f}")
        if acc_std < 1e-6:
            print(f"  ⚠️  5种子变化acc std≈0，疑似都退化成同一平凡解（与0.8729逐位相同一致）")


if __name__ == "__main__":
    main()
