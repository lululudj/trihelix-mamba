"""负面分身评估: 5 种 ablate 模式对比, 回答 "SDD 上三链 SSM 演化有贡献 vs 全靠 AnchorInit2 先验".

模式:
- normal: 三链全开 (baseline)
- ablate_s: 全程置零空间链
- ablate_t: 全程置零时间链
- ablate_c: 全程置零因果链
- ablate_all: 三链全置零 (只剩 AnchorInit2 + 残差 + LayerNorm)

关键判别 (用 position_iou, 排除"预测空" shortcut):
- 若 ablate_all 后 pos_iou ≈ normal → 三链无贡献 (全靠 AnchorInit2)
- 若 ablate_all 后 pos_iou 显著下降 → 三链有实质贡献
- 各单链 ablate 下降幅度 → 该链贡献大小
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
from eval_ood_advanced import evaluate_advanced


def parse_args():
    p = argparse.ArgumentParser(description="负面分身评估 (5 ablate 模式)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", default="./data/ood_T150_sdd")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_ablate(model, mode):
    """设置三链 ablate 开关."""
    m = model.mamba  # HeteroMamba2
    if mode == "normal":
        m.ablate_s = False; m.ablate_t = False; m.ablate_c = False
    elif mode == "ablate_s":
        m.ablate_s = True;  m.ablate_t = False; m.ablate_c = False
    elif mode == "ablate_t":
        m.ablate_s = False; m.ablate_t = True;  m.ablate_c = False
    elif mode == "ablate_c":
        m.ablate_s = False; m.ablate_t = False; m.ablate_c = True
    elif mode == "ablate_all":
        m.ablate_s = True;  m.ablate_t = True;  m.ablate_c = True
    else:
        raise ValueError(f"unknown mode {mode}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    cell_types = cfg.get("task", {}).get("cell_types", 16)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = state.get("model_name", "unknown")
    print(f"=== 负面分身评估 {model_name} ===")
    print(f"checkpoint: {args.checkpoint} step={state.get('step')}")
    print(f"cell_types={cell_types}, random agent baseline=1/{cell_types-1}="
          f"{1/(cell_types-1):.4f}")

    model = build_model(model_name, cfg).to(device)
    model.load_state_dict(state["model"])

    ds = GridWorldDataset(args.data_root)
    sampler = GroupedByNSampler(ds.Ns, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed, Ts=ds.Ts, Ks=ds.Ks)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    print(f"OOD 样本数: {len(ds)}, max_T={max(ds.Ts)}")

    modes = ["normal", "ablate_s", "ablate_t", "ablate_c", "ablate_all"]
    results = {}
    for mode in modes:
        print(f"\n--- 模式: {mode} ---")
        set_ablate(model, mode)
        metrics = evaluate_advanced(model, loader, device, cell_types)
        results[mode] = metrics
        c = metrics["curves"]
        # 打印关键 t 点
        for t in [50, 100, 150]:
            if str(t) in c:
                x = c[str(t)]
                print(f"  t={t}: ch_acc={x['changed_acc']:.4f} "
                      f"agent_id={x['agent_id_acc']:.4f} "
                      f"pos_iou={x['position_iou']:.4f} "
                      f"enter={x['enter_acc']:.4f} leave={x['leave_acc']:.4f}")

    # 对比汇总
    print("\n" + "=" * 70)
    print("=== 负面分身对比 (position_iou, 排除预测空 shortcut) ===")
    print(f"{'模式':>12} {'pos_iou@100':>12} {'pos_iou@150':>12} "
          f"{'ch_acc@100':>11} {'agent@100':>10}")
    normal_p100 = results["normal"]["curves"].get("100", {}).get("position_iou", 0)
    for mode in modes:
        c = results[mode]["curves"]
        p100 = c.get("100", {}).get("position_iou", 0)
        p150 = c.get("150", {}).get("position_iou", 0)
        ch100 = c.get("100", {}).get("changed_acc", 0)
        a100 = c.get("100", {}).get("agent_id_acc", 0)
        delta = ""
        if mode != "normal" and normal_p100 > 0:
            delta = f" ({(p100-normal_p100)/normal_p100*100:+.1f}%)"
        print(f"{mode:>12} {p100:>12.4f} {p150:>12.4f} {ch100:>11.4f} "
              f"{a100:>10.4f}{delta}")

    # 自动结论
    all_p100 = results["ablate_all"]["curves"].get("100", {}).get("position_iou", 0)
    print(f"\n=== 自动结论 ===")
    if normal_p100 > 0:
        ratio = all_p100 / normal_p100
        print(f"ablate_all / normal pos_iou@100 = {ratio:.2%}")
        if ratio > 0.9:
            print("→ 三链 SSM 演化在 SDD 上【无实质贡献】(pos_iou 保留 >90%),")
            print("  全靠 AnchorInit2 + 残差 + LayerNorm。印证 Stage 3.4 机制结论。")
        elif ratio > 0.5:
            print("→ 三链有【部分贡献】, 但 AnchorInit2 仍是主导。")
        else:
            print("→ 三链 SSM 演化【有实质贡献】(ablate_all 后 pos_iou 大降),")
            print("  SDD 上 SSM 确实在学习动力学。")

    out = args.out or str(Path(args.checkpoint).parent / "negative_shadow.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
