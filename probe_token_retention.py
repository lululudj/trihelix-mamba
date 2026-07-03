"""阶段 3.3: SSM vs Transformer 信号保留对比探针

机制假说:
  H4: SSM 对 t=0 注入的扰动信号, 在 t=150 仍能保留 (敏感度衰减 < 30%)
  H5: Transformer 对 t=0 注入的扰动信号, 在 t>100 急剧衰减 (敏感度衰减 > 70%)

方法:
  对每个 OOD 样本 (T=150, 训练时 T=100):
    1. 跑 clean forward 拿 logits_clean (B, T, N, N, C)
    2. 修改 S_0 某个非零 cell 的类型 (注入扰动信号)
    3. 跑 perturbed forward 拿 logits_pert
    4. 敏感度(t) = ||logits_pert[:,t] - logits_clean[:,t]||_2 / ||logits_clean[:,t]||_2
  画敏感度保留曲线: retention(t) = sensitivity(t) / sensitivity(1) vs t

科学含义:
  - retention 高 → 初始扰动信号在 t 时刻的预测中仍被保留 (状态保持)
  - retention 低 → 扰动信号被稀释/遗忘 (信息衰减)
  - SSM 递推状态保持 vs Transformer 注意力稀释 的直接对照

输出: results_stage3/retention_<label>.npz
  含 sensitivity_curve (t→敏感度), retention_curve (t→保留率), 元数据
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import GridWorldDataset, GroupedByNSampler, collate_fn
from torch.utils.data import DataLoader
from utils import set_seed, get_device, load_config, build_model


def parse_args():
    p = argparse.ArgumentParser(description="阶段 3.3 token 信号保留探针")
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint 路径 (SSM 或 Transformer)")
    p.add_argument("--config", required=True,
                   help="config 路径 (必须与 checkpoint 匹配)")
    p.add_argument("--data_root", default="./data/ood_T150",
                   help="OOD 数据目录 (T=150, 训练时 T=100)")
    p.add_argument("--out", default=None,
                   help="输出 .npz 路径 (默认: results_stage3/retention_<label>.npz)")
    p.add_argument("--label", default=None,
                   help="本次探针的标签 (ssm_30m / transformer_30m)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="batch size (建议 1, 逐样本扰动更干净)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=50,
                   help="最多探针多少样本")
    p.add_argument("--n_perturb", type=int, default=3,
                   help="每样本扰动多少个 cell (取非零 cell 中随机 n 个)")
    return p.parse_args()


def perturb_s0(S_0, n_perturb=3, generator=None):
    """对每个样本的 S_0 扰动: 把 n_perturb 个非零 cell 改成不同类型.

    S_0: (B, N, N) long tensor (cell type IDs)
    返回: S_0_pert (clone, 仅修改选定 cell), perturb_mask (B, N, N) bool 标记被改的位置

    设计:
      - 只改非零 cell (零=空格, 改无意义)
      - 新值 = (原值 + 1) % cell_types (确定性偏移, 避免随机性引入额外噪声)
      - 用 generator 保证可复现
    """
    S_0_pert = S_0.clone()
    B, N, _ = S_0.shape
    cell_types_max = int(S_0.max().item()) + 1
    perturb_mask = torch.zeros_like(S_0, dtype=torch.bool)

    for b in range(B):
        nonzero = (S_0[b] != 0).nonzero(as_tuple=False)  # (M, 2)
        if len(nonzero) == 0:
            continue
        n = min(n_perturb, len(nonzero))
        # 随机选 n 个位置 (用 generator 保证可复现); perm 在 CPU, nonzero 可能在 GPU
        perm = torch.randperm(len(nonzero), generator=generator)[:n]
        selected = nonzero[perm.to(nonzero.device)]  # (n, 2) 索引到 nonzero 所在设备
        for idx in selected:
            r, c = idx[0].item(), idx[1].item()
            orig = S_0[b, r, c].item()
            # 改成不同的 cell type (确定性偏移, 避免随机)
            new_val = (orig + 1) % cell_types_max
            if new_val == 0:
                new_val = 1  # 不改成空格 (避免退化为空)
            S_0_pert[b, r, c] = new_val
            perturb_mask[b, r, c] = True

    return S_0_pert, perturb_mask


@torch.no_grad()
def probe_retention(model, loader, device, max_samples=50, n_perturb=3, seed=0):
    """测量每个 t 的扰动敏感度.

    Returns:
        dict 含:
          sensitivity_curve: {str(t): mean_sensitivity} 敏感度随 t
          retention_curve: {str(t): retention_rate} 保留率 = sens(t)/sens(1)
          n_samples, max_T, n_perturb
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)  # 扰动位置可复现

    sensitivities = defaultdict(list)  # t(1-based) -> list of per-sample sensitivity
    n_samples = 0
    max_T = 0

    for batch in loader:
        if n_samples >= max_samples:
            break

        S_0 = batch["S_0"].to(device)  # (B, N, N) long
        actions = batch["actions"].to(device)

        # 1. Clean forward
        logits_clean, _ = model(S_0, actions)  # (B, T, N, N, C)

        # 2. Perturb S_0
        S_0_pert, mask = perturb_s0(S_0, n_perturb=n_perturb, generator=g)
        n_changed = mask.sum().item()
        if n_changed == 0:
            continue  # 跳过全空样本

        # 3. Perturbed forward
        logits_pert, _ = model(S_0_pert, actions)

        # 4. Sensitivity at each t
        B, T = logits_clean.shape[0], logits_clean.shape[1]
        max_T = max(max_T, T)
        for t in range(T):
            diff = logits_pert[:, t] - logits_clean[:, t]  # (B, N, N, C)
            base = logits_clean[:, t]
            # 相对 L2 范数 (per-sample), 加 epsilon 防除零
            sens = diff.flatten(start_dim=1).norm(dim=-1) / \
                   (base.flatten(start_dim=1).norm(dim=-1) + 1e-8)
            sensitivities[t + 1].extend(sens.cpu().tolist())

        n_samples += B
        if n_samples % 10 == 0:
            print(f"  probed {n_samples}/{max_samples} samples, T={T}", flush=True)

    # 聚合
    sensitivity_curve = {str(t): float(np.mean(v))
                         for t, v in sorted(sensitivities.items()) if len(v) > 0}
    # 保留率 = sens(t) / sens(1)
    sens_1 = sensitivity_curve.get("1", 0)
    retention_curve = {t: (v / sens_1 if sens_1 > 0 else 0)
                       for t, v in sensitivity_curve.items()}

    return {
        "sensitivity_curve": sensitivity_curve,
        "retention_curve": retention_curve,
        "n_samples": n_samples,
        "max_T": max_T,
        "n_perturb": n_perturb,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()

    print(f"=== 阶段 3.3 token 信号保留探针 ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"config: {args.config}")
    print(f"OOD 数据: {args.data_root}")

    # 加载 checkpoint
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = state.get("model_name", "transformer")
    print(f"模型: {model_name}, step={state.get('step')}")

    # 构建 + 加载
    # SSM 需要 enable_m3=false (probe 不需要 chain_states), Transformer 无此开关
    cfg.setdefault("model", {})
    if model_name in ("three_chain_mamba2", "three_chain_mamba2_hta", "three_chain_mamba3"):
        cfg["model"]["enable_m3"] = False
        cfg["model"]["enable_jepa"] = False
        print(f"  (SSM: enable_m3=false, enable_jepa=false)")

    model = build_model(model_name, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params / 1e6:.2f}M")

    ckpt_sd = state["model"]
    missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
    if missing:
        # 过滤 jepa 相关 (已关闭)
        real_missing = [k for k in missing if "jepa" not in k and "m3" not in k]
        if real_missing:
            print(f"  [警告] missing keys: {real_missing}")
        else:
            print(f"  [ok] 忽略 jepa/m3 missing keys (已关闭)")
    if unexpected:
        print(f"  [警告] unexpected keys: {unexpected}")

    # 加载 OOD 数据
    ds = GridWorldDataset(args.data_root)
    sampler = GroupedByNSampler(ds.Ns, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed, Ts=ds.Ts, Ks=ds.Ks)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    print(f"OOD 样本数: {len(ds)}, batch_size={args.batch_size}, max_samples={args.max_samples}")
    print(f"扰动: 每样本改 {args.n_perturb} 个非零 cell")

    # 探针
    results = probe_retention(model, loader, device,
                              max_samples=args.max_samples,
                              n_perturb=args.n_perturb, seed=args.seed)

    # 元数据
    label = args.label or Path(args.checkpoint).parent.name
    results["label"] = label
    results["checkpoint"] = args.checkpoint
    results["config"] = args.config
    results["model_name"] = model_name
    results["params"] = n_params
    results["d_model"] = cfg["model"]["d_model"]
    results["n_layers_cfg"] = cfg["model"]["n_layers"]

    # 输出
    out_path = Path(args.out) if args.out else (
        Path("results_stage3") / f"retention_{label}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: np.array(v, dtype=object) if isinstance(v, dict) else v
                          for k, v in results.items()})
    print(f"\n=== 探针结果 ({label}) ===")
    print(f"样本数: {results['n_samples']}, max_T={results['max_T']}")
    print(f"已写入: {out_path}")

    # 打印关键时间点的敏感度 + 保留率
    sens = results["sensitivity_curve"]
    ret = results["retention_curve"]
    print(f"\n{'t':>5}  {'sensitivity':>12}  {'retention%':>10}")
    for t in [1, 10, 50, 100, 120, 150]:
        k = str(t)
        if k in sens:
            print(f"  {t:>3}  {sens[k]:>12.6f}  {ret[k]*100:>9.1f}%")

    # H4/H5 快速判定 (t=1 → t=150)
    print(f"\n--- 假说判定 (t=1 → t=150) ---")
    s1 = sens.get("1", 0)
    s150 = sens.get("150", 0)
    if s1 > 0:
        retention_150 = s150 / s1
        print(f"  敏感度: t=1 {s1:.6f} → t=150 {s150:.6f}")
        print(f"  保留率 @t=150: {retention_150*100:.1f}%")
        if model_name in ("three_chain_mamba2", "three_chain_mamba2_hta", "three_chain_mamba3"):
            h4_ok = retention_150 > 0.70
            print(f"  → H4 {'支持' if h4_ok else '推翻'}: SSM 信号保留率 "
                  f"{'>70%' if h4_ok else '<70%'} ({retention_150*100:.1f}%)")
        else:
            h5_ok = retention_150 < 0.30
            print(f"  → H5 {'支持' if h5_ok else '推翻'}: Transformer 信号保留率 "
                  f"{'<30%' if h5_ok else '>30%'} ({retention_150*100:.1f}%)")
    print(f"\n(详细对比见 summarize_stage3_3.py)")


if __name__ == "__main__":
    main()
