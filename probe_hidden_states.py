"""阶段 3 任务 3.1: Hidden state 可视化探针

加载 3.26M baseline checkpoint (enable_m3=true 强制 collect_states=True),
跑 OOD eval (T=150) 抽取每层三链 (h_s/h_t/h_c) 全时序状态, 聚合统计量.

机制假说 (待验证):
  H1: 时间链 h_t 的 norm 在 t>100 (OOD 区) 保持稳定, 承载长程信息
  H2: 空间链 h_s 的 norm 随 t 衰减 (空间信息在长程外推时作用减弱)
  H3: 因果链 h_c 的 norm 与 action 序列强相关, 不随 t 单调变化

输出: results_stage3/probe_hidden_states.npz
  含三链 norm / 余弦相似度 全时序曲线 (t=1..150)
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import GridWorldDataset, GroupedByNSampler, collate_fn
from torch.utils.data import DataLoader
from utils import set_seed, get_device, load_config, build_model


def parse_args():
    p = argparse.ArgumentParser(description="阶段 3.1 hidden state 探针")
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint 路径 (e.g. results/run_three_chain_mamba2_100step_m3subset_seed0/best.pt)")
    p.add_argument("--config", default="configs/matched_mamba2_m3.yaml",
                   help="config (必须 enable_m3=true 以触发 collect_states)")
    p.add_argument("--data_root", default="./data/ood_T150",
                   help="OOD 数据目录 (T=150, 训练时 T=100)")
    p.add_argument("--out", default=None,
                   help="输出 .npz 路径 (默认: results_stage3/probe_<checkpoint_name>.npz)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=100,
                   help="最多探针多少样本 (避免 8G GPU OOM)")
    p.add_argument("--label", default=None,
                   help="本次探针的标签 (用于多 checkpoint 对比, 默认用 checkpoint 目录名)")
    return p.parse_args()


def _ckpt_label(ckpt_path):
    """从 checkpoint 路径提取标签 (e.g. run_30m_seed2_500/best.pt → seed2_500)"""
    name = Path(ckpt_path).parent.name
    # 去掉 run_ 前缀
    if name.startswith("run_"):
        name = name[4:]
    if name.startswith("three_chain_mamba2_"):
        name = name[len("three_chain_mamba2_"):]
    return name


@torch.no_grad()
def probe(model, loader, device, max_samples=100):
    """抽取最后一层三链全时序状态统计.

    Returns:
        dict 含:
          norm_s/t/c: {str(t): mean_norm} 三链 L2 norm 随 t
          cos_st/sc/tc: {str(t): mean_cosine} 三链间余弦相似度随 t
          norm_s_layer0/t_layer0/c_layer0: 第一层 (对照)
          n_samples, max_T
    """
    model.eval()

    # 最后一层统计 (跨样本累加)
    norms_s_last = defaultdict(list)
    norms_t_last = defaultdict(list)
    norms_c_last = defaultdict(list)
    cos_st_last = defaultdict(list)
    cos_sc_last = defaultdict(list)
    cos_tc_last = defaultdict(list)

    # 第一层统计 (对照, 看演化深度差异)
    norms_s_l0 = defaultdict(list)
    norms_t_l0 = defaultdict(list)
    norms_c_l0 = defaultdict(list)

    n_layers = model.n_layers
    n_samples = 0
    max_T = 0

    for batch in loader:
        if n_samples >= max_samples:
            break

        S_0 = batch["S_0"].to(device)
        actions = batch["actions"].to(device)
        S_t = batch["S_t"].to(device)

        # forward (enable_m3=True 自动 collect_states)
        logits, info = model(S_0, actions)
        chain_states = model._last_chain_states  # list of (h_s, h_t, h_c) per layer

        if chain_states is None:
            raise RuntimeError(
                "chain_states 为空 — 检查 enable_m3 是否为 true "
                "(collect_states = enable_m3 or enable_jepa)")

        B, T = logits.shape[0], logits.shape[1]
        K = actions.shape[1]
        N2 = S_0.shape[1] * S_0.shape[2]
        max_T = max(max_T, T)

        # === 第一层 (演化初期) ===
        h_s_l0, h_t_l0, h_c_l0 = chain_states[0]
        # h_s_l0: (B, T, N², d), h_t_l0: (B, T, N², d), h_c_l0: (B, K, T, d)

        # === 最后一层 (最抽象) ===
        h_s, h_t, h_c = chain_states[-1]

        for t in range(T):
            # --- 最后一层 norm (per-sample L2, batch 均值) ---
            # h_s[:,t]: (B, N², d) → flatten(B, N²*d) → norm(B,) → mean
            s_norm = h_s[:, t].flatten(start_dim=1).norm(dim=-1).mean().item()
            t_norm = h_t[:, t].flatten(start_dim=1).norm(dim=-1).mean().item()
            # h_c[:,:,t]: (B, K, d) → flatten(B, K*d) → norm(B,) → mean
            c_norm = h_c[:, :, t].flatten(start_dim=1).norm(dim=-1).mean().item()

            norms_s_last[t].append(s_norm)
            norms_t_last[t].append(t_norm)
            norms_c_last[t].append(c_norm)

            # --- 第一层 norm (对照) ---
            s0_norm = h_s_l0[:, t].flatten(start_dim=1).norm(dim=-1).mean().item()
            t0_norm = h_t_l0[:, t].flatten(start_dim=1).norm(dim=-1).mean().item()
            c0_norm = h_c_l0[:, :, t].flatten(start_dim=1).norm(dim=-1).mean().item()

            norms_s_l0[t].append(s0_norm)
            norms_t_l0[t].append(t0_norm)
            norms_c_l0[t].append(c0_norm)

            # --- 最后一层余弦相似度 (在 d 维上算, batch 均值) ---
            # 三链序列维度不同 (spatial/temporal: N², causal: K), 各自在序列维取均值 → (B, d)
            # 然后在 d 维上算余弦相似度, 避免 shape 不匹配
            s_avg = h_s[:, t].mean(dim=1)  # (B, d)
            t_avg = h_t[:, t].mean(dim=1)  # (B, d)
            c_avg = h_c[:, :, t].mean(dim=1)  # (B, d)

            cos_st = F.cosine_similarity(s_avg, t_avg, dim=-1).mean().item()
            cos_sc = F.cosine_similarity(s_avg, c_avg, dim=-1).mean().item()
            cos_tc = F.cosine_similarity(t_avg, c_avg, dim=-1).mean().item()

            cos_st_last[t].append(cos_st)
            cos_sc_last[t].append(cos_sc)
            cos_tc_last[t].append(cos_tc)

        n_samples += B
        if n_samples % 16 == 0:
            print(f"  probed {n_samples}/{max_samples} samples, T={T}", flush=True)

    # 聚合 (均值)
    def agg(d):
        return {str(t): float(np.mean(v)) for t, v in sorted(d.items())}

    return {
        # 最后一层
        "norm_s_last": agg(norms_s_last),
        "norm_t_last": agg(norms_t_last),
        "norm_c_last": agg(norms_c_last),
        "cos_st_last": agg(cos_st_last),
        "cos_sc_last": agg(cos_sc_last),
        "cos_tc_last": agg(cos_tc_last),
        # 第一层 (对照)
        "norm_s_layer0": agg(norms_s_l0),
        "norm_t_layer0": agg(norms_t_l0),
        "norm_c_layer0": agg(norms_c_l0),
        # 元数据
        "n_samples": n_samples,
        "n_layers": n_layers,
        "max_T": max_T,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()

    # 强制 enable_m3=true (触发 collect_states), enable_jepa=false (避免 missing keys)
    cfg.setdefault("model", {})
    cfg["model"]["enable_m3"] = True
    cfg["model"]["enable_jepa"] = False
    print(f"=== 阶段 3.1 hidden state 探针 ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"config: {args.config} (强制 enable_m3=true, enable_jepa=false)")
    print(f"OOD 数据: {args.data_root}")

    # 加载 checkpoint
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = state.get("model_name", "three_chain_mamba2")
    print(f"模型: {model_name}, step={state.get('step')}")

    model = build_model(model_name, cfg).to(device)
    ckpt_sd = state["model"]

    # strict=False: 兼容 enable_m3=true 但 checkpoint 无 m3 keys 的情况
    # (M3SnapshotManager 无参数, 不影响; jepa 已关, 无 missing)
    missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
    if missing:
        # 过滤掉 jepa 相关 (我们关了 jepa, 不需要)
        real_missing = [k for k in missing if "jepa" not in k]
        if real_missing:
            print(f"  [警告] missing keys (非 jepa): {real_missing}")
        else:
            print(f"  [ok] missing jepa keys (已关闭, 忽略): {[k for k in missing if 'jepa' in k]}")
    if unexpected:
        print(f"  [警告] unexpected keys: {unexpected}")

    # 加载 OOD 数据
    ds = GridWorldDataset(args.data_root)
    sampler = GroupedByNSampler(ds.Ns, batch_size=args.batch_size,
                                shuffle=False, seed=args.seed, Ts=ds.Ts, Ks=ds.Ks)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    print(f"OOD 样本数: {len(ds)}, batch_size={args.batch_size}, max_samples={args.max_samples}")

    # 探针
    results = probe(model, loader, device, max_samples=args.max_samples)

    # 加元数据
    label = args.label or _ckpt_label(args.checkpoint)
    results["label"] = label
    results["checkpoint"] = args.checkpoint
    results["config"] = args.config
    results["d_model"] = cfg["model"]["d_model"]
    results["n_layers_cfg"] = cfg["model"]["n_layers"]

    # 输出
    out_path = Path(args.out) if args.out else (
        Path("results_stage3") / f"probe_{label}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: np.array(v, dtype=object) if isinstance(v, dict) else v
                          for k, v in results.items()})
    print(f"\n=== 探针结果 ({label}) ===")
    print(f"样本数: {results['n_samples']}, max_T={results['max_T']}, n_layers={results['n_layers']}")
    print(f"已写入: {out_path}")

    # 打印关键时间点的 norm (快速 sanity check)
    ns, nt, nc = results["norm_s_last"], results["norm_t_last"], results["norm_c_last"]
    cst = results["cos_st_last"]
    # 用最后可用 t 代替 t=150 (数据索引 0..T-1)
    t_final = str(max(int(k) for k in ns.keys()))
    print(f"\n最后一层三链 norm (关键时间点):")
    print(f"  {'t':>5}  {'h_s':>10}  {'h_t':>10}  {'h_c':>10}  {'cos(s,t)':>10}")
    for t in [1, 50, 100, 120, int(t_final)]:
        k = str(t)
        if k in ns:
            print(f"  {t:>5}  {ns[k]:>10.4f}  {nt[k]:>10.4f}  {nc[k]:>10.4f}  {cst[k]:>10.4f}")

    # H1/H2/H3 快速判定 (t=100 → t=final)
    print(f"\n--- 假说快速判定 (t=100 → t={t_final}) ---")
    t100 = "100"
    if t100 in ns and t_final in ns:
        s_decay = (ns[t_final] - ns[t100]) / ns[t100] * 100
        t_decay = (nt[t_final] - nt[t100]) / nt[t100] * 100
        c_decay = (nc[t_final] - nc[t100]) / nc[t100] * 100
        print(f"  H1 (h_t 稳定): t@100={nt[t100]:.4f} → t@{t_final}={nt[t_final]:.4f}, 变化 {t_decay:+.1f}%")
        print(f"  H2 (h_s 衰减): s@100={ns[t100]:.4f} → s@{t_final}={ns[t_final]:.4f}, 变化 {s_decay:+.1f}%")
        print(f"  H3 (h_c 稳定): c@100={nc[t100]:.4f} → c@{t_final}={nc[t_final]:.4f}, 变化 {c_decay:+.1f}%")
        h1_ok = abs(t_decay) < 15
        h2_ok = s_decay < -15
        h3_ok = abs(c_decay) < 15
        print(f"  → H1 {'支持' if h1_ok else '推翻'}: 时间链在 OOD 区 {'稳定' if h1_ok else '不稳定'} ({t_decay:+.1f}%)")
        print(f"  → H2 {'支持' if h2_ok else '推翻'}: 空间链 {'衰减' if h2_ok else '未显著衰减'} ({s_decay:+.1f}%)")
        print(f"  → H3 {'支持' if h3_ok else '推翻'}: 因果链 {'稳定' if h3_ok else '不稳定'} ({c_decay:+.1f}%)")
    print(f"\n(详细可视化见 plot_hidden_probes.py)")


if __name__ == "__main__":
    main()
