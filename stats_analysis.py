"""统计检验模块: t-test + Cohen's d + 95% CI

对比: 三链Mamba3+BP (treatment) vs 三链Mamba3(baseline) vs Transformer vs Mamba3单链
指标: val_ch_acc, ood_ch_acc, ood_decay_pct

用法:
    python stats_analysis.py                    # 读 battle32_results.json, 输出 battle32_stats.json
    python stats_analysis.py <results.json>     # 指定结果文件
"""
import sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict

# 优先用 scipy (t分布更准), 不可用则 numpy fallback (正态近似)
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] scipy 未安装, 用 numpy 正态近似 (n>=3 时不准, 建议 pip install scipy)")


def cohen_d(a, b):
    """Cohen's d 效应量 (a vs b, a=treatment, b=control)。"""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def t_test(a, b):
    """Welch's t-test (不等方差), 返回 (t_stat, p_value, cohen_d, ci_95)。

    a, b: 两组样本 (list/array)
    返回:
        t_stat: t 统计量
        p_value: 双侧 p 值
        cohen_d: 效应量
        ci_95: 均值差的 95% 置信区间 (low, high)
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = cohen_d(a, b)
    diff = float(a.mean() - b.mean())

    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0, d, (diff, diff)

    if HAS_SCIPY:
        t_stat, p_val = sp_stats.ttest_ind(a, b, equal_var=False)
        t_stat, p_val = float(t_stat), float(p_val)
    else:
        # numpy fallback: Welch's t-test + 正态近似 p 值
        n1, n2 = len(a), len(b)
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(v1/n1 + v2/n2)
        t_stat = float((a.mean() - b.mean()) / se) if se > 0 else 0.0
        # 正态近似 (n 较大时准确, n=3 保守)
        from math import erf, sqrt
        p_val = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))

    # 95% CI of mean difference
    se_diff = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    ci = (diff - 1.96*se_diff, diff + 1.96*se_diff)

    return t_stat, p_val, d, (float(ci[0]), float(ci[1]))


def sig_mark(p):
    """显著性标记。"""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def run_statistical_tests(results_path, out_path):
    """对所有场景做 t-test: 三链Mamba3+BP vs 其他3个模型。"""
    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # 聚合: metrics[scen][model][metric] = [v_seed1, v_seed2, v_seed3]
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    bp_alphas = defaultdict(lambda: defaultdict(list))  # 记录BP的alpha最终值
    for key, r in all_results.items():
        if "error" in r:
            continue
        parts = key.split("|")
        scen = parts[0]
        model = r["model"]
        for m in ["val_ch_acc", "ood_ch_acc", "ood_decay_pct"]:
            metrics[scen][model][m].append(r[m])
        if "bp_alpha_final" in r:
            bp_alphas[scen][model].extend(r["bp_alpha_final"])

    treatment = "三链Mamba3+BP"
    controls = ["Transformer", "Mamba3单链", "三链Mamba3"]
    metric_names = ["val_ch_acc", "ood_ch_acc", "ood_decay_pct"]
    metric_labels = {"val_ch_acc": "val_ch", "ood_ch_acc": "ood_ch", "ood_decay_pct": "decay"}

    stats_results = {"per_scene": {}, "summary": {}, "bp_alpha": {}}

    # 1. 逐场景 t-test
    for scen in sorted(metrics.keys()):
        stats_results["per_scene"][scen] = {}
        for ctrl in controls:
            for metric in metric_names:
                a = metrics[scen].get(treatment, {}).get(metric, [])
                b = metrics[scen].get(ctrl, {}).get(metric, [])
                if len(a) < 2 or len(b) < 2:
                    continue
                t_stat, p_val, d, ci = t_test(a, b)
                stats_results["per_scene"][scen][f"{ctrl}_{metric}"] = {
                    "treatment_mean": round(float(np.mean(a)), 2),
                    "control_mean": round(float(np.mean(b)), 2),
                    "treatment_std": round(float(np.std(a, ddof=1)), 2) if len(a) > 1 else 0,
                    "control_std": round(float(np.std(b, ddof=1)), 2) if len(b) > 1 else 0,
                    "t_stat": round(t_stat, 4),
                    "p_value": round(p_val, 6),
                    "cohen_d": round(d, 4),
                    "ci_95": [round(ci[0], 2), round(ci[1], 2)],
                    "significant": p_val < 0.05,
                }

    # 2. 全场景汇总 (所有场景的 seed 值合并做整体 t-test, n=32×3=96)
    for ctrl in controls:
        for metric in metric_names:
            all_a, all_b = [], []
            for scen in metrics:
                all_a.extend(metrics[scen].get(treatment, {}).get(metric, []))
                all_b.extend(metrics[scen].get(ctrl, {}).get(metric, []))
            if len(all_a) < 2 or len(all_b) < 2:
                continue
            t_stat, p_val, d, ci = t_test(all_a, all_b)
            stats_results["summary"][f"{ctrl}_{metric}"] = {
                "treatment_mean": round(float(np.mean(all_a)), 2),
                "control_mean": round(float(np.mean(all_b)), 2),
                "n_treatment": len(all_a),
                "n_control": len(all_b),
                "t_stat": round(t_stat, 4),
                "p_value": round(p_val, 6),
                "cohen_d": round(d, 4),
                "ci_95": [round(ci[0], 2), round(ci[1], 2)],
                "significant": p_val < 0.05,
            }

    # 3. BP alpha 值汇总 (验证碱基对是否在学习)
    for scen in sorted(bp_alphas.keys()):
        for model in bp_alphas[scen]:
            alphas = bp_alphas[scen][model]
            if alphas:
                stats_results["bp_alpha"][scen] = {
                    "mean": round(float(np.mean(alphas)), 4),
                    "max": round(float(np.max(alphas)), 4),
                    "all": [round(a, 4) for a in alphas],
                }

    # ============ 打印汇总表 ============
    print("\n" + "=" * 100)
    print(f"  统计检验汇总: {treatment} vs 对照组")
    print(f"  (全场景合并, n={len(metrics)}场景×3seed={len(metrics)*3}样本)")
    print("=" * 100)
    print(f"{'对照组':<14} {'指标':<10} {'treatment':>10} {'control':>10} "
          f"{'t':>8} {'p':>10} {'d':>8} {'显著':>6}")
    print("-" * 100)
    for key, s in stats_results["summary"].items():
        parts = key.rsplit("_", 2)
        ctrl = parts[0]
        metric = "_".join(parts[1:])
        label = metric_labels.get(metric, metric)
        sig = sig_mark(s["p_value"])
        print(f"{ctrl:<14} {label:<10} {s['treatment_mean']:>9.2f}% {s['control_mean']:>9.2f}% "
              f"{s['t_stat']:>8.3f} {s['p_value']:>10.4f} {s['cohen_d']:>8.3f} {sig:>6}")
    print("=" * 100)

    # ============ 逐场景显著性统计 ============
    print(f"\n  逐场景显著性统计 (p<0.05 占比):")
    for ctrl in controls:
        for metric in metric_names:
            key_prefix = f"{ctrl}_{metric}"
            total_scenes = 0
            sig_scenes = 0
            for scen, scene_stats in stats_results["per_scene"].items():
                if key_prefix in scene_stats:
                    total_scenes += 1
                    if scene_stats[key_prefix]["significant"]:
                        sig_scenes += 1
            if total_scenes > 0:
                label = metric_labels.get(metric, metric)
                print(f"    {treatment} vs {ctrl:<14} {label:<10}: "
                      f"{sig_scenes}/{total_scenes} 场景显著 ({sig_scenes/total_scenes*100:.0f}%)")

    # ============ BP alpha 汇总 ============
    if stats_results["bp_alpha"]:
        print(f"\n  碱基对 alpha 值 (验证是否在学习, 初始0):")
        alpha_vals = [v["mean"] for v in stats_results["bp_alpha"].values()]
        print(f"    场景数: {len(alpha_vals)}")
        print(f"    alpha均值: {np.mean(alpha_vals):.4f} (范围: {np.min(alpha_vals):.4f} ~ {np.max(alpha_vals):.4f})")
        if np.mean(alpha_vals) < 0.001:
            print(f"    [警告] alpha几乎没增长, 碱基对可能未生效 (梯度信号不足)")
        else:
            print(f"    [OK] alpha已增长, 碱基对耦合已生效")

    # 效应量解读
    print(f"\n  效应量解读 (Cohen's d):")
    print(f"    |d| < 0.2: 微小效应 | 0.2-0.5: 小效应 | 0.5-0.8: 中等效应 | > 0.8: 大效应")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats_results, f, ensure_ascii=False, indent=2)
    print(f"\n统计结果已保存: {out_path}")
    return stats_results


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "battle32_results.json"
    out_path = results_path.replace("results", "stats")
    if out_path == results_path:
        out_path = "battle32_stats.json"
    run_statistical_tests(results_path, out_path)
