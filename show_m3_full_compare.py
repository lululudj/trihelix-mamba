"""官方 Mamba3 vs Mamba3_ref vs Mamba2 完整三方对比

关键发现:
1. 官方 Mamba3 OOD decay = -11.3% vs mamba3_ref = -11.5% → 基本相同!
   说明 OOD 退化是 Mamba3 算法本身特性, 不是第三方移植问题
2. 官方 Mamba3 训练 69s vs mamba3_ref 1031s → 快 15x (Triton 内核)
3. 两者都输给 Mamba2 (+0.7% 零衰减) → Mamba2 SSD 结构对长程外推更优
"""
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt_pct(x):
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.1f}%"


print("=" * 78)
print(" 三方对照: Mamba2 (SSD) vs Mamba3_ref (Python) vs Mamba3_official (Triton)")
print("=" * 78)

# --- 训练数据 ---
m2_train = load("results/run_three_chain_mamba2_100step_m3subset_seed0/summary.json")
m3ref_train = load("results/run_three_chain_mamba3_seed0/summary.json")
m3off_train = load("results/run_three_chain_mamba3_official_seed0/summary.json")

# log.jsonl 取最后一步 ch_acc
def last_ch_acc(path):
    with open(path) as f:
        lines = f.readlines()
    last = json.loads(lines[-1])
    return last.get("changed_acc", 0.0), last.get("loss", 0.0)

m2_ch, m2_loss = last_ch_acc("results/run_three_chain_mamba2_100step_m3subset_seed0/log.jsonl")
m3ref_ch, m3ref_loss = last_ch_acc("results/run_three_chain_mamba3_seed0/log.jsonl")
m3off_ch, m3off_loss = last_ch_acc("results/run_three_chain_mamba3_official_seed0/log.jsonl")

# --- OOD 数据 ---
m2_ood = load("results/run_three_chain_mamba2_100step_m3subset_seed0/ood_metrics.json")
m3ref_ood = load("results/run_three_chain_mamba3_seed0/ood_metrics.json")
m3off_ood = load("results/run_three_chain_mamba3_official_seed0/ood_metrics.json")


def curve_at(d, t):
    return d.get("changed_acc_curve", {}).get(str(t), None)


print()
print(f"{'指标':<28} {'Mamba2 (SSD)':>15} {'Mamba3_ref':>15} {'Mamba3_official':>17}")
print("-" * 78)
print(f"{'实现':<28} {'Triton (官方)':>15} {'纯 Python':>15} {'Triton (官方)':>17}")
print(f"{'参数量':<28} {m2_train['params']:>15,} {m3ref_train['params']:>15,} {m3off_train['params']:>17,}")
print(f"{'训练 100 步耗时(s)':<28} {m2_train['elapsed']:>15.1f} {m3ref_train['elapsed']:>15.1f} {m3off_train['elapsed']:>17.1f}")
print(f"{'最后一步 loss':<28} {m2_loss:>15.3f} {m3ref_loss:>15.3f} {m3off_loss:>17.3f}")
print(f"{'训练 ch_acc (step 100)':<28} {m2_ch:>15.4f} {m3ref_ch:>15.4f} {m3off_ch:>17.4f}")
print()
print("  --- OOD 长程外推 (训练 T=100, 评估 T=150) ---")
print(f"{'changed_acc @ t=50':<28} {curve_at(m2_ood, 50):>15.4f} {curve_at(m3ref_ood, 50):>15.4f} {curve_at(m3off_ood, 50):>17.4f}")
print(f"{'changed_acc @ t=100':<28} {curve_at(m2_ood, 100):>15.4f} {curve_at(m3ref_ood, 100):>15.4f} {curve_at(m3off_ood, 100):>17.4f}")
print(f"{'changed_acc @ t=120':<28} {curve_at(m2_ood, 120):>15.4f} {curve_at(m3ref_ood, 120):>15.4f} {curve_at(m3off_ood, 120):>17.4f}")
print(f"{'changed_acc @ t=150':<28} {curve_at(m2_ood, 150):>15.4f} {curve_at(m3ref_ood, 150):>15.4f} {curve_at(m3off_ood, 150):>17.4f}")
print()
m2_decay = m2_ood.get("ood_decay_pct", 0)
m3ref_decay = m3ref_ood.get("ood_decay_pct", 0)
m3off_decay = m3off_ood.get("ood_decay_pct", 0)
print(f"{'OOD 衰减 (t=100→150)':<28} {fmt_pct(m2_decay):>15} {fmt_pct(m3ref_decay):>15} {fmt_pct(m3off_decay):>17}")
print(f"{'OOD 衰减绝对值':<28} {m2_ood.get('ood_decay_100_to_150', 0):>15.4f} {m3ref_ood.get('ood_decay_100_to_150', 0):>15.4f} {m3off_ood.get('ood_decay_100_to_150', 0):>17.4f}")
print()

print("=" * 78)
print(" 核心结论")
print("=" * 78)
print()
print("1. [算法本质] 官方 Mamba3 (-11.3%) ≈ Mamba3_ref (-11.5%):")
print("   OOD 退化是 Mamba3 算法(梯形离散化+RoPE)本身特性, 不是实现问题")
print()
print("2. [训练速度] 官方 Mamba3 (69s) 比 mamba3_ref (1031s) 快 15x:")
print("   Triton 内核生效, 但仍比 Mamba2 (60s) 慢 1.15x")
print()
print("3. [论文价值] 两版 Mamba3 都输给 Mamba2 (+0.7% 零衰减):")
print("   Mamba2 的 SSD 结构对长程外推显著优于 Mamba3 的梯形+RoPE")
print("   这构成强消融实验: 排除了'实现差异'解释, 确认是算法差异")
print()
print("=" * 78)
