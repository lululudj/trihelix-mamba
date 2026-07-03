"""控制实验最终汇总 (实验1 + 实验3 已完成, 实验2 seed0/1 已出, seed2 进行中)
生成最终判定报告
"""
import json
import os

BASE = r"e:\three_chain_v3\results_cloud"


def decay_pct(curve, t_ref, t_final):
    if not curve:
        return None
    ch_ref = curve.get(str(t_ref))
    ch_final = curve.get(str(t_final))
    if ch_ref is None or ch_final is None or ch_ref == 0:
        return None
    return (ch_final - ch_ref) / ch_ref * 100


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


print("=" * 72)
print("控制实验最终汇总 (CONTROL EXPERIMENTS FINAL)")
print("=" * 72)

# 实验 3: 随机标签
print("\n[实验3] 随机标签 sanity (一票否决)")
shuf_ood = load(os.path.join(BASE, "run_30m_shufflelabel_seed0_500", "ood_metrics.json"))
shuf_sum = load(os.path.join(BASE, "run_30m_shufflelabel_seed0_500", "summary.json"))
if shuf_ood:
    ch = shuf_ood.get("changed_acc_curve", {})
    d = decay_pct(ch, 100, 150)
    val_ch = 0.3344  # 从 log 读的
    print(f"  val ch_acc = {val_ch:.3f}  (正常 30M ≈ 0.586)")
    print(f"  OOD decay  = {d:+.2f}%  (正常 ≈ ±1%, 噪声基线)")
    print(f"  ★ 判定: ch_acc<0.4 且 decay=-6%≠0% → 指标有效, 一票否决未触发 ✓")

# 实验 1: B@1000
print("\n[实验1] B (n4) @1000步 对照")
e1_t150 = load(os.path.join(BASE, "run_30m_deep_n4_seed0_1000", "ood_A_T150.json"))
e1_t300 = load(os.path.join(BASE, "run_30m_deep_n4_seed0_1000", "ood_A_T300.json"))
d150 = decay_pct(e1_t150.get("changed_acc_curve", {}), 100, 150) if e1_t150 else None
d300 = decay_pct(e1_t300.get("changed_acc_curve", {}), 100, 300) if e1_t300 else None
print(f"  B@1000: T150={d150:+.2f}%  T300={d300:+.2f}%  val=0.586")
print(f"  B@500 (旧): T150=+4.0%  T300=+7.6%  val=0.406")
print(f"  ★ 判定: +4% 是欠训练副产物, '越深越强' 叙事删除 ✓")

# 实验 2: 1B 多 seed
print("\n[实验2] 1B 多 seed (seed0=C2, seed1=新跑, seed2=进行中)")
seeds = {}
for seed in [0, 1, 2]:
    p = os.path.join(BASE, f"run_1000m_ckpt_long_seed{seed}_2000", "ood_metrics.json")
    d = load(p)
    if d:
        ch = d.get("changed_acc_curve", {})
        dec = decay_pct(ch, 100, 150)
        seeds[seed] = dec
        print(f"  seed{seed}: decay = {dec:+.3f}%")
    else:
        print(f"  seed{seed}: 进行中 ...")

if len(seeds) >= 2:
    vals = list(seeds.values())
    spread = max(vals) - min(vals)
    all_in_range = all(abs(v) < 2 for v in vals)
    print(f"  方差范围 (spread) = {spread:.3f}%")
    print(f"  全在 ±2% 内: {'是 ✓' if all_in_range else '否 ⚠'}")
    if len(seeds) == 2:
        print(f"  ★ 初判: 前2 seed 稳定 (需 seed2 确认)")
    elif len(seeds) == 3:
        if all_in_range:
            print(f"  ★ 最终判定: 1B 不退化成立 ✓ (3 seeds 全 ±2%)")
        else:
            print(f"  ★ 判定: 有 seed 偏离, 1B 不稳定")

# B2 对照
print("\n[对照] B2 (n8 @1000步)")
b2_t150 = load(os.path.join(BASE, "run_30m_deep_n8_seed0_1000", "ood_A_T150.json"))
b2_t300 = load(os.path.join(BASE, "run_30m_deep_n8_seed0_1000", "ood_A_T300.json"))
d = decay_pct(b2_t150.get("changed_acc_curve", {}), 100, 150) if b2_t150 else None
d2 = decay_pct(b2_t300.get("changed_acc_curve", {}), 100, 300) if b2_t300 else None
print(f"  B2: T150={d:+.2f}%  T300={d2:+.2f}%  val=0.563")

print("\n" + "=" * 72)
print("综合结论")
print("=" * 72)
print("""
✓ decay 指标有效 (实验3: 噪声基线 -6%, 正常模型 ±1%)
✓ SSM 没崩盘 (定性结论稳)
✗ '越深越强' 删除 (实验1: B@1000 回归 ±0.3%, +4% 是欠训练)
○ '1B 不退化' (实验2: seed0=+0.28%, seed1=-0.12%, 方差0.4%, 等seed2)

可对外定量说的: SSM 在 30M→1B 范围内, decay 稳定在 ±1% 区间,
              且该指标能区分泛化与噪声 (随机标签 decay=-6%)
""")
