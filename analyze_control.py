"""分析控制实验结果: 实验1 (B@1000) + 实验3 (随机标签 sanity)"""
import json
import os

BASE = r"e:\three_chain_v3\results_cloud"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def decay_pct(curve, t_ref, t_final):
    """计算 decay: (ch@final - ch@ref) / ch@ref * 100"""
    if not curve:
        return None
    ch_ref = curve.get(str(t_ref))
    ch_final = curve.get(str(t_final))
    if ch_ref is None or ch_final is None or ch_ref == 0:
        return None
    return (ch_final - ch_ref) / ch_ref * 100


print("=" * 75)
print("控制实验结果分析")
print("=" * 75)

# ============================================================
# 实验 3: 随机标签 sanity check (一票否决!)
# ============================================================
print("\n" + "─" * 75)
print("【实验3】随机标签 sanity check (一票否决)")
print("─" * 75)
shuf_summary = load_json(os.path.join(BASE, "run_30m_shufflelabel_seed0_500", "summary.json"))
shuf_ood = load_json(os.path.join(BASE, "run_30m_shufflelabel_seed0_500", "ood_metrics.json"))

# log.jsonl 是 jsonl, 需要逐行解析
shuf_log_path = os.path.join(BASE, "run_30m_shufflelabel_seed0_500", "log.jsonl")
train_ch_accs = []
val_ch_acc = None
if os.path.exists(shuf_log_path):
    with open(shuf_log_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "val":
                val_ch_acc = r.get("changed_acc")
            elif "changed_acc" in r:
                train_ch_accs.append(r["changed_acc"])

print(f"  训练 ch_acc (最后10步): {[f'{a:.3f}' for a in train_ch_accs[-10:]]}")
print(f"  训练 ch_acc 均值(后10步): {sum(train_ch_accs[-10:])/len(train_ch_accs[-10:]):.4f}")
print(f"  val ch_acc: {val_ch_acc:.4f}")
print(f"  val acc_final: {shuf_summary['best_val']:.4f}")

if shuf_ood:
    ch_curve = shuf_ood.get("changed_acc_curve", {})
    ch100 = ch_curve.get("100")
    ch150 = ch_curve.get("150")
    decay = decay_pct(ch_curve, 100, 150)
    print(f"\n  OOD T150 eval:")
    print(f"    changed_acc @ t=100: {ch100:.4f}")
    print(f"    changed_acc @ t=150: {ch150:.4f}")
    print(f"    decay (100→150): {decay:+.3f}%" if decay is not None else "    decay: N/A")
    print(f"    ood_decay_pct (脚本算): {shuf_ood.get('ood_decay_pct', 'N/A')}")

print("\n  ★ 判定 (按清单):")
print(f"    - 随机标签 val ch_acc = {val_ch_acc:.3f}  (>0.4? {'是 ⚠' if val_ch_acc > 0.4 else '否'})")
if shuf_ood:
    decay_val = decay if decay is not None else 0
    print(f"    - decay = {decay_val:+.3f}%  (≈0%? {'是 ⚠' if abs(decay_val) < 2 else '否'})")
    if val_ch_acc > 0.4 and abs(decay_val) < 2:
        print(f"\n  ★★★ 一票否决触发! ★★★")
        print(f"      指标坏了: 随机标签模型也能 ch_acc>0.4 且 decay≈0%")
        print(f"      decay 无法区分泛化与噪声 → 所有 decay 定量结论作废")
    else:
        print(f"\n  ✓ 指标有效: decay 能区分泛化与噪声")

# ============================================================
# 实验 1: B (n_layers=4) @ 1000 步
# ============================================================
print("\n" + "─" * 75)
print("【实验1】B (n_layers=4) @ 1000步 对照")
print("─" * 75)
e1_log_path = os.path.join(BASE, "run_30m_deep_n4_seed0_1000", "log.jsonl")
e1_t150 = load_json(os.path.join(BASE, "run_30m_deep_n4_seed0_1000", "ood_A_T150.json"))
e1_t300 = load_json(os.path.join(BASE, "run_30m_deep_n4_seed0_1000", "ood_A_T300.json"))

train_ch_accs1 = []
val_ch_acc1 = None
if os.path.exists(e1_log_path):
    with open(e1_log_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "val":
                val_ch_acc1 = r.get("changed_acc")
            elif "changed_acc" in r:
                train_ch_accs1.append(r["changed_acc"])
print(f"  训练 ch_acc (最后5步): {[f'{a:.3f}' for a in train_ch_accs1[-5:]]}")
print(f"  val ch_acc: {val_ch_acc1:.4f}" if val_ch_acc1 else "  val ch_acc: N/A")

if e1_t150:
    c = e1_t150.get("changed_acc_curve", {})
    ch100 = c.get("100")
    ch150 = c.get("150")
    d = decay_pct(c, 100, 150)
    print(f"\n  OOD T150 eval:")
    print(f"    changed_acc @ t=100: {ch100:.4f}")
    print(f"    changed_acc @ t=150: {ch150:.4f}")
    print(f"    decay (100→150): {d:+.3f}%" if d is not None else "    decay: N/A")

if e1_t300:
    c = e1_t300.get("changed_acc_curve", {})
    ch100 = c.get("100")
    ch300 = c.get("300")
    d = decay_pct(c, 100, 300)
    print(f"\n  OOD T300 eval:")
    print(f"    changed_acc @ t=100: {ch100:.4f}")
    print(f"    changed_acc @ t=300: {ch300:.4f}")
    print(f"    decay (100→300): {d:+.3f}%" if d is not None else "    decay: N/A")

print("\n  ★ 判定 (按清单):")
if e1_t150 and e1_t300:
    d150 = decay_pct(e1_t150.get("changed_acc_curve", {}), 100, 150) or 0
    d300 = decay_pct(e1_t300.get("changed_acc_curve", {}), 100, 300) or 0
    # 原 B (500步): T150 decay=+4.0%, T300 decay=+7.6%
    print(f"    - B@1000 步: T150 decay={d150:+.2f}%, T300 decay={d300:+.2f}%")
    print(f"    - 原 B@500 步: T150 decay=+4.0%, T300 decay=+7.6% (memory)")
    if abs(d150) < 2 and abs(d300) < 2:
        print(f"    → B@1000 回归 ±1%: B 的 +4% 是欠训练副产物, 删掉'越深越强'叙事")
    elif d150 > 3 or d300 > 3:
        print(f"    → B@1000 仍 +decay: 'n_layers=4 甜点'可能真实, 需深究")
    else:
        print(f"    → 结果介于两者之间, 不下定论")

# ============================================================
# 对照: 正常 30M (B2 n_layers=8 @1000步) 的 decay
# ============================================================
print("\n" + "─" * 75)
print("【对照】正常 30M 模型的 decay (来自之前 V2 结果)")
print("─" * 75)
b2_t150 = load_json(os.path.join(BASE, "run_30m_deep_n8_seed0_1000", "ood_A_T150.json"))
b2_t300 = load_json(os.path.join(BASE, "run_30m_deep_n8_seed0_1000", "ood_A_T300.json"))
if b2_t150:
    d = decay_pct(b2_t150.get("changed_acc_curve", {}), 100, 150)
    print(f"  B2 (n_layers=8 @1000步) T150 decay: {d:+.3f}%" if d else "  B2 T150: N/A")
if b2_t300:
    d = decay_pct(b2_t300.get("changed_acc_curve", {}), 100, 300)
    print(f"  B2 (n_layers=8 @1000步) T300 decay: {d:+.3f}%" if d else "  B2 T300: N/A")
print(f"  随机标签模型 T150 decay: -0.06% (上方实验3)")
print(f"\n  ★ 关键对比: 正常模型 decay vs 随机标签 decay")
print(f"    如果两者接近 → decay 测的是噪声基线, 不是泛化信号")
