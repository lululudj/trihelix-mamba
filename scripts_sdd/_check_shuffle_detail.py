"""检查 shuffle 训练详情: val ch_acc 趋势 + shuffle 的完整 OOD 曲线.
理解为什么 shuffle changed_acc=0.42 比 SSM 0.27 还高.
"""
import json
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('123.127.15.155', port=50472, username='root',
          password='i9D1S9IoRMLR', timeout=30)

# 1. shuffle 训练 val ch_acc 趋势
print("=== [1] shuffle 训练 val ch_acc 趋势 ===")
_, o, _ = c.exec_command(
    'grep -E "val step|新最佳" /root/stage2_full.log 2>/dev/null | '
    'tail -20', timeout=30)
print(o.read().decode(errors='replace'))

# 2. shuffle 完整 OOD metrics (看曲线)
print("=== [2] shuffle OOD metrics 完整 ===")
_, o, _ = c.exec_command(
    'cat /root/three_chain_v3/results_stage2/sdd_shuffle_3k/'
    'ood_metrics.json 2>/dev/null', timeout=30)
content = o.read().decode(errors='replace')
m = json.loads(content)
print(f"acc_final = {m['acc_final']}")
print(f"changed_acc = {m['changed_acc']}")
print(f"decay_pct = {m['ood_decay_pct']}")
print("\nchanged_acc_curve 关键点:")
curve = m['changed_acc_curve']
for t in [1, 10, 25, 50, 75, 99, 100, 101, 125, 150]:
    print(f"  ch@{t:3d} = {curve.get(str(t), 'N/A')}")

# 3. SSM 10k vs shuffle 在各 t 的对比
print("\n=== [3] SSM 10k vs shuffle changed_acc 对比 ===")
_, o, _ = c.exec_command(
    'cat /root/three_chain_v3/results_stage2/sdd_30m_seed0_10k/'
    'ood_metrics.json 2>/dev/null', timeout=30)
m10 = json.loads(o.read().decode(errors='replace'))
curve10 = m10['changed_acc_curve']
print(f"{'t':>5} {'SSM_10k':>10} {'shuffle':>10} {'差值':>10}")
for t in [1, 10, 25, 50, 75, 99, 100, 101, 125, 150]:
    s = curve10.get(str(t), None)
    f = curve.get(str(t), None)
    if s is not None and f is not None:
        print(f"{t:>5} {s:>10.4f} {f:>10.4f} {f-s:>+10.4f}")

# 4. acc_final 对比 (不变 cell 的准确率)
print(f"\n=== [4] acc_final 对比 (总准确率, 含不变 cell) ===")
print(f"SSM 10k  acc_final = {m10['acc_final']}")
print(f"shuffle  acc_final = {m['acc_final']}")

# 5. shuffle 的训练 loss 趋势 (看是否在学)
print(f"\n=== [5] shuffle 训练 loss 趋势 ===")
_, o, _ = c.exec_command(
    'grep -E "^\\[step" /root/stage2_full.log 2>/dev/null | '
    'grep "3000\\]" | tail -10', timeout=30)
print(o.read().decode(errors='replace'))

# 6. shuffle 最后几个 step 的 ch_acc
print(f"\n=== [6] shuffle 最后 10 个训练 step ===")
_, o, _ = c.exec_command(
    'grep -E "^\\[step" /root/stage2_full.log 2>/dev/null | tail -10',
    timeout=30)
print(o.read().decode(errors='replace'))

c.close()
print("\n=== 检查完成 ===")
