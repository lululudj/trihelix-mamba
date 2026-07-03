"""下载 10k eval 结果到本地 results_stage2/sdd_30m_seed0_10k/.
用 IP 直连.
"""
import os
import json
import paramiko

IP = '123.127.15.155'
PORT = 50472
LOCAL = r'e:\three_chain_v3\results_stage2\sdd_30m_seed0_10k'
os.makedirs(LOCAL, exist_ok=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(IP, port=PORT, username='root', password='i9D1S9IoRMLR',
          timeout=30)

# 1. ood_metrics.json
print("[1] 下载 ood_metrics.json ...")
_, o, _ = c.exec_command(
    'cat /root/three_chain_v3/results_stage2/sdd_30m_seed0_10k/'
    'ood_metrics.json', timeout=30)
content = o.read().decode(errors='replace')
with open(os.path.join(LOCAL, 'ood_metrics.json'), 'w') as f:
    f.write(content)
m = json.loads(content)
print(f"  ch@100 = {m['changed_acc_curve']['100']}")
print(f"  ch@150 = {m['changed_acc_curve']['150']}")
print(f"  decay_pct = {m['ood_decay_pct']}")
print(f"  decay_abs = {m['ood_decay_100_to_150']}")

# 2. 训练日志(关键 val 点)
print("\n[2] 下载训练日志关键点 ...")
_, o, _ = c.exec_command(
    'grep -E "val step|新最佳|train done" /root/stage2_full.log 2>/dev/null '
    '| head -40', timeout=30)
log = o.read().decode(errors='replace')
with open(os.path.join(LOCAL, 'val_log.txt'), 'w') as f:
    f.write(log)
print(log)

# 3. log 尾部
print("\n[3] log 尾部 20 行 ...")
_, o, _ = c.exec_command('tail -20 /root/stage2_full.log 2>/dev/null',
                         timeout=30)
print(o.read().decode(errors='replace'))

# 4. window decay 计算(ch@100 ±5 邻域 vs ch@150 ±5 邻域)
print("\n[4] 计算 window-smoothed decay ...")
curve = m['changed_acc_curve']
def window_avg(curve, center, half=5):
    vals = [curve.get(str(t)) for t in range(center-half, center+half+1)
            if curve.get(str(t)) is not None]
    return sum(vals)/len(vals) if vals else None

w100 = window_avg(curve, 100, 5)
w150 = window_avg(curve, 150, 5)
print(f"  window ch@100 (±5) = {w100}")
print(f"  window ch@150 (±5) = {w150}")
if w100 and w150:
    win_decay = (w150 - w100) / w100
    print(f"  window decay_pct   = {win_decay*100:.2f}%")

# 保存计算结果
summary = {
    'ch_acc_100_single': m['changed_acc_curve']['100'],
    'ch_acc_150_single': m['changed_acc_curve']['150'],
    'decay_pct_single': m['ood_decay_pct'],
    'ch_acc_100_window': w100,
    'ch_acc_150_window': w150,
    'decay_pct_window': (w150 - w100) / w100 if w100 and w150 else None,
    'n_samples': m['n_samples'],
    'train_T': m['train_T'],
    'ood_T': m['ood_T'],
    'model': m['model'],
}
with open(os.path.join(LOCAL, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n  summary.json 已保存")

c.close()
print("\n=== 下载完成 ===")
