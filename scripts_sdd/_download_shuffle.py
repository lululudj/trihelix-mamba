"""下载 shuffle 结果到本地 results_stage2/sdd_shuffle_3k/."""
import os
import json
import paramiko

LOCAL = r'e:\three_chain_v3\results_stage2\sdd_shuffle_3k'
os.makedirs(LOCAL, exist_ok=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('123.127.15.155', port=50472, username='root',
          password='i9D1S9IoRMLR', timeout=30)

print("[1] 下载 shuffle ood_metrics.json ...")
_, o, _ = c.exec_command(
    'cat /root/three_chain_v3/results_stage2/sdd_shuffle_3k/'
    'ood_metrics.json', timeout=30)
content = o.read().decode(errors='replace')
with open(os.path.join(LOCAL, 'ood_metrics.json'), 'w') as f:
    f.write(content)
m = json.loads(content)
print(f"  ch@100 = {m['changed_acc_curve']['100']}")
print(f"  ch@150 = {m['changed_acc_curve']['150']}")
print(f"  decay_pct = {m['ood_decay_pct']}")

print("\n[2] 下载 stage2_full.log 完整 ...")
_, o, _ = c.exec_command('cat /root/stage2_full.log 2>/dev/null',
                         timeout=30)
log = o.read().decode(errors='replace')
with open(os.path.join(LOCAL, 'stage2_full.log'), 'w') as f:
    f.write(log)
print(f"  日志 {len(log)} 字节已保存")

c.close()
print("\n=== 下载完成 ===")
