"""后台轮询 stage2_full, 完成后下载结果并汇报."""
import time
import json
import os
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_BASE = r"e:\three_chain_v3"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)

def tail(n=10):
    _, o, _ = c.exec_command(f'tail -n {n} /root/stage2_full.log 2>/dev/null')
    return o.read().decode(errors='replace')

def gpu():
    _, o, _ = c.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null')
    return o.read().decode(errors='replace').strip()

def done():
    _, o, _ = c.exec_command('test -f /root/STAGE2_FULL_DONE && echo DONE || echo RUNNING')
    return o.read().decode(errors='replace').strip() == 'DONE'

print(f"开始轮询 stage2_full, 预计 ~80 min")
print(f"初始 GPU: {gpu()}")

waited = 0
while True:
    if done():
        print(f"\n✓✓✓ Stage 2 Full 完成! (等了 {waited} min)")
        break
    time.sleep(180)
    waited += 3
    print(f"\n[已等 {waited}min] GPU={gpu()}")
    lt = tail(5)
    for line in lt.strip().split('\n')[-4:]:
        if line:
            print(f"  {line}")

print("\n=== 最终日志 (尾部 50 行) ===")
print(tail(50))

# 下载结果
sftp = c.open_sftp()
for sub in ['sdd_30m_seed0_10k', 'sdd_shuffle_3k']:
    local_dir = os.path.join(LOCAL_BASE, "results_stage2", sub)
    os.makedirs(local_dir, exist_ok=True)
    remote = f"/root/three_chain_v3/results_stage2/{sub}/ood_metrics.json"
    try:
        sftp.get(remote, os.path.join(local_dir, "ood_metrics.json"))
        print(f"OK 下载 {sub}/ood_metrics.json")
    except Exception as e:
        print(f"[警告] {sub}: {e}")

# 下载日志
sftp.get("/root/stage2_full.log", os.path.join(LOCAL_BASE, "results_stage2", "stage2_full.log"))
print("OK 下载 stage2_full.log")

# 打印关键指标对比
print("\n" + "=" * 60)
print("Stage 2 Phase A 完整结果对比")
print("=" * 60)
for sub, label in [('sdd_30m_seed0_3k', 'SSM 3k (初步)'), ('sdd_30m_seed0_10k', 'SSM 10k (充分)'), ('sdd_shuffle_3k', 'shuffle_labels 3k (sanity)')]:
    p = os.path.join(LOCAL_BASE, "results_stage2", sub, "ood_metrics.json")
    if os.path.exists(p):
        m = json.load(open(p))
        ch100 = m.get("changed_acc_curve", {}).get("100", 0)
        ch150 = m.get("changed_acc", 0)
        decay = m.get("ood_decay_pct", 0)
        print(f"  {label:25s}: ch@100={ch100:.4f} ch@150={ch150:.4f} decay={decay:+.2%}")

sftp.close()
c.close()
