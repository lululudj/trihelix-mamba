"""下载 stage2 MVP 结果到本地 + 查看训练日志关键点."""
import os
import json
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_BASE = r"e:\three_chain_v3"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
sftp = c.open_sftp()

# 1. 下载 ood_metrics.json
local_dir = os.path.join(LOCAL_BASE, "results_stage2", "sdd_30m_seed0_3k")
os.makedirs(local_dir, exist_ok=True)
sftp.get("/root/three_chain_v3/results_stage2/sdd_30m_seed0_3k/ood_metrics.json",
         os.path.join(local_dir, "ood_metrics.json"))
print("OK ood_metrics.json 下载")

# 2. 下载训练日志
sftp.get("/root/stage2_retry.log",
         os.path.join(local_dir, "stage2_retry.log"))
print("OK stage2_retry.log 下载")

# 3. 查看 val 评估点 (训练日志中的 val 行)
_, o, _ = c.exec_command("grep -E '\\[val' /root/stage2_retry.log 2>/dev/null")
print("\n=== Val 评估点 ===")
print(o.read().decode(errors='replace'))

# 4. 查看最后 15 行训练日志
_, o, _ = c.exec_command("tail -15 /root/stage2_retry.log 2>/dev/null")
print("=== 训练尾部 ===")
print(o.read().decode(errors='replace'))

sftp.close()
c.close()
