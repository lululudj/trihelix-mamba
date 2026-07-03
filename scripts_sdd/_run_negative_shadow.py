"""上传 eval_negative_shadow.py 到云端, 跑 SSM 10k 负面分身评估, 下载结果."""
import os
import json
import paramiko

IP = '123.127.15.155'
PORT = 50472
LOCAL_SCRIPT = r'e:\three_chain_v3\scripts_sdd\eval_negative_shadow.py'
REMOTE_DIR = '/root/three_chain_v3'
PY = '/root/miniconda3/bin/python'
LOCAL_RES = r'e:\three_chain_v3\results_stage2'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(IP, port=PORT, username='root', password='i9D1S9IoRMLR', timeout=30)

# 上传
print("=== [1] 上传 eval_negative_shadow.py ===")
sftp = c.open_sftp()
sftp.put(LOCAL_SCRIPT, f'{REMOTE_DIR}/eval_negative_shadow.py')
print("  已上传")
sftp.close()

# 跑负面分身评估
print("\n=== [2] 跑 SSM 10k 负面分身评估 (5 模式, ~10min) ===")
cmd = (f'cd {REMOTE_DIR} && {PY} -u eval_negative_shadow.py '
       f'--checkpoint results_stage2/sdd_30m_seed0_10k/best.pt '
       f'--config configs/sdd_mamba2_30m.yaml '
       f'--data_root ./data/ood_T150_sdd '
       f'--batch_size 1 2>&1')
_, o, e = c.exec_command(cmd, timeout=900)
out = o.read().decode(errors='replace')
err = e.read().decode(errors='replace')
print(out)
if err.strip():
    print(f"[stderr] {err}")

# 下载
print("\n=== [3] 下载 negative_shadow.json ===")
sftp = c.open_sftp()
remote = f'{REMOTE_DIR}/results_stage2/sdd_30m_seed0_10k/negative_shadow.json'
local = os.path.join(LOCAL_RES, 'sdd_30m_seed0_10k', 'negative_shadow.json')
try:
    sftp.get(remote, local)
    print(f"  下载 ✓")
except Exception as ex:
    print(f"  [失败] {ex}")
sftp.close()

c.close()
print("\n=== 完成 ===")
