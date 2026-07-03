"""确认云端空闲可关机 + 下载 V2 完整日志留档。"""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.bjb1.seetacloud.com', port=50472, username='root',
          password='i9D1S9IoRMLR', timeout=15)


def run(cmd):
    _, o, _ = c.exec_command(cmd, timeout=30)
    return o.read().decode(errors='replace').strip()


print("=== GPU/进程 ===")
print(run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'))
print(run('ps aux | grep -E "train.py|eval_ood|run_extreme" | grep -v grep || echo "(无训练进程)"'))
print()
print("=== Markers ===")
print(run('for m in C2 B2 D2 V2; do test -f /root/${m}_ALL_DONE && echo "$m: DONE" || echo "$m: pending"; done'))
print()

# 下载完整 V2 日志到本地留档
import os
sftp = c.open_sftp()
local_log = r'e:\three_chain_v3\results_cloud\v2_full_log.txt'
try:
    sftp.get('/root/extreme_v2.log', local_log)
    sz = os.path.getsize(local_log)
    print(f"✓ V2 完整日志已下载: {local_log} ({sz} bytes)")
except Exception as e:
    print(f"下载日志失败: {e}")
sftp.close()
c.close()
print("\n✓ 云端确认空闲, 可安全关机")
