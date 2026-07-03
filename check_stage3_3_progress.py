"""快速检查 Stage 3.3 云端进度"""
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=10)

def run(cmd):
    _, o, e = c.exec_command(cmd, timeout=30)
    return o.read().decode(errors='replace').strip()

# GPU 状态
gpu = run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo 'N/A'")
print(f"GPU: {gpu}")

# 训练进程
procs = run("pgrep -af 'python.*train.py' 2>/dev/null || echo '无训练进程'")
print(f"训练进程: {procs}")

# probe 进程
probe = run("pgrep -af 'probe_token' 2>/dev/null || echo '无探针进程'")
print(f"探针进程: {probe}")

# marker
marker = run("test -f /root/STAGE3_3_DONE && echo DONE || echo RUNNING")
print(f"Stage 3.3 状态: {marker}")

# 日志末尾
log = run("tail -n 15 /root/stage3_3.log 2>/dev/null || echo '(无日志)'")
print(f"\n日志末尾:")
print(log)

# 结果文件
files = run("ls -la /root/three_chain_v3/results_stage3/*.npz /root/three_chain_v3/results_stage3/*/ood_metrics.json 2>/dev/null || echo '无结果文件'")
print(f"\n结果文件:")
print(files)

c.close()
