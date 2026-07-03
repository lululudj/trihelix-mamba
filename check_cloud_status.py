"""快速检查 AutoDL 是否还在运行 + GPU 状态"""
import paramiko
import sys

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

try:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=10)
    print("✓ SSH 连接成功 - AutoDL 实例仍在运行")
    # 检查 GPU
    stdin, stdout, stderr = c.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1")
    gpu = stdout.read().decode().strip()
    print(f"GPU 状态: {gpu}")
    # 检查实验进程
    stdin, stdout, stderr = c.exec_command("pgrep -af 'python.*train.py' 2>&1 || echo '无训练进程'")
    procs = stdout.read().decode().strip()
    print(f"训练进程: {procs}")
    c.close()
except Exception as e:
    print(f"✗ SSH 连接失败: {e}")
    print("→ AutoDL 实例可能已关机或网络中断")
sys.stdout.flush()
