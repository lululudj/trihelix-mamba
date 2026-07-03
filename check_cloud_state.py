"""快速检查云端状态: 哪些 checkpoint 还在, 数据是否齐全"""
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15)
print("✓ 已连接")

def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
    return stdout.read().decode().strip()

# 1. 检查所有 final.pt
print("\n=== 1. 可用 checkpoint ===")
out = run("find /root/three_chain_v3/results -name 'final.pt' -o -name 'best.pt' 2>/dev/null")
print(out if out else "无任何 .pt 文件 (都被删了)")

# 2. 检查数据目录
print("\n=== 2. OOD 数据 ===")
for d in ['ood_T150', 'ood_T200_noise', 'ood_T250_mask', 'ood_T300', 'ood_T500']:
    out = run(f"ls /root/three_chain_v3/data/{d} 2>/dev/null | wc -l")
    print(f"  {d}: {out} 文件")

# 3. 检查磁盘
print("\n=== 3. 磁盘 ===")
print(run("df -h / | tail -1"))

# 4. 检查 GPU
print("\n=== 4. GPU ===")
print(run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))

c.close()
