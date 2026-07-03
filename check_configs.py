"""检查云端配置文件和 700M 训练状态"""
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15, banner_timeout=15)
print("✓ connected")

# 1. 检查云端配置文件
out, _ = run(c, "ls /root/three_chain_v3/configs/matched_mamba2_hta* 2>/dev/null")
print(f"HTA configs: {out.strip()}")
out, _ = run(c, "ls /root/three_chain_v3/configs/matched_mamba2_100m* 2>/dev/null")
print(f"100M configs: {out.strip()}")

# 2. 上传 matched_mamba2_hta.yaml (如果云端没有)
sftp = c.open_sftp()
try:
    sftp.stat("/root/three_chain_v3/configs/matched_mamba2_hta.yaml")
    print("matched_mamba2_hta.yaml: 已存在")
except:
    print("matched_mamba2_hta.yaml: 上传中...")
    sftp.put(r"e:\three_chain_v3\configs\matched_mamba2_hta.yaml",
             "/root/three_chain_v3/configs/matched_mamba2_hta.yaml")
    print("matched_mamba2_hta.yaml: 已上传")
sftp.close()

# 3. 检查已完成的 hta 实验用的什么配置
out, _ = run(c, "cat /root/three_chain_v3/results/run_100m_hta_seed0_500/summary.json 2>/dev/null")
print(f"hta_seed0 summary: {out.strip()[:200]}")

# 4. 700M 训练状态
out, _ = run(c, "tail -10 /root/run_700m_v2.log 2>/dev/null")
print(f"\n700M log tail:\n{out.strip()}")

cmd = "pgrep -af 'bash /root/run_700m_v2.sh' | grep -v grep || echo NONE"
out, _ = run(c, cmd)
print(f"\n700M process: {out.strip()}")

# 5. GPU 状态
out, _ = run(c, "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null")
print(f"GPU: {out.strip()}")

# 6. 磁盘
out, _ = run(c, "df -h / | tail -1")
print(f"Disk: {out.strip()}")

c.close()
print("\n✓ done")
