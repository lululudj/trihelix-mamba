"""快速检查云端当前状态"""
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
print("✓ connected\n")

# 1. 进程
cmd = "pgrep -af 'run_700m_v2|run_hta_seed1|train.py|eval_ood' | grep -v grep || echo NONE"
out, _ = run(c, cmd)
print(f"[进程] {out.strip()}")

# 2. 700M 日志
out, _ = run(c, "tail -10 /root/run_700m_v2.log 2>/dev/null")
print(f"\n[700M log]")
print(out.strip())

# 3. hta_seed1 日志
out, _ = run(c, "tail -5 /root/run_hta_seed1.log 2>/dev/null")
print(f"\n[hta_seed1 log]")
print(out.strip())

# 4. GPU
out, _ = run(c, "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null")
print(f"\n[GPU] {out.strip()}")

# 5. 磁盘
out, _ = run(c, "df -h / | tail -1")
print(f"[Disk] {out.strip()}")

# 6. DONE 标记 (正确检查方式)
out, _ = run(c, "grep 'RERUN_700M_V2_DONE' /root/run_700m_v2.log 2>/dev/null || echo NOTFOUND")
print(f"\n[700M done?] {out.strip()}")
out, _ = run(c, "grep 'HTA_SEED1_DONE' /root/run_hta_seed1.log 2>/dev/null || echo NOTFOUND")
print(f"[hta done?] {out.strip()}")

# 7. 结果文件
for exp in ["run_700m_seed0_2k", "run_100m_hta_seed1_500"]:
    out, _ = run(c, f"ls -la /root/three_chain_v3/results/{exp}/ 2>/dev/null")
    print(f"\n[{exp}]")
    print(out.strip() if out.strip() else "  (不存在)")

c.close()
