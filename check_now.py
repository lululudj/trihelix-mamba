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
print("✓ 已连接\n")

# 1. 进程状态
out, _ = run(c, "pgrep -af 'final_rerun|rerun_700m|train.py|eval_ood' | grep -v grep || echo NONE")
print(f"[进程] {out.strip()}")

# 2. final_rerun 日志尾部
out, _ = run(c, "tail -15 /root/final_rerun.log 2>/dev/null")
print(f"\n[final_rerun.log 尾部]")
print(out.strip())

# 3. rerun_700m 日志尾部
out, _ = run(c, "tail -15 /root/rerun_700m.log 2>/dev/null")
print(f"\n[rerun_700m.log 尾部]")
print(out.strip())

# 4. 磁盘
out, _ = run(c, "df -h / | tail -1")
print(f"\n[磁盘] {out.strip()}")

# 5. GPU
out, _ = run(c, "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null")
print(f"[GPU] {out.strip()}")

# 6. 检查结果文件
print("\n[结果文件检查]")
for exp in ["run_300m_seed2_2k", "run_100m_hta_seed1_500", "run_700m_seed0_2k"]:
    out, _ = run(c, f"ls -la /root/three_chain_v3/results/{exp}/ 2>/dev/null")
    if out.strip():
        print(f"\n  {exp}:")
        for line in out.strip().split("\n"):
            print(f"    {line}")
    else:
        print(f"\n  {exp}: 目录不存在")

# 7. 检查 ood_metrics.json 内容
print("\n[OOD 结果检查]")
for exp in ["run_300m_seed2_2k", "run_100m_hta_seed1_500", "run_700m_seed0_2k"]:
    out, _ = run(c, f"cat /root/three_chain_v3/results/{exp}/ood_metrics.json 2>/dev/null | head -5")
    if out.strip():
        print(f"  {exp}: {out.strip()[:200]}")
    else:
        print(f"  {exp}: (无 ood_metrics.json)")

c.close()
print("\n✓ 检查完毕")
