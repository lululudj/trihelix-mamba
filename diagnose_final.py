"""诊断: 700M 为什么无结果, 重跑为什么失败"""
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

def run(c, cmd, t=60):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)

print("[1] 700M 目录内容 ...")
out,_ = run(c, "ls -la /root/three_chain_v3/results/run_700m_seed0_2k/")
print(out.strip())

print("\n[2] 700M log.jsonl 最后 10 行 ...")
out,_ = run(c, "tail -10 /root/three_chain_v3/results/run_700m_seed0_2k/log.jsonl")
print(out.strip())

print("\n[3] 700M tail_rerun.log 详细 (700M 部分) ...")
out,_ = run(c, "cat /root/tail_rerun.log")
print(out.strip())

print("\n[4] 1000M 目录内容 ...")
out,_ = run(c, "ls -la /root/three_chain_v3/results/run_1000m_seed0_2k/ 2>/dev/null || echo '(无目录)'")
print(out.strip())

print("\n[5] 1000M log.jsonl ...")
out,_ = run(c, "cat /root/three_chain_v3/results/run_1000m_seed0_2k/log.jsonl 2>/dev/null || echo '(无日志)'")
print(out.strip())

print("\n[6] run_700m_1000m.sh 完整日志 (700M+1000M 部分) ...")
# 看 run_700m_1000m_nohup.log 或 run_700m_1000m.sh 的输出
out,_ = run(c, "cat /root/run_700m_1000m_nohup.log 2>/dev/null | tail -60 || echo '(无 nohup log)'")
print(out.strip())

print("\n[7] 磁盘状态 ...")
out,_ = run(c, "df -h / && echo '---' && du -sh /root/three_chain_v3/results/ 2>/dev/null")
print(out.strip())

print("\n[8] GPU 当前状态 ...")
out,_ = run(c, "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv && nvidia-smi | tail -10")
print(out.strip())

print("\n[9] 300m_seed2 重跑日志 ...")
out,_ = run(c, "cat /root/three_chain_v3/results/run_300m_seed2_2k/log.jsonl 2>/dev/null || echo '(无 log)'")
print(out.strip())
out,_ = run(c, "ls -la /root/three_chain_v3/results/run_300m_seed2_2k/ 2>/dev/null")
print(out.strip())

print("\n[10] 100m_hta_seed1 重跑日志 ...")
out,_ = run(c, "cat /root/three_chain_v3/results/run_100m_hta_seed1_500/log.jsonl 2>/dev/null || echo '(无 log)'")
print(out.strip())
out,_ = run(c, "ls -la /root/three_chain_v3/results/run_100m_hta_seed1_500/ 2>/dev/null")
print(out.strip())

print("\n[11] tail_rerun_nohup.log (完整, 看错误信息) ...")
out,_ = run(c, "cat /root/tail_rerun_nohup.log 2>/dev/null | tail -80")
print(out.strip())

c.close()
