"""快速检查 tail_rerun.sh 和 700M 进度"""
import paramiko, sys

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)

print("[1] pid 21037 是什么 ...")
out,_ = run(c, "ps -p 21037 -o pid,cmd --no-headers 2>/dev/null || echo '(进程不存在)'")
print(out.strip())

print("\n[2] tail_rerun 相关进程 ...")
out,_ = run(c, "ps aux | grep tail_rerun | grep -v grep")
print(out.strip() if out.strip() else "(无)")

print("\n[3] tail_rerun.log ...")
out,_ = run(c, "cat /root/tail_rerun.log 2>/dev/null || echo '(无日志)'")
print(out.strip())

print("\n[4] 700M 训练进度 (log.jsonl 最后 5 行) ...")
out,_ = run(c, "tail -5 /root/three_chain_v3/results/run_700m_seed0_2k/log.jsonl 2>/dev/null")
print(out.strip())

print("\n[5] 700M 训练进程是否还在 ...")
out,_ = run(c, "pgrep -f '700m' && echo RUNNING || echo ENDED")
print(out.strip())

print("\n[6] run_700m_1000m.sh 进程 ...")
out,_ = run(c, "pgrep -f run_700m_1000m.sh && echo RUNNING || echo ENDED")
print(out.strip())

print("\n[7] 1000M 是否开始 ...")
out,_ = run(c, "ls /root/three_chain_v3/results/run_1000m_seed0_2k/ 2>/dev/null || echo '(1000M 尚未开始)'")
print(out.strip())

print("\n[8] GPU ...")
out,_ = run(c, "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
print(out.strip())

c.close()
