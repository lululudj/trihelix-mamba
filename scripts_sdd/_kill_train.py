"""杀掉 train.py 让 stage2_full.sh 自动继续到 eval + shuffle (用 IP 直连)."""
import time
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('123.127.15.155', port=50472, username='root', password='i9D1S9IoRMLR', timeout=30)

print("[1] 杀掉 train.py ...")
c.exec_command('pkill -9 -f "train.py" 2>/dev/null; sleep 3')
time.sleep(5)
_, o, _ = c.exec_command('pgrep -af "train.py" 2>&1 | grep -v grep')
print("  剩余 train.py:", o.read().decode(errors='replace').strip() or "(无)")

_, o, _ = c.exec_command('ls -la /root/three_chain_v3/results_stage2/sdd_30m_seed0_10k/best.pt 2>&1')
print("\n[2] best.pt:", o.read().decode(errors='replace').strip())

print("\n[3] 等待脚本进入 eval 阶段 ...")
time.sleep(10)
_, o, _ = c.exec_command('tail -8 /root/stage2_full.log 2>/dev/null')
print(o.read().decode(errors='replace'))

c.close()
