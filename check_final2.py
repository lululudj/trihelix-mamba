"""检查 final_rerun 进度"""
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
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)

print("[1] final_rerun.log ...")
out,_ = run(c, "cat /root/final_rerun.log 2>/dev/null")
print(out.strip())

print("\n[2] 进程状态 ...")
out,_ = run(c, "pgrep -af 'bash /root/final_rerun.sh' | grep -v grep || echo '(已结束)'")
print(out.strip())

print("\n[3] GPU ...")
out,_ = run(c, "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
print(out.strip())

print("\n[4] 磁盘 ...")
out,_ = run(c, "df -h / | tail -1")
print(out.strip())

print("\n[5] 700M ood_metrics 是否已生成 ...")
out,_ = run(c, "cat /root/three_chain_v3/results/run_700m_seed0_2k/ood_metrics.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d.get(\"changed_acc_curve\",{}); print(f\"ch@100={c.get(\\\"100\\\")}, ch@150={c.get(\\\"150\\\")}\")' 2>/dev/null || echo '(尚未生成)'")
print(out.strip())

c.close()
