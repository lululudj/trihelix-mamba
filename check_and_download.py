"""下载已完成结果 + 诊断未完成的实验"""
import paramiko, json
from pathlib import Path

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_DIR = Path(r"e:\three_chain_v3\results_cloud")

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15, banner_timeout=15)
print("✓ 已连接\n")

# 1. 下载 300m_seed2 完整结果
print("[1] 下载 300m_seed2 结果 ...")
sftp = c.open_sftp()
local_exp = LOCAL_DIR / "run_300m_seed2_2k"
local_exp.mkdir(parents=True, exist_ok=True)
remote_exp = "/root/three_chain_v3/results/run_300m_seed2_2k"
for fname in sftp.listdir(remote_exp):
    remote_file = f"{remote_exp}/{fname}"
    local_file = local_exp / fname
    sftp.get(remote_file, str(local_file))
    print(f"  ✓ {fname} ({local_file.stat().st_size} bytes)")

# 验证 300m_seed2 OOD 结果
ood_file = local_exp / "ood_metrics.json"
if ood_file.exists() and ood_file.stat().st_size > 0:
    m = json.loads(ood_file.read_text(encoding="utf-8"))
    ch = m.get("changed_acc_curve", {})
    a100 = ch.get("100")
    a150 = ch.get("150")
    if a100 and a150:
        decay = (a150 - a100) / a100 * 100
        print(f"  → ch@100={a100:.4f}, ch@150={a150:.4f}, decay={decay:+.1f}%")
    else:
        print(f"  → (曲线不完整)")
sftp.close()

# 2. 检查 rerun_700m.sh 完整日志
print("\n[2] rerun_700m.log 完整内容:")
out, _ = run(c, "cat /root/rerun_700m.log 2>/dev/null")
print(out.strip() if out.strip() else "(空)")

# 3. 检查 rerun_700m_nohup.log
print("\n[3] rerun_700m_nohup.log:")
out, _ = run(c, "cat /root/rerun_700m_nohup.log 2>/dev/null")
print(out.strip()[-500:] if out.strip() else "(空)")

# 4. 检查 final_rerun.sh 是否正常结束
print("\n[4] final_rerun.log 最后 30 行:")
out, _ = run(c, "tail -30 /root/final_rerun.log 2>/dev/null")
print(out.strip())

# 5. 检查 100m_hta_seed1 的 final.pt 是否有效
print("\n[5] 100m_hta_seed1 final.pt 检查:")
out, _ = run(c, "ls -la /root/three_chain_v3/results/run_100m_hta_seed1_500/final.pt 2>/dev/null")
print(f"  {out.strip()}")

# 6. 检查 ALL_FINAL_DONE 标记
out, _ = run(c, "grep -c 'ALL_FINAL_DONE' /root/final_rerun.log 2>/dev/null || echo 0")
print(f"\n[6] ALL_FINAL_DONE 标记: {out.strip()}")

# 7. 检查 RERUN_700M_DONE 标记
out, _ = run(c, "grep -c 'RERUN_700M_DONE' /root/rerun_700m.log 2>/dev/null || echo 0")
print(f"[7] RERUN_700M_DONE 标记: {out.strip()}")

c.close()
print("\n✓ 诊断完毕")
