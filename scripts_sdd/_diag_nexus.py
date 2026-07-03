"""诊断 nexus advanced eval 失败原因."""
import paramiko

HOST = "123.127.15.155"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)

def run(cmd):
    _, o, e = c.exec_command(cmd, timeout=30)
    return o.read().decode('utf-8', errors='replace'), e.read().decode('utf-8', errors='replace')

print("=== 1. stage2_nexus.log 全文 (关注 6a/6b/6c) ===")
o, _ = run(f"grep -A 30 '6a\\|6b\\|6c\\|eval_ood_advanced\\|eval_negative_shadow\\|Traceback\\|Error' /root/stage2_nexus.log | tail -80")
print(o)

print("\n=== 2. log 尾部 50 行 ===")
o, _ = run("tail -50 /root/stage2_nexus.log")
print(o)

print("\n=== 3. 检查 advanced eval 脚本是否存在 ===")
o, _ = run("ls -la /root/three_chain_v3/scripts_sdd/eval_*.py 2>/dev/null")
print(o)

print("\n=== 4. 检查 SSM 10k ood_metrics (基础 eval 成功?) ===")
o, _ = run("cat /root/three_chain_v3/results_stage2/nexus_30m_seed0_10k/ood_metrics.json 2>/dev/null | head -30")
print(o)

print("\n=== 5. results 目录 ===")
o, _ = run("ls -la /root/three_chain_v3/results_stage2/nexus_30m_seed0_10k/ /root/three_chain_v3/results_stage2/nexus_shuffle_3k/")
print(o)

print("\n=== 6. 手动重跑 eval_ood_advanced (看完整报错) ===")
o, e = run("cd /root/three_chain_v3 && /root/miniconda3/bin/python -u scripts_sdd/eval_ood_advanced.py "
           "--checkpoint results_stage2/nexus_30m_seed0_10k/best.pt "
           "--config configs/sdd_mamba2_30m_nexus.yaml "
           "--data_root ./data/ood_T150_sdd_nexus "
           "--out results_stage2/nexus_30m_seed0_10k/ood_metrics_advanced.json 2>&1 | tail -40")
print("STDOUT:", o)
print("STDERR:", e)

c.close()
