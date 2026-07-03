"""检查 stage2 全部进度: 进程 + GPU + markers + 最近 metrics."""
import paramiko
import json
from pathlib import Path

HOST = "123.127.15.155"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
REMOTE_DIR = "/root/three_chain_v3"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)

def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    return out, err

print("=== 1. GPU ===")
o, e = run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader")
print(o.strip() if o.strip() else "(empty)")

print("\n=== 2. 训练/eval 进程 ===")
o, e = run("ps aux | grep -E 'train.py|eval_|python' | grep -v grep")
print(o.strip() if o.strip() else "(无 python 进程)")

print("\n=== 3. stage2_full markers ===")
o, e = run(f"ls -la {REMOTE_DIR}/results_stage2/ 2>/dev/null | head -50")
print(o.strip() if o.strip() else "(no results_stage2)")
o, e = run(f"find {REMOTE_DIR} -name 'STAGE2_FULL_DONE' -o -name '*DONE*' 2>/dev/null | head")
print("markers:", o.strip() or "(none)")

print("\n=== 4. 最近 metrics 文件 ===")
o, e = run(f"find {REMOTE_DIR}/results_stage2 -name '*.json' -newer {REMOTE_DIR}/results_stage2/sdd_30m_seed0_10k/ood_metrics.json 2>/dev/null")
print("newer than 10k_metrics:", o.strip() or "(none newer)")

print("\n=== 5. 训练 log tail (stage2_full.log) ===")
o, e = run(f"tail -30 {REMOTE_DIR}/stage2_full.log 2>/dev/null")
print(o.strip() if o.strip() else "(no stage2_full.log)")

print("\n=== 6. nexus/deathCircle/multi 数据是否生成 ===")
o, e = run(f"ls {REMOTE_DIR}/data_sdd_split/train/ 2>/dev/null | head -5; echo ---; ls {REMOTE_DIR}/data_sdd_nexus 2>/dev/null | head -5; echo ---; ls {REMOTE_DIR}/data_sdd_multi 2>/dev/null | head -5")
print(o.strip())

c.close()
