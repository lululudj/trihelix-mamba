"""跑真实自进化循环 (三链 Mamba3 模式).

三链 Mamba3 架构:
  - 时间链/空间链/因果链独立 SSM (真 mamba_ssm.Mamba2 内核)
  - best.pt 含三链预训练权重
  - char_embed_v3.pt 做 ppl 打分

对比旧 CodeBERT 模式 (run_evolve2.py):
  - 旧: MAMBA_BRAIN=text_hf + HF_CODEBERT 检索
  - 新: MAMBA_BRAIN=mamba + 真 Mamba2 三链检索+生成
"""
import sys
sys.path.insert(0, r"e:\mambacoding")
import paramiko
import time
from cloud_ssh_config import HOST, PORT, USER, PASS, REMOTE_DIR

INIT = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && export HF_ENDPOINT=https://hf-mirror.com && "

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
print(f"[SSH] 连接成功 {HOST}:{PORT}")

# 1. 验证 .env (AGNES_API_KEY)
print("\n--- Step 1: 验证 AGNES_API_KEY ---")
cmd = f"grep -c AGNES_API_KEY {REMOTE_DIR}/.env 2>/dev/null && echo 'KEY_FOUND' || echo 'NO_KEY'"
stdin, stdout, stderr = client.exec_command(cmd, timeout=10)
print(stdout.read().decode("utf-8", errors="ignore"))

# 2. 跑真实自进化 (三链 Mamba3 模式)
# MAMBA_BRAIN=mamba: 真 mamba_ssm.Mamba2 + best.pt 三链权重
# MAMBA_BEST_PT: 指向三链预训练权重
# MAMBA_CHAR_EMBED_V1: 指向 char_embed_v1.pt (score ppl 打分依赖)
print("\n--- Step 2: 跑真实自进化循环 (三链 Mamba3, 3 任务) ---")
cmd = INIT + (
    f"cd {REMOTE_DIR} && "
    f"MAMBA_BRAIN=mamba "
    f"MAMBA_BEST_PT={REMOTE_DIR}/checkpoints/best.pt "
    f"MAMBA_CHAR_EMBED_V1={REMOTE_DIR}/checkpoints/char_embed_v1.pt "
    f"timeout 900 python run_real_evolve.py 2>&1"
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=960, get_pty=True)
out = []
while True:
    line = stdout.readline()
    if not line:
        break
    print(line, end="")
    out.append(line)
full_out = "".join(out)

# 3. 下载报告
print("\n--- Step 3: 下载报告 ---")
LOCAL_TRANS = r"e:\mambacoding\transcripts"
import os
os.makedirs(LOCAL_TRANS, exist_ok=True)
sftp = client.open_sftp()
remote_report = f"{REMOTE_DIR}/transcripts/real_evolve_report_mamba.md"
local_report = f"{LOCAL_TRANS}\\real_evolve_report_mamba.md"
try:
    sftp.get(remote_report, local_report)
    print(f"  ✅ 下载报告 → {local_report}")
except Exception as e:
    print(f"  ❌ 下载报告失败: {e}")

# 训练对文件
remote_pairs = f"/root/autodl-tmp/mambacoding/memory_bank_data/contrastive_train_from_cases_mamba.jsonl"
local_pairs = f"{LOCAL_TRANS}\\contrastive_train_from_cases_mamba.jsonl"
try:
    sftp.get(remote_pairs, local_pairs)
    print(f"  ✅ 下载训练对 → {local_pairs}")
except Exception as e:
    print(f"  ❌ 下载训练对失败: {e}")
sftp.close()

client.close()
print(f"\n{'='*60}\n[SSH] 完成")
