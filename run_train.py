"""跑 CodeBERT 训练 (后台 nohup, 轮询输出)。"""
import paramiko
import time
import sys

HOST = "connect.bjb1.seetacloud.com"
PORT = 16141
USER = "root"
PASS = "BaJ+0sL75oMj"
PROJ = "/root/autodl-tmp/mambacoding"

INIT = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && export HF_ENDPOINT=https://hf-mirror.com && "

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
print(f"[SSH] 连接成功\n{'='*60}")

# 1. 先验证 sandbox 能 import
print("\n--- Step 1: 验证 sandbox import ---")
cmd = INIT + f"cd {PROJ} && python -c 'import sandbox; print(\"sandbox ok, BRAIN=\", type(sandbox.BRAIN).__name__)'"
stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
out = stdout.read().decode("utf-8", errors="ignore")
err = stderr.read().decode("utf-8", errors="ignore")
print(out[-1000:])
if err.strip():
    print(f"[stderr] {err[-1000:]}")

# 2. 启动训练 (nohup 后台)
print("\n--- Step 2: 启动 CodeBERT 训练 (后台) ---")
cmd = INIT + f"cd {PROJ} && nohup python train_b_codebert.py > train_b_codebert.log 2>&1 & echo $! > train.pid && sleep 2 && cat train.pid && echo '训练已启动'"
stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
out = stdout.read().decode("utf-8", errors="ignore")
print(out)

# 3. 轮询训练进度
print("\n--- Step 3: 轮询训练进度 ---")
max_wait = 600  # 最多等 10 分钟
wait = 0
last_size = 0
while wait < max_wait:
    time.sleep(20)
    wait += 20
    cmd = f"tail -5 {PROJ}/train_b_codebert.log 2>/dev/null; echo '---'; ps -p $(cat {PROJ}/train.pid 2>/dev/null) -o pid,etime,cmd --no-headers 2>/dev/null || echo 'PROCESS_DONE'"
    stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
    out = stdout.read().decode("utf-8", errors="ignore")
    # 检查是否完成
    if "PROCESS_DONE" in out:
        print(f"\n[+{wait}s] 训练进程已结束")
        print(out)
        break
    print(f"\n[+{wait}s]")
    print(out)

# 4. 训练完成后查看完整结果
print("\n--- Step 4: 训练结果 ---")
cmd = f"tail -30 {PROJ}/train_b_codebert.log; echo '==='; ls -la {PROJ}/checkpoints/text_encoder_b_codebert*.pt 2>/dev/null; echo '==='; cat {PROJ}/checkpoints/text_encoder_b_codebert_loss.json 2>/dev/null | tail -5"
stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
out = stdout.read().decode("utf-8", errors="ignore")
print(out)

client.close()
print(f"\n{'='*60}\n[SSH] 完成")
