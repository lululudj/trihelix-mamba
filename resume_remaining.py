"""1. 跑 100m_hta_seed1 eval (final.pt 已存在)
   2. 启动 700M 训练 (nohup, 真正后台)
   3. 下载 300m_seed2 结果 (已下载则跳过)"""
import paramiko, time

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

# 0. 检查 100m_hta 用的配置
print("[0] 检查 100m_hta 配置 ...")
out, _ = run(c, "cat /root/three_chain_v3/results/run_100m_hta_seed1_500/summary.json 2>/dev/null")
print(f"  summary: {out.strip()}")
out, _ = run(c, "ls /root/three_chain_v3/configs/matched_mamba2_100m* 2>/dev/null")
print(f"  configs: {out.strip()}")

# 确认 eval_ood.py 有 no_grad
out, _ = run(c, "grep -n 'no_grad' /root/three_chain_v3/eval_ood.py | head -3")
print(f"  eval_ood no_grad: {out.strip()}")

# 1. 先清理磁盘 (删 100m_hta_seed1 best.pt, 700m 旧 log)
print("\n[1] 清理磁盘 ...")
out, _ = run(c, "rm -f /root/three_chain_v3/results/run_100m_hta_seed1_500/best.pt")
out, _ = run(c, "df -h / | tail -1")
print(f"  磁盘: {out.strip()}")

# 2. 跑 100m_hta_seed1 eval (同步, 约 1-2 分钟)
print("\n[2] 跑 100m_hta_seed1 eval (batch=1, no_grad) ...")
eval_cmd = """source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && \
cd /root/three_chain_v3 && \
export CUDA_HOME=/usr/local/cuda && \
python -u eval_ood.py --checkpoint results/run_100m_hta_seed1_500/final.pt \
  --config configs/matched_mamba2_100m_hta.yaml \
  --data_root ./data/ood_T150 --batch_size 1 --seed 1 \
  --out results/run_100m_hta_seed1_500/ood_metrics.json 2>&1"""
out, err = run(c, eval_cmd, t=300)
print(out[-800:] if len(out) > 800 else out)
if err.strip():
    print(f"STDERR: {err[-500:]}")

# 检查结果
out2, _ = run(c, "cat /root/three_chain_v3/results/run_100m_hta_seed1_500/ood_metrics.json 2>/dev/null | head -20")
if out2.strip():
    print(f"\n  ✓ 100m_hta_seed1 eval 完成!")
    print(f"  {out2.strip()[:300]}")
else:
    print(f"\n  ✗ 100m_hta_seed1 eval 仍无结果")

# 3. 删 100m_hta final.pt 腾空间
print("\n[3] 删 100m_hta_seed1 final.pt 腾空间 ...")
run(c, "rm -f /root/three_chain_v3/results/run_100m_hta_seed1_500/final.pt")
out, _ = run(c, "df -h / | tail -1")
print(f"  磁盘: {out.strip()}")

# 4. 启动 700M 训练 (nohup, 真正后台)
print("\n[4] 启动 700M 训练 (2000 步, batch=1) ...")

RUN_700M_SH = r"""#!/bin/bash
LOG=/root/run_700m_v2.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

echo "[$(date '+%F %T')] === 700M 训练 v2 ===" > $LOG

# 清理残留进程
pkill -f "train.py" 2>/dev/null || true
pkill -f "eval_ood.py" 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $LOG 2>&1
df -h / | tail -1 >> $LOG 2>&1

# 训练 700M
echo "[$(date '+%F %T')] 开始训练 700M (2000 步)" >> $LOG
python -u train.py --model three_chain_mamba2 --config configs/matched_mamba2_700m.yaml \
    --max_steps 2000 --batch_size 1 --seed 0 --data_root ./data_split_m3 \
    --out_dir results/run_700m_seed0_2k >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] 训练结束 rc=$TRAIN_RC" >> $LOG

if [ $TRAIN_RC -eq 0 ] && [ -f results/run_700m_seed0_2k/final.pt ]; then
    rm -f results/run_700m_seed0_2k/best.pt
    echo "[$(date '+%F %T')] 开始 eval" >> $LOG
    python -u eval_ood.py --checkpoint results/run_700m_seed0_2k/final.pt \
        --config configs/matched_mamba2_700m.yaml --data_root ./data/ood_T150 \
        --batch_size 1 --seed 0 --out results/run_700m_seed0_2k/ood_metrics.json >> $LOG 2>&1
    EVAL_RC=$?
    echo "[$(date '+%F %T')] eval 结束 rc=$EVAL_RC" >> $LOG
    # 删大文件腾空间
    rm -f results/run_700m_seed0_2k/best.pt results/run_700m_seed0_2k/final.pt
else
    echo "[$(date '+%F %T')] 训练失败, 无法 eval" >> $LOG
fi

df -h / | tail -1 >> $LOG 2>&1
echo "[$(date '+%F %T')] === RERUN_700M_V2_DONE ===" >> $LOG
"""

# 上传脚本
sftp = c.open_sftp()
with sftp.open("/root/run_700m_v2.sh", "w") as f:
    f.write(RUN_700M_SH)
sftp.close()
run(c, "chmod +x /root/run_700m_v2.sh")
print("  ✓ run_700m_v2.sh 已上传")

# 用 nohup + setsid 真正后台启动
run(c, "nohup setsid bash /root/run_700m_v2.sh > /root/run_700m_v2_nohup.log 2>&1 < /dev/null &", t=10)
time.sleep(3)
out, _ = run(c, "pgrep -af 'bash /root/run_700m_v2.sh' | grep -v grep || echo NONE")
print(f"  进程: {out.strip()}")

out, _ = run(c, "tail -3 /root/run_700m_v2.log 2>/dev/null")
print(f"  日志: {out.strip()}")

c.close()
print("\n✓ 100m_hta_seed1 eval 已跑, 700M 训练已启动")
