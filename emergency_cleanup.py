"""紧急清理磁盘, 然后上传 final_rerun.sh"""
import paramiko, time

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

print("[1] 磁盘清理前 ...")
out,_ = run(c, "df -h / | tail -1")
print(out.strip())

print("\n[2] kill 残留进程 ...")
run(c, "pkill -f train.py 2>/dev/null; pkill -f eval_ood.py 2>/dev/null; sleep 3")
out,_ = run(c, "nvidia-smi --query-gpu=memory.used --format=csv,noheader")
print(f"  GPU: {out.strip()}")

print("\n[3] 删除大文件 ...")
# 700M final.pt.tmp (4.7GB) + best.pt (8.7GB) - 先删 tmp, best.pt 留给 eval
out,_ = run(c, "rm -f /root/three_chain_v3/results/run_700m_seed0_2k/final.pt.tmp")
print(f"  删 700M final.pt.tmp: done")
# 300m_seed2 best.pt.tmp (2GB)
run(c, "rm -f /root/three_chain_v3/results/run_300m_seed2_2k/best.pt.tmp")
print(f"  删 300m_seed2 best.pt.tmp: done")
# 100m_hta_seed1 best.pt.tmp
run(c, "rm -f /root/three_chain_v3/results/run_100m_hta_seed1_500/best.pt.tmp")
print(f"  删 100m_hta_seed1 best.pt.tmp: done")
# triton cache
run(c, "rm -rf /root/.triton/cache/*")
print(f"  清 triton cache: done")
# pip cache
run(c, "pip cache purge 2>/dev/null; rm -rf /root/.cache/pip/* 2>/dev/null")
print(f"  清 pip cache: done")
# 清理旧的 100 步实验的 .pt (这些已经有 OOD 结果了)
run(c, "find /root/three_chain_v3/results/ -name 'final.pt' -size +100M -exec rm -f {} \\; 2>/dev/null")
print(f"  删所有 final.pt (>100M): done")
run(c, "find /root/three_chain_v3/results/ -name 'best.pt' -size +100M ! -path '*700m*' -exec rm -f {} \\; 2>/dev/null")
print(f"  删所有 best.pt (>100M, 保留 700M): done")

print("\n[4] 磁盘清理后 ...")
out,_ = run(c, "df -h / | tail -1")
print(out.strip())

print("\n[5] 上传 final_rerun.sh ...")
sftp = c.open_sftp()
FINAL_SH = """#!/bin/bash
LOG=/root/final_rerun.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

echo "[$(date '+%F %T')] === 最终重跑 ===" > $LOG

# 700M eval (用 best.pt)
echo "[1] 700M eval (用 best.pt, batch=1) ..." >> $LOG
if [ -f results/run_700m_seed0_2k/best.pt ]; then
    python -u eval_ood.py --checkpoint results/run_700m_seed0_2k/best.pt \\
        --config configs/matched_mamba2_700m.yaml --data_root ./data/ood_T150 \\
        --batch_size 1 --seed 0 --out results/run_700m_seed0_2k/ood_metrics.json >> $LOG 2>&1
    EVAL_RC=$?
    if [ $EVAL_RC -eq 0 ]; then
        echo "  700M eval 完成!" >> $LOG
    else
        echo "  700M eval 失败 rc=$EVAL_RC" >> $LOG
    fi
    rm -f results/run_700m_seed0_2k/best.pt
    echo "  best.pt 已删除 (省 8.7GB)" >> $LOG
else
    echo "  best.pt 不存在, 跳过" >> $LOG
fi
df -h / >> $LOG 2>&1

# 重跑 300m_seed2_2k
echo "[2] 重跑 300m_seed2_2k ..." >> $LOG
python -u train.py --model three_chain_mamba2 --config configs/matched_mamba2_300m.yaml \\
    --max_steps 2000 --batch_size 1 --seed 2 --data_root ./data_split_m3 \\
    --out_dir results/run_300m_seed2_2k >> $LOG 2>&1
TRAIN_RC=$?
if [ $TRAIN_RC -eq 0 ] && [ -f results/run_300m_seed2_2k/final.pt ]; then
    rm -f results/run_300m_seed2_2k/best.pt
    python -u eval_ood.py --checkpoint results/run_300m_seed2_2k/final.pt \\
        --config configs/matched_mamba2_300m.yaml --data_root ./data/ood_T150 \\
        --batch_size 4 --seed 2 --out results/run_300m_seed2_2k/ood_metrics.json >> $LOG 2>&1
    echo "  300m_seed2 eval 完成" >> $LOG
else
    echo "  300m_seed2 训练失败 rc=$TRAIN_RC" >> $LOG
fi
rm -f results/run_300m_seed2_2k/best.pt results/run_300m_seed2_2k/final.pt
df -h / >> $LOG 2>&1

# 重跑 100m_hta_seed1_500
echo "[3] 重跑 100m_hta_seed1_500 ..." >> $LOG
python -u train.py --model three_chain_mamba2_hta --config configs/matched_mamba2_100m.yaml \\
    --max_steps 500 --batch_size 2 --seed 1 --data_root ./data_split_m3 \\
    --out_dir results/run_100m_hta_seed1_500 >> $LOG 2>&1
TRAIN_RC=$?
if [ $TRAIN_RC -eq 0 ] && [ -f results/run_100m_hta_seed1_500/final.pt ]; then
    rm -f results/run_100m_hta_seed1_500/best.pt
    python -u eval_ood.py --checkpoint results/run_100m_hta_seed1_500/final.pt \\
        --config configs/matched_mamba2_100m.yaml --data_root ./data/ood_T150 \\
        --batch_size 4 --seed 1 --out results/run_100m_hta_seed1_500/ood_metrics.json >> $LOG 2>&1
    echo "  100m_hta_seed1 eval 完成" >> $LOG
else
    echo "  100m_hta_seed1 训练失败 rc=$TRAIN_RC" >> $LOG
fi
rm -f results/run_100m_hta_seed1_500/best.pt results/run_100m_hta_seed1_500/final.pt
df -h / >> $LOG 2>&1

echo "[$(date '+%F %T')] === ALL_FINAL_DONE ===" >> $LOG
"""
with sftp.open("/root/final_rerun.sh", "w") as f:
    f.write(FINAL_SH)
sftp.close()
run(c, "chmod +x /root/final_rerun.sh")
print("    ✓ 已上传")

print("\n[6] 启动 nohup ...")
run(c, "setsid bash /root/final_rerun.sh > /root/final_rerun_nohup.log 2>&1 < /dev/null &", t=10)
time.sleep(5)
out,_ = run(c, "pgrep -af 'bash /root/final_rerun.sh' | grep -v grep || echo NONE")
print(f"    进程: {out.strip()}")

print("\n[7] 初步日志 ...")
out,_ = run(c, "cat /root/final_rerun.log 2>/dev/null")
print(out.strip())

c.close()
print("\n✓ 已启动, 预计 15-20 分钟")
