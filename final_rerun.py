"""
final_rerun.py - 最终清理 + 700M eval + 重跑 2 个失败实验
上传 final_rerun.sh 到云端, nohup 后台运行
"""
import paramiko, time

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

FINAL_RERUN_SH = r"""#!/bin/bash
LOG=/root/final_rerun.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TRITON_CACHE_DIR=/root/.triton/cache

echo "[$(date '+%F %T')] === 最终清理 + 重跑 ===" > $LOG

# 0. kill 残留训练进程 (保留 jupyter/tensorboard)
echo "[0] 清理残留进程 ..." >> $LOG
pkill -f "train.py" 2>/dev/null || true
pkill -f "eval_ood.py" 2>/dev/null || true
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $LOG 2>&1

# 1. 清理磁盘
echo "[1] 清理磁盘 ..." >> $LOG
rm -f results/run_700m_seed0_2k/final.pt.tmp
rm -f results/run_300m_seed2_2k/best.pt.tmp
rm -f results/run_100m_hta_seed1_500/best.pt.tmp
rm -f results/run_300m_seed2_2k/final.pt
rm -f results/run_100m_hta_seed1_500/final.pt
rm -f results/run_300m_seed2_2k/best.pt
rm -f results/run_100m_hta_seed1_500/best.pt
# 清理 triton cache (磁盘满时可能损坏)
rm -rf /root/.triton/cache/*
# 清理 pip cache
pip cache purge 2>/dev/null || true
df -h / >> $LOG 2>&1

# 2. 700M eval (用 best.pt)
echo "[2] 700M eval (用 best.pt) ..." >> $LOG
if [ -f results/run_700m_seed0_2k/best.pt ]; then
    echo "  best.pt 存在, 开始 eval (batch=1) ..." >> $LOG
    python -u eval_ood.py --checkpoint results/run_700m_seed0_2k/best.pt \
        --config configs/matched_mamba2_700m.yaml --data_root ./data/ood_T150 \
        --batch_size 1 --seed 0 --out results/run_700m_seed0_2k/ood_metrics.json >> $LOG 2>&1
    EVAL_RC=$?
    if [ $EVAL_RC -eq 0 ]; then
        echo "  700M eval 完成!" >> $LOG
    else
        echo "  700M eval 失败 rc=$EVAL_RC" >> $LOG
    fi
    # 无论成功失败都删 best.pt (8.7GB 太大)
    rm -f results/run_700m_seed0_2k/best.pt
    echo "  best.pt 已删除 (省 8.7GB)" >> $LOG
else
    echo "  best.pt 不存在, 跳过" >> $LOG
fi
df -h / >> $LOG 2>&1

# 3. 重跑 300m_seed2_2k
echo "[3] 重跑 300m_seed2_2k (2000 步) ..." >> $LOG
echo "[$(date '+%F %T')] 开始训练" >> $LOG
python -u train.py --model three_chain_mamba2 --config configs/matched_mamba2_300m.yaml \
    --max_steps 2000 --batch_size 1 --seed 2 --data_root ./data_split_m3 \
    --out_dir results/run_300m_seed2_2k >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] 训练结束 rc=$TRAIN_RC" >> $LOG
if [ $TRAIN_RC -eq 0 ] && [ -f results/run_300m_seed2_2k/final.pt ]; then
    rm -f results/run_300m_seed2_2k/best.pt
    echo "  开始 eval ..." >> $LOG
    python -u eval_ood.py --checkpoint results/run_300m_seed2_2k/final.pt \
        --config configs/matched_mamba2_300m.yaml --data_root ./data/ood_T150 \
        --batch_size 4 --seed 2 --out results/run_300m_seed2_2k/ood_metrics.json >> $LOG 2>&1
    echo "  300m_seed2 eval 完成" >> $LOG
else
    echo "  300m_seed2 训练失败 rc=$TRAIN_RC" >> $LOG
fi
rm -f results/run_300m_seed2_2k/best.pt
rm -f results/run_300m_seed2_2k/final.pt
df -h / >> $LOG 2>&1

# 4. 重跑 100m_hta_seed1_500
echo "[4] 重跑 100m_hta_seed1_500 (500 步) ..." >> $LOG
echo "[$(date '+%F %T')] 开始训练" >> $LOG
python -u train.py --model three_chain_mamba2_hta --config configs/matched_mamba2_100m.yaml \
    --max_steps 500 --batch_size 2 --seed 1 --data_root ./data_split_m3 \
    --out_dir results/run_100m_hta_seed1_500 >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] 训练结束 rc=$TRAIN_RC" >> $LOG
if [ $TRAIN_RC -eq 0 ] && [ -f results/run_100m_hta_seed1_500/final.pt ]; then
    rm -f results/run_100m_hta_seed1_500/best.pt
    echo "  开始 eval ..." >> $LOG
    python -u eval_ood.py --checkpoint results/run_100m_hta_seed1_500/final.pt \
        --config configs/matched_mamba2_100m.yaml --data_root ./data/ood_T150 \
        --batch_size 4 --seed 1 --out results/run_100m_hta_seed1_500/ood_metrics.json >> $LOG 2>&1
    echo "  100m_hta_seed1 eval 完成" >> $LOG
else
    echo "  100m_hta_seed1 训练失败 rc=$TRAIN_RC" >> $LOG
fi
rm -f results/run_100m_hta_seed1_500/best.pt
rm -f results/run_100m_hta_seed1_500/final.pt
df -h / >> $LOG 2>&1

echo "[$(date '+%F %T')] === ALL_FINAL_DONE ===" >> $LOG
"""

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

def main():
    print("[1] 连接 ...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)
    print("    ✓ 连上了")

    print("[2] 上传 final_rerun.sh ...")
    sftp = c.open_sftp()
    with sftp.open("/root/final_rerun.sh", "w") as f:
        f.write(FINAL_RERUN_SH)
    sftp.close()
    run(c, "chmod +x /root/final_rerun.sh")
    print("    ✓ 已上传")

    print("[3] 检查是否已有 final_rerun 在跑 ...")
    out, _ = run(c, "pgrep -af 'bash /root/final_rerun.sh' | grep -v grep || echo NONE")
    print(f"    {out.strip()}")
    if "NONE" not in out:
        print("    已在运行, 不重启")
        c.close()
        return

    print("[4] 启动 nohup ...")
    run(c, "setsid bash /root/final_rerun.sh > /root/final_rerun_nohup.log 2>&1 < /dev/null &", t=10)
    time.sleep(3)
    out, _ = run(c, "pgrep -af 'bash /root/final_rerun.sh' | grep -v grep || echo NONE")
    print(f"    进程: {out.strip()}")

    print("\n[5] 初步日志 (10s 后) ...")
    time.sleep(10)
    out, _ = run(c, "cat /root/final_rerun.log 2>/dev/null")
    print(out.strip())

    c.close()
    print("\n✓ final_rerun.sh 已启动, 预计 15-20 分钟完成")
    print("  监控: python check_final.py")

if __name__ == "__main__":
    main()
