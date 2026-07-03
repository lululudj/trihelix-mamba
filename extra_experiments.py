"""
额外实验: 充分利用包天云端算力
等当前 700M + hta_seed1 完成后, 依次跑:
1. 30M HTA seed 0    (~5 min,  batch=2, 500步)
2. 300M HTA seed 0   (~15 min, batch=1, 2000步)
3. 700M seed 1       (~25 min, batch=1, 2000步)
4. 700M HTA seed 0   (~25 min, batch=1, 2000步)
每个实验后自动删 .pt 腾磁盘, eval 后自动下载结果
"""
import paramiko, time, json
from pathlib import Path

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_DIR = Path(r"e:\three_chain_v3\results_cloud")

EXTRA_SH = r"""#!/bin/bash
LOG=/root/extra_experiments.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

echo "[$(date '+%F %T')] === 额外实验开始 ===" > $LOG

# 等 700M + hta_seed1 完成
echo "[0] 等待当前实验完成 ..." >> $LOG
while pgrep -f 'run_700m_v2.sh' > /dev/null 2>&1 || pgrep -f 'run_hta_seed1.sh' > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date '+%F %T')] 当前实验已完成" >> $LOG

# 清理
pkill -f "train.py" 2>/dev/null || true
pkill -f "eval_ood.py" 2>/dev/null || true
sleep 3
rm -rf /root/.triton/cache/*
df -h / | tail -1 >> $LOG 2>&1
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $LOG 2>&1

run_exp() {
    local NAME=$1
    local MODEL=$2
    local CONFIG=$3
    local STEPS=$4
    local BATCH=$5
    local SEED=$6
    local OUTDIR=$7

    echo "" >> $LOG
    echo "========================================" >> $LOG
    echo "  ${NAME} (model=${MODEL}, seed=${SEED}, steps=${STEPS})" >> $LOG
    echo "  $(date '+%F %T') | 磁盘:" >> $LOG
    df -h / | tail -1 >> $LOG 2>&1
    echo "========================================" >> $LOG

    # 训练
    echo "[$(date '+%F %T')] 训练 ..." >> $LOG
    python -u train.py --model ${MODEL} --config configs/${CONFIG} \
        --max_steps ${STEPS} --batch_size ${BATCH} --seed ${SEED} \
        --data_root ./data_split_m3 --out_dir results/${OUTDIR} >> $LOG 2>&1
    local TRAIN_RC=$?
    echo "[$(date '+%F %T')] 训练结束 rc=${TRAIN_RC}" >> $LOG

    if [ ${TRAIN_RC} -eq 0 ] && [ -f results/${OUTDIR}/final.pt ]; then
        rm -f results/${OUTDIR}/best.pt
        # eval
        echo "[$(date '+%F %T')] eval ..." >> $LOG
        python -u eval_ood.py --checkpoint results/${OUTDIR}/final.pt \
            --config configs/${CONFIG} --data_root ./data/ood_T150 \
            --batch_size 1 --seed ${SEED} \
            --out results/${OUTDIR}/ood_metrics.json >> $LOG 2>&1
        local EVAL_RC=$?
        echo "[$(date '+%F %T')] eval 结束 rc=${EVAL_RC}" >> $LOG
        # 删大文件
        rm -f results/${OUTDIR}/final.pt results/${OUTDIR}/best.pt
        echo "  ${NAME}: 完成" >> $LOG
    else
        echo "  ${NAME}: 训练失败 rc=${TRAIN_RC}" >> $LOG
        rm -f results/${OUTDIR}/final.pt results/${OUTDIR}/best.pt results/${OUTDIR}/final.pt.tmp
    fi

    df -h / | tail -1 >> $LOG 2>&1
}

# 1. 30M HTA seed 0 (500步, batch=2)
run_exp "30m_hta_seed0" "three_chain_mamba2_hta" "matched_mamba2_30m.yaml" 500 2 0 "run_30m_hta_seed0_500"

# 2. 300M HTA seed 0 (2000步, batch=1)
run_exp "300m_hta_seed0" "three_chain_mamba2_hta" "matched_mamba2_300m.yaml" 2000 1 0 "run_300m_hta_seed0_2k"

# 3. 700M seed 1 (2000步, batch=1)
run_exp "700m_seed1" "three_chain_mamba2" "matched_mamba2_700m.yaml" 2000 1 1 "run_700m_seed1_2k"

# 4. 700M HTA seed 0 (2000步, batch=1)
run_exp "700m_hta_seed0" "three_chain_mamba2_hta" "matched_mamba2_700m.yaml" 2000 1 0 "run_700m_hta_seed0_2k"

echo "" >> $LOG
echo "[$(date '+%F %T')] === ALL_EXTRA_DONE ===" >> $LOG
df -h / | tail -1 >> $LOG 2>&1
"""

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15, banner_timeout=15)
    return c

def main():
    print("=== 额外实验: 充分利用包天算力 ===")
    print("实验队列:")
    print("  1. 30M HTA seed 0   (~5 min)")
    print("  2. 300M HTA seed 0  (~15 min)")
    print("  3. 700M seed 1     (~25 min)")
    print("  4. 700M HTA seed 0 (~25 min)")
    print("  总计: ~70 min (当前实验完成后开始)\n")

    c = connect()
    print("✓ 已连接")

    # 1. 上传脚本
    print("\n[1] 上传 extra_experiments.sh ...")
    sftp = c.open_sftp()
    with sftp.open("/root/extra_experiments.sh", "w") as f:
        f.write(EXTRA_SH)
    sftp.close()
    run(c, "chmod +x /root/extra_experiments.sh")
    print("    ✓ 已上传")

    # 2. 检查是否已在运行
    out, _ = run(c, "pgrep -af 'bash /root/extra_experiments.sh' | grep -v grep || echo NONE")
    if "NONE" not in out:
        print(f"\n[2] 已在运行: {out.strip()}")
    else:
        print("\n[2] 启动 extra_experiments.sh ...")
        run(c, "nohup setsid bash /root/extra_experiments.sh > /root/extra_nohup.log 2>&1 < /dev/null &", t=10)
        time.sleep(3)
        out, _ = run(c, "pgrep -af 'bash /root/extra_experiments.sh' | grep -v grep || echo NONE")
        print(f"    进程: {out.strip()}")

    # 3. 当前状态
    out, _ = run(c, "tail -3 /root/extra_experiments.log 2>/dev/null")
    print(f"\n[3] 日志:\n{out.strip()}")

    # 4. 检查当前实验进度
    out, _ = run(c, "tail -3 /root/run_700m_v2.log 2>/dev/null")
    print(f"\n[当前 700M]\n{out.strip()}")
    out, _ = run(c, "tail -3 /root/run_hta_seed1.log 2>/dev/null")
    print(f"\n[当前 hta_seed1]\n{out.strip()}")

    c.close()
    print("\n✓ 额外实验已排队, 会等当前实验完成后自动开始")
    print("  预计总计: 当前~15min + 额外~70min = ~85min")

if __name__ == "__main__":
    main()
