"""Stage 2 Phase A 完整: 10k 步 SSM 训练 + shuffle_labels sanity (串行)."""
import os
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

RUN_SH = r"""#!/bin/bash
LOG=/root/stage2_full.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date +%H:%M:%S)] Stage 2 Full Start ===" > $LOG

# ============================================================
# 1. 10k 步 SSM 充分训练
# ============================================================
echo "[$(date +%H:%M:%S)] === 1. Train SSM 10k steps ===" >> $LOG
rm -rf results_stage2/sdd_30m_seed0_10k
python -u train.py --model three_chain_mamba2 --config configs/sdd_mamba2_30m.yaml \
    --max_steps 10000 --batch_size 1 --seed 0 \
    --data_root ./data_sdd_split \
    --out_dir results_stage2/sdd_30m_seed0_10k >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] SSM 10k train done" >> $LOG

# eval OOD T150
echo "[$(date +%H:%M:%S)] === Eval SSM 10k ===" >> $LOG
python -u eval_ood.py \
    --checkpoint results_stage2/sdd_30m_seed0_10k/best.pt \
    --config configs/sdd_mamba2_30m.yaml \
    --data_root ./data/ood_T150_sdd --batch_size 1 --seed 0 \
    --out results_stage2/sdd_30m_seed0_10k/ood_metrics.json >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] SSM 10k eval done" >> $LOG
echo "=== SSM 10k OOD decay ===" >> $LOG
python3 -c "import json; m=json.load(open('results_stage2/sdd_30m_seed0_10k/ood_metrics.json')); print(f'ch_acc@100={m[\"changed_acc_curve\"][\"100\"]:.4f} ch_acc@150={m[\"changed_acc\"]:.4f} decay={m[\"ood_decay_pct\"]:.4f}')" >> $LOG 2>&1

# ============================================================
# 2. shuffle_labels sanity (3k 步, 验证指标在真实数据上有效)
# ============================================================
echo "[$(date +%H:%M:%S)] === 2. shuffle_labels sanity 3k ===" >> $LOG
rm -rf results_stage2/sdd_shuffle_3k
python -u train.py --model three_chain_mamba2 --config configs/sdd_mamba2_30m.yaml \
    --max_steps 3000 --batch_size 1 --seed 0 \
    --data_root ./data_sdd_split \
    --out_dir results_stage2/sdd_shuffle_3k --shuffle_labels >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] shuffle train done" >> $LOG

# eval OOD T150 (shuffle)
echo "[$(date +%H:%M:%S)] === Eval shuffle ===" >> $LOG
python -u eval_ood.py \
    --checkpoint results_stage2/sdd_shuffle_3k/best.pt \
    --config configs/sdd_mamba2_30m.yaml \
    --data_root ./data/ood_T150_sdd --batch_size 1 --seed 0 \
    --out results_stage2/sdd_shuffle_3k/ood_metrics.json >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] shuffle eval done" >> $LOG
echo "=== shuffle OOD decay ===" >> $LOG
python3 -c "import json; m=json.load(open('results_stage2/sdd_shuffle_3k/ood_metrics.json')); print(f'ch_acc@100={m[\"changed_acc_curve\"][\"100\"]:.4f} ch_acc@150={m[\"changed_acc\"]:.4f} decay={m[\"ood_decay_pct\"]:.4f}')" >> $LOG 2>&1

touch /root/STAGE2_FULL_DONE
echo "[$(date +%H:%M:%S)] === STAGE2_FULL_DONE ===" >> $LOG
"""


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    sftp = c.open_sftp()

    # 上传脚本
    with sftp.open('/root/stage2_full.sh', 'w') as f:
        f.write(RUN_SH)
    sftp.chmod('/root/stage2_full.sh', 0o755)
    print("[1] 上传 stage2_full.sh OK")

    # 确认 GPU 空闲
    _, o, _ = c.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null')
    print(f"[2] GPU 状态: {o.read().decode(errors='replace').strip()}")
    _, o, _ = c.exec_command('pgrep -af "train.py" 2>/dev/null | grep -v grep')
    ps = o.read().decode(errors='replace').strip()
    print(f"[3] 训练进程: {ps or '(无, 空闲)'}")

    # 清理 marker + 后台启动
    c.exec_command('rm -f /root/STAGE2_FULL_DONE')
    time.sleep(1)
    c.exec_command('nohup bash /root/stage2_full.sh > /root/stage2_full.out 2>&1 &')
    time.sleep(8)

    _, o, _ = c.exec_command('cat /root/stage2_full.log 2>/dev/null')
    print("\n[4] 启动日志:")
    print(o.read().decode(errors='replace'))

    sftp.close()
    c.close()
    print("\n✓ 已启动: 10k 步 SSM + shuffle_labels sanity (预计 ~80 min)")


if __name__ == "__main__":
    main()
