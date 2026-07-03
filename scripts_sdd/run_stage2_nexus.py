"""Stage 2 实验 2: nexus 多场景验证 (云端编排).

流程:
  1. 上传 config (sdd_mamba2_30m_nexus.yaml) + eval 脚本
  2. 云端生成 nexus T=100 raw + T=150 OOD 数据
  3. split_dataset → data_sdd_split_nexus
  4. 训练 SSM 10k + eval OOD
  5. 训练 shuffle 3k + eval OOD
  6. eval_ood_advanced (SSM + shuffle) + eval_negative_shadow (SSM)
  7. touch STAGE2_NEXUS_DONE

预计耗时: ~5h (数据 gen ~10min + SSM 10k ~70min + shuffle 3k ~25min + evals ~30min)
"""
import os
import sys
import time
import paramiko

HOST = "123.127.15.155"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_ROOT = r"e:\three_chain_v3"
REMOTE_DIR = "/root/three_chain_v3"
PY = "/root/miniconda3/bin/python"

RUN_SH = r"""#!/bin/bash
LOG=/root/stage2_nexus.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python

echo "=== [$(date +%H:%M:%S)] Stage 2 Nexus Start ===" > $LOG

# ============================================================
# 1. 生成 nexus T=100 raw 数据
# ============================================================
echo "[$(date +%H:%M:%S)] === 1. Gen nexus T=100 raw ===" >> $LOG
rm -rf /root/three_chain_v3/data_sdd_nexus_raw
$PY -u data/gen_sdd_grid.py \
    --sdd_root /root/three_chain_v3/SDD \
    --out_dir /root/three_chain_v3/data_sdd_nexus_raw \
    --scenes nexus \
    --N 24 --K 8 --T 100 --fps_stride 6 >> $LOG 2>&1
N_RAW=$(ls /root/three_chain_v3/data_sdd_nexus_raw/*.npz 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] nexus T=100 raw: $N_RAW windows" >> $LOG

# ============================================================
# 2. 生成 nexus T=150 OOD 数据
# ============================================================
echo "[$(date +%H:%M:%S)] === 2. Gen nexus T=150 OOD ===" >> $LOG
rm -rf /root/three_chain_v3/data/ood_T150_sdd_nexus
$PY -u data/gen_sdd_grid.py \
    --sdd_root /root/three_chain_v3/SDD \
    --out_dir /root/three_chain_v3/data/ood_T150_sdd_nexus \
    --scenes nexus \
    --N 24 --K 8 --T 150 --fps_stride 6 >> $LOG 2>&1
N_OOD=$(ls /root/three_chain_v3/data/ood_T150_sdd_nexus/*.npz 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] nexus T=150 OOD: $N_OOD windows" >> $LOG

# ============================================================
# 3. split_dataset → data_sdd_split_nexus/{train,val,test}
# ============================================================
echo "[$(date +%H:%M:%S)] === 3. Split dataset ===" >> $LOG
rm -rf /root/three_chain_v3/data_sdd_split_nexus
$PY -c "
import sys; sys.path.insert(0, '/root/three_chain_v3')
from data.dataset import split_dataset
split_dataset('/root/three_chain_v3/data_sdd_nexus_raw', '/root/three_chain_v3/data_sdd_split_nexus', ratios=(0.8, 0.1, 0.1), seed=0)
" >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] split done" >> $LOG

# ============================================================
# 4. 训练 SSM 10k + eval OOD
# ============================================================
echo "[$(date +%H:%M:%S)] === 4. Train SSM 10k ===" >> $LOG
rm -rf results_stage2/nexus_30m_seed0_10k
$PY -u train.py --model three_chain_mamba2 --config configs/sdd_mamba2_30m_nexus.yaml \
    --max_steps 10000 --batch_size 1 --seed 0 \
    --data_root ./data_sdd_split_nexus \
    --out_dir results_stage2/nexus_30m_seed0_10k >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] SSM 10k train done" >> $LOG

echo "[$(date +%H:%M:%S)] === Eval SSM 10k OOD ===" >> $LOG
$PY -u eval_ood.py \
    --checkpoint results_stage2/nexus_30m_seed0_10k/best.pt \
    --config configs/sdd_mamba2_30m_nexus.yaml \
    --data_root ./data/ood_T150_sdd_nexus --batch_size 1 --seed 0 \
    --out results_stage2/nexus_30m_seed0_10k/ood_metrics.json >> $LOG 2>&1
$PY -c "import json; m=json.load(open('results_stage2/nexus_30m_seed0_10k/ood_metrics.json')); print(f'SSM ch@100={m[\"changed_acc_curve\"][\"100\"]:.4f} ch@150={m[\"changed_acc\"]:.4f} decay={m[\"ood_decay_pct\"]:.4f}')" >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] SSM 10k eval done" >> $LOG

# ============================================================
# 5. 训练 shuffle 3k + eval OOD
# ============================================================
echo "[$(date +%H:%M:%S)] === 5. Train shuffle 3k ===" >> $LOG
rm -rf results_stage2/nexus_shuffle_3k
$PY -u train.py --model three_chain_mamba2 --config configs/sdd_mamba2_30m_nexus.yaml \
    --max_steps 3000 --batch_size 1 --seed 0 \
    --data_root ./data_sdd_split_nexus \
    --out_dir results_stage2/nexus_shuffle_3k --shuffle_labels >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] shuffle train done" >> $LOG

echo "[$(date +%H:%M:%S)] === Eval shuffle OOD ===" >> $LOG
$PY -u eval_ood.py \
    --checkpoint results_stage2/nexus_shuffle_3k/best.pt \
    --config configs/sdd_mamba2_30m_nexus.yaml \
    --data_root ./data/ood_T150_sdd_nexus --batch_size 1 --seed 0 \
    --out results_stage2/nexus_shuffle_3k/ood_metrics.json >> $LOG 2>&1
$PY -c "import json; m=json.load(open('results_stage2/nexus_shuffle_3k/ood_metrics.json')); print(f'shuffle ch@100={m[\"changed_acc_curve\"][\"100\"]:.4f} ch@150={m[\"changed_acc\"]:.4f} decay={m[\"ood_decay_pct\"]:.4f}')" >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] shuffle eval done" >> $LOG

# ============================================================
# 6. eval_ood_advanced (SSM + shuffle) + eval_negative_shadow (SSM)
# ============================================================
echo "[$(date +%H:%M:%S)] === 6a. eval_ood_advanced SSM ===" >> $LOG
$PY -u scripts_sdd/eval_ood_advanced.py \
    --checkpoint results_stage2/nexus_30m_seed0_10k/best.pt \
    --config configs/sdd_mamba2_30m_nexus.yaml \
    --data_root ./data/ood_T150_sdd_nexus \
    --out results_stage2/nexus_30m_seed0_10k/ood_metrics_advanced.json >> $LOG 2>&1

echo "[$(date +%H:%M:%S)] === 6b. eval_ood_advanced shuffle ===" >> $LOG
$PY -u scripts_sdd/eval_ood_advanced.py \
    --checkpoint results_stage2/nexus_shuffle_3k/best.pt \
    --config configs/sdd_mamba2_30m_nexus.yaml \
    --data_root ./data/ood_T150_sdd_nexus \
    --out results_stage2/nexus_shuffle_3k/ood_metrics_advanced.json >> $LOG 2>&1

echo "[$(date +%H:%M:%S)] === 6c. eval_negative_shadow SSM ===" >> $LOG
$PY -u scripts_sdd/eval_negative_shadow.py \
    --checkpoint results_stage2/nexus_30m_seed0_10k/best.pt \
    --config configs/sdd_mamba2_30m_nexus.yaml \
    --data_root ./data/ood_T150_sdd_nexus \
    --out results_stage2/nexus_30m_seed0_10k/negative_shadow.json >> $LOG 2>&1

touch /root/STAGE2_NEXUS_DONE
echo "[$(date +%H:%M:%S)] === STAGE2_NEXUS_DONE ===" >> $LOG
"""


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    sftp = c.open_sftp()

    # 1. 上传 config
    local_cfg = os.path.join(LOCAL_ROOT, "configs", "sdd_mamba2_30m_nexus.yaml")
    remote_cfg = f"{REMOTE_DIR}/configs/sdd_mamba2_30m_nexus.yaml"
    sftp.put(local_cfg, remote_cfg)
    print(f"[1] 上传 config: {remote_cfg}")

    # 2. 确保 scripts_sdd 目录存在 + 上传 eval 脚本 (覆盖, 保证最新版)
    try:
        sftp.mkdir(f"{REMOTE_DIR}/scripts_sdd")
    except IOError:
        pass
    for fn in ["eval_ood_advanced.py", "eval_negative_shadow.py"]:
        local_path = os.path.join(LOCAL_ROOT, "scripts_sdd", fn)
        remote_path = f"{REMOTE_DIR}/scripts_sdd/{fn}"
        sftp.put(local_path, remote_path)
        print(f"[2] 上传 {fn}: {remote_path}")

    # 3. 上传 shell
    with sftp.open('/root/stage2_nexus.sh', 'w') as f:
        f.write(RUN_SH)
    sftp.chmod('/root/stage2_nexus.sh', 0o755)
    print("[3] 上传 stage2_nexus.sh")

    # 4. 确认 GPU 空闲
    _, o, _ = c.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null')
    print(f"[4] GPU: {o.read().decode(errors='replace').strip()}")
    _, o, _ = c.exec_command('pgrep -af "train.py" 2>/dev/null | grep -v grep')
    ps = o.read().decode(errors='replace').strip()
    print(f"[5] 训练进程: {ps or '(无, 空闲)'}")

    # 5. 清理 marker + 后台启动
    c.exec_command('rm -f /root/STAGE2_NEXUS_DONE')
    time.sleep(1)
    c.exec_command('nohup bash /root/stage2_nexus.sh > /root/stage2_nexus.out 2>&1 &')
    time.sleep(8)

    _, o, _ = c.exec_command('cat /root/stage2_nexus.log 2>/dev/null')
    print("\n[6] 启动日志:")
    print(o.read().decode(errors='replace'))

    sftp.close()
    c.close()
    print("\n✓ 实验 2 (nexus) 已启动, 预计 ~5h 完成 (数据 gen ~10min + SSM 10k ~70min + shuffle 3k ~25min + evals ~30min)")
    print("  marker: /root/STAGE2_NEXUS_DONE")
    print("  log: /root/stage2_nexus.log")


if __name__ == "__main__":
    main()
