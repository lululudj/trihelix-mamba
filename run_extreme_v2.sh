#!/bin/bash
# V2 极限测试: C2 长训 + B2 超深 + D2 retry
# 三阶段顺序执行, 每阶段完成写 marker, 编排器轮询 marker 后立即下载
LOG=/root/extreme_v2.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%F %T')] === V2 极限测试开始 ===" > $LOG

# ====== Phase 1: C2 1B 长训 2000 步 ======
echo "" >> $LOG
echo "[$(date '+%F %T')] === Phase 1: C2 1B 2000 steps (lr=1e-4, warmup=1000) ===" >> $LOG
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_1000m_ckpt_long.yaml \
    --max_steps 2000 --batch_size 1 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_1000m_ckpt_long_seed0_2000 >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] C2 训练结束 rc=$TRAIN_RC" >> $LOG

if [ $TRAIN_RC -eq 0 ]; then
    # C2 eval T150
    python -u eval_ood.py \
        --checkpoint results/run_1000m_ckpt_long_seed0_2000/final.pt \
        --config configs/matched_mamba2_1000m_ckpt_long.yaml \
        --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
        --out results/run_1000m_ckpt_long_seed0_2000/ood_metrics.json >> $LOG 2>&1
    echo "[$(date '+%F %T')] C2 T150 eval 完成" >> $LOG

    # C2 eval T300 (用 --extend_max_T 1024 扩展 time_embed)
    python -u eval_ood.py \
        --checkpoint results/run_1000m_ckpt_long_seed0_2000/final.pt \
        --config configs/matched_mamba2_1000m_ckpt_long.yaml \
        --data_root ./data/ood_T300 --batch_size 1 --seed 0 \
        --extend_max_T 1024 \
        --out results/run_1000m_ckpt_long_seed0_2000/ood_A_T300.json >> $LOG 2>&1
    echo "[$(date '+%F %T')] C2 T300 eval 完成" >> $LOG

    # 删除 checkpoint (省空间, JSON 已写完)
    rm -f results/run_1000m_ckpt_long_seed0_2000/final.pt
    rm -f results/run_1000m_ckpt_long_seed0_2000/best.pt
    echo "[$(date '+%F %T')] C2 checkpoint 已删除" >> $LOG
fi
touch /root/C2_ALL_DONE
echo "[$(date '+%F %T')] === C2_ALL_DONE ===" >> $LOG

# ====== Phase 2: B2 n_layers=8 超深 ======
echo "" >> $LOG
echo "[$(date '+%F %T')] === Phase 2: B2 n_layers=8, 1000 steps ===" >> $LOG
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_30m_deep8.yaml \
    --max_steps 1000 --batch_size 2 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_30m_deep_n8_seed0_1000 >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] B2 训练结束 rc=$TRAIN_RC" >> $LOG

if [ $TRAIN_RC -eq 0 ]; then
    # B2 eval T150, T300, T500 (用 --extend_max_T 1024)
    for T in 150 300 500; do
        python -u eval_ood.py \
            --checkpoint results/run_30m_deep_n8_seed0_1000/final.pt \
            --config configs/matched_mamba2_30m_deep8.yaml \
            --data_root ./data/ood_T${T} --batch_size 1 --seed 0 \
            --extend_max_T 1024 \
            --out results/run_30m_deep_n8_seed0_1000/ood_A_T${T}.json >> $LOG 2>&1
        echo "[$(date '+%F %T')] B2 T${T} eval 完成" >> $LOG
    done
    # T150 也复制一份叫 ood_T150.json (与其他 run 命名一致)
    cp results/run_30m_deep_n8_seed0_1000/ood_A_T150.json \
       results/run_30m_deep_n8_seed0_1000/ood_T150.json
    rm -f results/run_30m_deep_n8_seed0_1000/final.pt
    rm -f results/run_30m_deep_n8_seed0_1000/best.pt
    echo "[$(date '+%F %T')] B2 checkpoint 已删除" >> $LOG
fi
touch /root/B2_ALL_DONE
echo "[$(date '+%F %T')] === B2_ALL_DONE ===" >> $LOG

# ====== Phase 3: D2 T250mask retry (重训 100M) ======
echo "" >> $LOG
echo "[$(date '+%F %T')] === Phase 3: 重训 100M + D2 T250mask retry ===" >> $LOG
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_100m_maxT1024.yaml \
    --max_steps 500 --batch_size 2 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_100m_maxT1024_v2_seed0_500 >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] 100M 重训结束 rc=$TRAIN_RC" >> $LOG

if [ $TRAIN_RC -eq 0 ]; then
    # D2 retry: T250_mask with batch=1 (上次崩溃, 这次 batch=1 更稳)
    python -u eval_ood.py \
        --checkpoint results/run_100m_maxT1024_v2_seed0_500/final.pt \
        --config configs/matched_mamba2_100m_maxT1024.yaml \
        --data_root ./data/ood_T250_mask --batch_size 1 --seed 0 \
        --out results/run_100m_maxT1024_v2_seed0_500/ood_D_T250mask.json >> $LOG 2>&1
    echo "[$(date '+%F %T')] D2 T250mask retry 完成" >> $LOG
    rm -f results/run_100m_maxT1024_v2_seed0_500/final.pt
    rm -f results/run_100m_maxT1024_v2_seed0_500/best.pt
    echo "[$(date '+%F %T')] 100M v2 checkpoint 已删除" >> $LOG
fi
touch /root/D2_ALL_DONE
echo "[$(date '+%F %T')] === D2_ALL_DONE ===" >> $LOG

echo "" >> $LOG
echo "[$(date '+%F %T')] === V2_ALL_DONE ===" >> $LOG
