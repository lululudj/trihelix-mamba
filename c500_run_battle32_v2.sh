#!/bin/bash
# C500 (沐曦 MXMACA) 32场景实验启动脚本 - 加固版v2
# 死锁防护策略:
#   1. Python代码中每步torch.cuda.synchronize() (不全局阻塞kernel, 只防队列溢出)
#   2. 崩溃自动重试(断点续跑)
#   3. 启动时清理僵尸进程
#   4. BATCH_SIZE=1 防OOM
#   5. 连续5个错误后GPU状态重置

source /opt/maca/env.sh 2>/dev/null
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib
export MAMBA3_FORCE_REF=1
export MAMBA3_FORCE_GPU_SCAN=1
export MAMBA3_USE_TRITON_SCAN=0
export PYTORCH_MACA_ALLOC_CONF=max_split_size_mb:128

cd /root/three_chain_v3

LOG=battle32_c500_log.txt
echo "" >> $LOG
echo "=== C500 32-Scenario Battle (BP v2.1) ===" >> $LOG
echo "Start: $(date -u) UTC (PID=$$)" >> $LOG
echo "MACA_PATH: $MACA_PATH" >> $LOG
echo "MAMBA3_FORCE_REF: 1 (GPU mamba3_ref)" >> $LOG
echo "MAMBA3_USE_TRITON_SCAN: 0 (纯Python GPU scan最优)" >> $LOG
echo "BP version: v2.1 (修复bp1_gate和bp2_reset乘法死锁)" >> $LOG
echo "Python sync: per-step cuda.synchronize()" >> $LOG

# 清理残留进程
echo "清理残留进程..." >> $LOG
pkill -9 -f battle32_c500_gpu.py 2>/dev/null
pkill -9 -f battle32_gpu.py 2>/dev/null
sleep 3

# 确认85个结果还在
NRES=$(python3 -c "import json; r=json.load(open('battle32_results.json')); print(len(r))" 2>/dev/null || echo "0")
echo "已有结果数: $NRES (断点续跑)" >> $LOG

MAX_RETRIES=10
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "" >> $LOG
    echo "=== Attempt $((RETRY+1))/$MAX_RETRIES at $(date -u) UTC ===" >> $LOG

    setsid python3 -u battle32_c500_gpu.py >> $LOG 2>&1
    RC=$?

    # 检查是否完成 (battle32_c500_gpu.py正常结束会输出COMPLETE)
    if grep -q "C500 EXPERIMENT COMPLETE" $LOG 2>/dev/null; then
        echo "=== EXPERIMENT COMPLETE! RC=0 at $(date -u) UTC ===" >> $LOG
        break
    fi

    echo "battle32_c500_gpu.py exited RC=$RC at $(date -u) UTC (incomplete)" >> $LOG
    RETRY=$((RETRY+1))

    if [ $RETRY -lt $MAX_RETRIES ]; then
        echo "等待15秒后重试($RETRY/$MAX_RETRIES)..." >> $LOG
        sleep 15
        pkill -9 -f battle32_c500_gpu.py 2>/dev/null
        pkill -9 -f battle32_gpu.py 2>/dev/null
        sleep 5
        # GPU reset via mx-smi if available
        mx-smi -r 2>/dev/null && echo "GPU reset via mx-smi" >> $LOG
        sleep 3
    fi
done

if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "=== 已达最大重试次数 $MAX_RETRIES, 停止 ===" >> $LOG
fi

echo "=== Running stats_analysis.py ===" >> $LOG
python3 -u stats_analysis.py >> $LOG 2>&1
echo "=== ALL DONE at $(date -u) UTC ===" >> $LOG
