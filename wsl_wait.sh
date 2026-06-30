#!/bin/bash
# 等待三模型训练完成
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"

echo "等待三模型训练完成..."
deadline=$(( $(date +%s) + 600 ))  # 最多等 10 分钟

while [ $(date +%s) -lt $deadline ]; do
    all_done=true
    for m in three_chain single_chain transformer; do
        f="results_wsl/run_${m}_seed0/summary.json"
        if [ ! -f "$f" ]; then
            all_done=false
        fi
    done
    if [ "$all_done" = true ]; then
        echo "=== 所有训练完成 $(date) ==="
        for m in three_chain single_chain transformer; do
            echo "--- $m ---"
            cat "results_wsl/run_${m}_seed0/summary.json"
            echo ""
        done
        exit 0
    fi
    # 打印进度
    echo "[$(date +%H:%M:%S)] 进度:"
    for m in three_chain single_chain transformer; do
        f="results_wsl/run_${m}_seed0/summary.json"
        log="results_wsl/run_${m}_seed0/log.jsonl"
        if [ -f "$f" ]; then
            echo "  $m: DONE"
        elif [ -f "$log" ]; then
            last=$(tail -1 "$log" 2>/dev/null)
            echo "  $m: $last"
        else
            echo "  $m: 未开始"
        fi
    done
    sleep 60
done

echo "超时退出"
exit 1
