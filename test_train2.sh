#!/bin/bash
cd /mnt/e/three_chain_v3
rm -rf /tmp/test_sc3 /tmp/test_sc4
# Check data directory
echo "data_split check:"
ls data_split/*.pt 2>/dev/null | head -5
echo "data_split count: $(ls data_split/*.pt 2>/dev/null | wc -l)"
echo "data/ood_T150 check:"
ls data/ood_T150/*.pt 2>/dev/null | head -5
echo "---"
python3 -u train.py --model single_chain --seed 42 --max_steps 20 --batch_size 4 --log_every 5 --out_dir /tmp/test_sc4 > /tmp/train2_stdout.log 2> /tmp/train2_stderr.log &
PID=$!
sleep 45
kill $PID 2>/dev/null
echo "=== STDOUT ==="
cat /tmp/train2_stdout.log
echo "=== STDERR ==="
cat /tmp/train2_stderr.log
