#!/bin/bash
cd /mnt/e/three_chain_v3
python3 -u train.py --model single_chain --seed 42 --max_steps 20 --batch_size 4 --log_every 5 --out_dir /tmp/test_sc3 > /tmp/train_stdout.log 2> /tmp/train_stderr.log &
PID=$!
sleep 30
kill $PID 2>/dev/null
echo "=== STDOUT ==="
cat /tmp/train_stdout.log
echo "=== STDERR ==="
cat /tmp/train_stderr.log
echo "=== DONE ==="
