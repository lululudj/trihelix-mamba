#!/bin/bash
# Quick progress check for benchmark
cd /mnt/e/three_chain_v3
echo "=== Benchmark Progress ==="
echo "Time: $(date)"
echo ""

# Count completed runs
COMPLETED=0
TOTAL=15
for model in three_chain single_chain transformer; do
  for seed in 42 123 456 789 1024; do
    METRICS="results_wsl/benchmark_multiseed/${model}_seed${seed}/metrics.json"
    if [ -f "$METRICS" ]; then
      ACC=$(python3 -c "import json; print(f\"{json.load(open('$METRICS')).get('acc_final',0):.4f}\")" 2>/dev/null || echo "?")
      echo "  [DONE] ${model}_seed${seed}  acc=${ACC}"
      COMPLETED=$((COMPLETED + 1))
    elif [ -f "results_wsl/benchmark_multiseed/${model}_seed${seed}/best.pt" ]; then
      echo "  [TRAINED] ${model}_seed${seed} (eval pending)"
    fi
  done
done

# Check currently running
echo ""
echo "Running processes:"
ps aux | grep "python3 train.py" | grep -v grep | while read line; do
  OUTDIR=$(echo "$line" | grep -oP 'results_wsl/benchmark_multiseed/\S+' | head -1)
  echo "  ACTIVE: $OUTDIR"
done

echo ""
echo "Progress: ${COMPLETED}/${TOTAL} completed"
