#!/bin/bash
# Collect all benchmark results and generate summary JSON
cd /mnt/e/three_chain_v3

echo "{" > results_wsl/benchmark_multiseed/all_results.json
echo '  "benchmark": "multi_seed_v1",' >> results_wsl/benchmark_multiseed/all_results.json
echo '  "seeds": [42, 123, 456, 789, 1024],' >> results_wsl/benchmark_multiseed/all_results.json
echo '  "models": {' >> results_wsl/benchmark_multiseed/all_results.json

FIRST_MODEL=true
for model in three_chain single_chain transformer; do
  if [ "$FIRST_MODEL" = true ]; then FIRST_MODEL=false; else echo "," >> results_wsl/benchmark_multiseed/all_results.json; fi
  echo '    "'$model'": {' >> results_wsl/benchmark_multiseed/all_results.json
  echo '      "runs": [' >> results_wsl/benchmark_multiseed/all_results.json
  
  FIRST_SEED=true
  for seed in 42 123 456 789 1024; do
    METRICS="results_wsl/benchmark_multiseed/${model}_seed${seed}/metrics.json"
    if [ -f "$METRICS" ]; then
      if [ "$FIRST_SEED" = true ]; then FIRST_SEED=false; else echo "," >> results_wsl/benchmark_multiseed/all_results.json; fi
      python3 -c "
import json
d = json.load(open('$METRICS'))
out = {
    'seed': $seed,
    'acc_final': d.get('acc_final', 0),
    'changed_acc': d.get('changed_acc', 0),
    'params': d.get('params', 0),
}
print('        ' + json.dumps(out))
" >> results_wsl/benchmark_multiseed/all_results.json
    fi
  done
  
  echo '      ]' >> results_wsl/benchmark_multiseed/all_results.json
  echo '    }' >> results_wsl/benchmark_multiseed/all_results.json
done

echo '  }' >> results_wsl/benchmark_multiseed/all_results.json
echo '}' >> results_wsl/benchmark_multiseed/all_results.json

echo "Results collected to results_wsl/benchmark_multiseed/all_results.json"
