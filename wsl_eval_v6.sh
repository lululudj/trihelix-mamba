#!/bin/bash
cd /mnt/e/three_chain_v3
echo "=== V6 Eval ==="
python3 eval.py --checkpoint results_wsl/run_hetero_v6_seed0/best.pt --data_root data_split --out results_wsl/run_hetero_v6_seed0/metrics.json 2>&1 | grep -E "acc_final|changed_acc|params"
echo ""
echo "=== V6 OOD ==="
python3 eval_ood.py --checkpoint results_wsl/run_hetero_v6_seed0/best.pt --out results_wsl/run_hetero_v6_seed0/ood_metrics.json 2>&1 | grep -E "acc_final|changed_acc|OOD|decay"
echo ""
echo "=== V6 Exp1 ==="
python3 killer_experiments/exp1_chain_ablation/run_exp1.py --checkpoint results_wsl/run_hetero_v6_seed0/best.pt --data_root data_split --out_dir results_wsl/exp1_v6 2>&1 | grep -E "Ablate|Normal|collapse"
