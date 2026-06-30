#!/bin/bash
PIDFILE=/tmp/ablation_fixed.pid
if [ -f "$PIDFILE" ]; then
  OLDPID=$(cat "$PIDFILE")
  kill $OLDPID 2>/dev/null
  killall -9 python3 2>/dev/null
  sleep 2
fi
echo $$ > "$PIDFILE"

cd /mnt/e/three_chain_v3
python3 -c "import torch; torch.cuda.empty_cache()"
mkdir -p results_wsl/ablation

ABL_MODELS=(three_chain three_chain_no_eagle three_chain_no_bind concat_mamba single_chain)
ABL_LABELS=(full no_eagle no_bind concat single)
ABL_SEEDS=(42 123 456)

for i in "${!ABL_MODELS[@]}"; do
  for seed in "${ABL_SEEDS[@]}"; do
    NAME="abl_${ABL_LABELS[$i]}_s${seed}"
    OUT="results_wsl/ablation/${NAME}"
    if [ -f "${OUT}/ood_metrics.json" ]; then
      echo "[SKIP] ${NAME} - already complete" >> results_wsl/ablation/progress.log
      continue
    fi
    rm -rf "${OUT}"
    echo "=== [${NAME}] START: $(date) ===" >> results_wsl/ablation/progress.log
    python3 -c "import torch; torch.cuda.empty_cache()"
    python3 train.py --model ${ABL_MODELS[$i]} --seed ${seed} --max_steps 2000 --batch_size 16 --eval_every 500 --log_every 100 --out_dir ${OUT} >> "${OUT}/run.log" 2>&1 || echo "TRAIN FAILED: ${NAME}" >> results_wsl/ablation/progress.log
    python3 -c "import torch; torch.cuda.empty_cache()"
    if [ -f "${OUT}/best.pt" ]; then
      python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json" >> "${OUT}/run.log" 2>&1 || echo "EVAL FAILED: ${NAME}" >> results_wsl/ablation/progress.log
      python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data/ood_T150 --out "${OUT}/ood_metrics.json" >> "${OUT}/run.log" 2>&1 || echo "OOD FAILED: ${NAME}" >> results_wsl/ablation/progress.log
    else
      echo "[WARN] No best.pt for ${NAME}" >> results_wsl/ablation/progress.log
    fi
    python3 -c "import torch; torch.cuda.empty_cache()"
    echo "=== [${NAME}] DONE: $(date) ===" >> results_wsl/ablation/progress.log
  done
done

rm -f "$PIDFILE"
echo "=== ALL ABLATION DONE: $(date) ===" >> results_wsl/ablation/progress.log
