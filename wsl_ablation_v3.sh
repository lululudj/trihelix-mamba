#!/bin/bash
PIDFILE=/tmp/ablation_v3.pid
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
      echo "[SKIP] ${NAME}" >> results_wsl/ablation/progress.log
      continue
    fi
    rm -rf "${OUT}"
    mkdir -p "${OUT}"
    echo "=== [${NAME}] START: $(date) ===" | tee -a results_wsl/ablation/progress.log
    python3 -c "import torch; torch.cuda.empty_cache()"
    python3 train.py --model ${ABL_MODELS[$i]} --seed ${seed} --max_steps 2000 --batch_size 16 --eval_every 500 --log_every 100 --out_dir ${OUT} >> "${OUT}/run.log" 2>&1
    RC=$?
    python3 -c "import torch; torch.cuda.empty_cache()"
    if [ $RC -ne 0 ]; then
      echo "TRAIN FAILED (rc=$RC): ${NAME}" | tee -a results_wsl/ablation/progress.log
    fi
    if [ -f "${OUT}/best.pt" ]; then
      echo "EVAL ${NAME}..." | tee -a results_wsl/ablation/progress.log
      python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json" >> "${OUT}/run.log" 2>&1
      python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data/ood_T150 --out "${OUT}/ood_metrics.json" >> "${OUT}/run.log" 2>&1
    else
      echo "[WARN] No best.pt for ${NAME}" | tee -a results_wsl/ablation/progress.log
    fi
    python3 -c "import torch; torch.cuda.empty_cache()"
    echo "=== [${NAME}] DONE: $(date) ===" | tee -a results_wsl/ablation/progress.log
  done
done

rm -f "$PIDFILE"
echo "=== ALL DONE: $(date) ===" | tee -a results_wsl/ablation/progress.log
