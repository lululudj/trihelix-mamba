#!/bin/bash
# Full benchmark: 15 seeds x 3 models + 5 ablation
# All output to files, no stdout blocking

cd /mnt/e/three_chain_v3
rm -rf results_wsl/v2_benchmark
mkdir -p results_wsl/v2_benchmark
LOG=results_wsl/v2_benchmark/progress.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

log "=== BENCHMARK START ==="

MODELS=("three_chain" "single_chain" "transformer")
CONFIGS=("configs/matched_three.yaml" "configs/matched_single.yaml" "configs/matched_transformer.yaml")
OOD_DIRS=("data/ood_T150" "data/ood_T200_noise" "data/ood_T250_mask")
OOD_LABELS=("T150" "T200n" "T250m")

SEEDS=(42 123 456 789 1024 1111 1337 2048 31337 42069 55555 66666 77777 88888 99999)

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  cfg="${CONFIGS[$i]}"
  for seed in "${SEEDS[@]}"; do
    NAME="${model}_s${seed}"
    OUT="results_wsl/v2_benchmark/${NAME}"
    if [ -f "${OUT}/done.flag" ]; then
      log "[SKIP] ${NAME}"
      continue
    fi
    rm -rf "${OUT}"
    mkdir -p "${OUT}"
    
    log "=== TRAIN ${NAME} ==="
    python3 -c "import torch; torch.cuda.empty_cache()"
    python3 train.py --model ${model} --config ${cfg} --seed ${seed} --max_steps 2000 --batch_size 16 --eval_every 500 --log_every 100 --out_dir ${OUT} >> "${OUT}/run.log" 2>&1
    RC=$?
    python3 -c "import torch; torch.cuda.empty_cache()"
    
    if [ -f "${OUT}/best.pt" ]; then
      # Standard eval
      log "  EVAL std ${NAME}"
      python3 eval.py --checkpoint "${OUT}/best.pt" --config ${cfg} --data_root data_split --out "${OUT}/metrics.json" >> "${OUT}/run.log" 2>&1
      
      # OOD evals at 3 levels
      for j in "${!OOD_DIRS[@]}"; do
        ood_dir="${OOD_DIRS[$j]}"
        ood_label="${OOD_LABELS[$j]}"
        log "  OOD ${ood_label} ${NAME}"
        python3 eval_ood.py --checkpoint "${OUT}/best.pt" --config ${cfg} --data_root ${ood_dir} --out "${OUT}/ood_${ood_label}.json" >> "${OUT}/run.log" 2>&1
      done
      
      touch "${OUT}/done.flag"
      log "[OK] ${NAME}"
    else
      log "[FAIL] ${NAME} (rc=$RC, no best.pt)"
    fi
  done
done

log "=== ABLATION START ==="
ABL_MODELS=("concat_mamba" "single_chain")
ABL_LABELS=("concat" "single")
ABL_SEEDS=(42 123 456 789 1024)

for i in "${!ABL_MODELS[@]}"; do
  model="${ABL_MODELS[$i]}"
  label="${ABL_LABELS[$i]}"
  for seed in "${ABL_SEEDS[@]}"; do
    NAME="abl_${label}_s${seed}"
    OUT="results_wsl/v2_benchmark/${NAME}"
    if [ -f "${OUT}/done.flag" ]; then
      log "[SKIP] ${NAME}"
      continue
    fi
    rm -rf "${OUT}"
    mkdir -p "${OUT}"
    
    log "=== TRAIN ${NAME} ==="
    python3 -c "import torch; torch.cuda.empty_cache()"
    python3 train.py --model ${model} --seed ${seed} --max_steps 2000 --batch_size 16 --eval_every 500 --log_every 100 --out_dir ${OUT} >> "${OUT}/run.log" 2>&1
    python3 -c "import torch; torch.cuda.empty_cache()"
    
    if [ -f "${OUT}/best.pt" ]; then
      python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json" >> "${OUT}/run.log" 2>&1
      for j in "${!OOD_DIRS[@]}"; do
        python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root "${OOD_DIRS[$j]}" --out "${OUT}/ood_${OOD_LABELS[$j]}.json" >> "${OUT}/run.log" 2>&1
      done
      touch "${OUT}/done.flag"
      log "[OK] ${NAME}"
    else
      log "[FAIL] ${NAME}"
    fi
  done
done

log "=== ALL DONE ==="
