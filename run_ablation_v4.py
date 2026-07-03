import subprocess, sys, os, time, torch, shutil
from pathlib import Path

BASE = Path("/mnt/e/three_chain_v3")
OUT_BASE = BASE / "results_wsl" / "ablation"
OUT_BASE.mkdir(parents=True, exist_ok=True)

MODELS = ["concat_mamba", "single_chain"]
LABELS = ["concat", "single"]
SEEDS = [42, 123, 456]

progress_log = OUT_BASE / "progress.log"

for i, model in enumerate(MODELS):
    for seed in SEEDS:
        name = f"abl_{LABELS[i]}_s{seed}"
        out_dir = OUT_BASE / name
        ood_file = out_dir / "ood_metrics.json"
        
        if ood_file.exists():
            msg = f"[SKIP] {name}"
            print(msg)
            with open(progress_log, "a") as f:
                f.write(msg + "\n")
            continue
        
        if out_dir.exists():
            shutil.rmtree(str(out_dir))
        out_dir.mkdir(parents=True)
        
        run_log = out_dir / "run.log"
        msg = f"=== [{name}] START: {time.ctime()} ==="
        print(msg)
        with open(progress_log, "a") as f:
            f.write(msg + "\n")
        
        torch.cuda.empty_cache()
        time.sleep(2)
        
        # Use shell redirect, NOT capture_output (which blocks on pipe buffer)
        train_cmd = f"'{sys.executable}' '{BASE}/train.py' --model {model} --seed {seed} --max_steps 2000 --batch_size 16 --eval_every 500 --log_every 100 --out_dir '{out_dir}' >> '{run_log}' 2>&1"
        rc = subprocess.run(train_cmd, shell=True, cwd=str(BASE)).returncode
        torch.cuda.empty_cache()
        time.sleep(2)
        
        best_pt = out_dir / "best.pt"
        if best_pt.exists():
            eval_cmd = f"'{sys.executable}' '{BASE}/eval.py' --checkpoint '{best_pt}' --data_root '{BASE}/data_split' --out '{out_dir}/metrics.json' >> '{run_log}' 2>&1"
            subprocess.run(eval_cmd, shell=True, cwd=str(BASE))
            
            ood_cmd = f"'{sys.executable}' '{BASE}/eval_ood.py' --checkpoint '{best_pt}' --data_root '{BASE}/data/ood_T150' --out '{out_dir}/ood_metrics.json' >> '{run_log}' 2>&1"
            subprocess.run(ood_cmd, shell=True, cwd=str(BASE))
            
            msg = f"[OK] {name} DONE"
            print(msg)
        else:
            msg = f"[WARN] No best.pt for {name} (rc={rc})"
            print(msg)
        
        with open(progress_log, "a") as f:
            f.write(msg + "\n")
        torch.cuda.empty_cache()
        time.sleep(1)

print("=== ALL DONE ===")
with open(progress_log, "a") as f:
    f.write(f"=== ALL DONE: {time.ctime()} ===\n")
