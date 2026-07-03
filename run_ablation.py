import subprocess, sys, os, json, time, torch, shutil
from pathlib import Path

BASE = Path("/mnt/e/three_chain_v3")
OUT_BASE = BASE / "results_wsl" / "ablation"
OUT_BASE.mkdir(parents=True, exist_ok=True)

MODELS = ["three_chain", "three_chain_no_eagle", "three_chain_no_bind", "concat_mamba", "single_chain"]
LABELS = ["full", "no_eagle", "no_bind", "concat", "single"]
SEEDS = [42, 123, 456]

def run(cmd, log_file):
    with open(log_file, "a") as f:
        f.write(f"\n=== CMD: {' '.join(cmd)} ===\n")
        f.flush()
    result = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True)
    with open(log_file, "a") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n" + result.stderr)
    return result.returncode

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
        time.sleep(1)
        
        train_cmd = [
            sys.executable, str(BASE / "train.py"),
            "--model", model,
            "--seed", str(seed),
            "--max_steps", "2000",
            "--batch_size", "8",
            "--eval_every", "500",
            "--log_every", "100",
            "--out_dir", str(out_dir),
        ]
        rc = run(train_cmd, run_log)
        torch.cuda.empty_cache()
        time.sleep(1)
        
        best_pt = out_dir / "best.pt"
        if best_pt.exists():
            eval_cmd = [
                sys.executable, str(BASE / "eval.py"),
                "--checkpoint", str(best_pt),
                "--data_root", str(BASE / "data_split"),
                "--out", str(out_dir / "metrics.json"),
            ]
            run(eval_cmd, run_log)
            
            ood_cmd = [
                sys.executable, str(BASE / "eval_ood.py"),
                "--checkpoint", str(best_pt),
                "--data_root", str(BASE / "data/ood_T150"),
                "--out", str(out_dir / "ood_metrics.json"),
            ]
            run(ood_cmd, run_log)
        else:
            msg = f"[WARN] No best.pt for {name} (rc={rc})"
            print(msg)
            with open(progress_log, "a") as f:
                f.write(msg + "\n")
        
        torch.cuda.empty_cache()
        time.sleep(1)
        msg = f"=== [{name}] DONE: {time.ctime()} ==="
        print(msg)
        with open(progress_log, "a") as f:
            f.write(msg + "\n")

print("=== ALL DONE ===")
with open(progress_log, "a") as f:
    f.write(f"=== ALL DONE: {time.ctime()} ===\n")
