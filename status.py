import json, glob, os

scaled_dir = '/mnt/e/three_chain_v3/results_wsl/benchmark_scaled'
ablation_dir = '/mnt/e/three_chain_v3/results_wsl/ablation'

# Count completed
scaled_done = len(glob.glob(f'{scaled_dir}/*/metrics.json'))
scaled_trained = len(glob.glob(f'{scaled_dir}/*/best.pt'))
ablation_done = len(glob.glob(f'{ablation_dir}/*/metrics.json'))

# Check active step
log_files = glob.glob(f'{scaled_dir}/*/log.jsonl')
for lf in log_files:
    try:
        lines = open(lf).readlines()
        if lines:
            last = json.loads(lines[-1])
            name = lf.split('/')[-2]
            print(f'Active: {name} step={last.get("step","?")}/{3000} elapsed={last.get("elapsed","?")}s')
    except: pass

print(f'Scaled done: {scaled_done}/10')
print(f'Scaled trained: {scaled_trained}/10')
print(f'Ablation done: {ablation_done}/15')
