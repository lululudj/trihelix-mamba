import json
with open("/mnt/e/three_chain_v3/results_wsl/benchmark_scaled/single_scaled_s42/log.jsonl") as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")
for l in lines[-5:]:
    d = json.loads(l)
    print(f"  step={d['step']} elapsed={d['elapsed']}s loss={d['loss']:.3f}")
