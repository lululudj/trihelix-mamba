import json
with open("/mnt/e/three_chain_v3/results_wsl/benchmark_scaled/single_scaled_s42/log.jsonl") as f:
    lines = f.readlines()
print(f"Lines: {len(lines)}")
for l in lines[-3:]:
    d = json.loads(l)
    print(d)
