import json
path = "/mnt/e/three_chain_v3/results_wsl/benchmark_scaled/single_scaled_s42/log.jsonl"
try:
    lines = open(path).readlines()
    if lines:
        last = json.loads(lines[-1])
        print(f"step={last.get('step','?')}, elapsed={last.get('elapsed','?')}s, loss={last.get('loss','?'):.4f}")
        print(f"Progress: {last.get('step',0)}/3000")
    else:
        print("empty log")
except Exception as e:
    print(f"Error: {e}")
