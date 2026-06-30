"""持续监控直到三模型训练全部完成（summary.json 全部生成）。"""
import time
import sys
from pathlib import Path

root = Path("results")
models = ["three_chain", "single_chain", "transformer"]

deadline = time.time() + 3600  # 最多等 1 小时
last_print = 0

while time.time() < deadline:
    # 检查是否所有 summary 都生成了
    summaries = [Path(f"results/run_{m}_seed0/summary.json") for m in models]
    done = [s.exists() for s in summaries]

    now = time.time()
    if now - last_print > 30:
        last_print = now
        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}] 状态检查:")
        for m, s, d in zip(models, summaries, done):
            if d:
                content = s.read_text(encoding="utf-8")
                print(f"  {m}: DONE - {content[:150]}")
            else:
                log = Path(f"results/run_{m}_seed0/log.jsonl")
                if log.exists():
                    lines = log.read_text(encoding="utf-8").strip().split("\n")
                    print(f"  {m}: 训练中 - {lines[-1][:100]}")
                else:
                    print(f"  {m}: 未开始")

    if all(done):
        print(f"\n!!! 所有训练完成 !!!")
        for m, s in zip(models, summaries):
            print(f"\n=== {m} summary ===")
            print(s.read_text(encoding="utf-8"))
        sys.exit(0)

    time.sleep(15)

print("\n超时退出")
sys.exit(1)
