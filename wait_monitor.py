"""监控训练进度，每 30 秒打印一次 Python 进程状态和最新 log。"""
import time
import subprocess
from pathlib import Path

log_file = Path("results/run_three_chain_seed0/log.jsonl")
single_log = Path("results/run_single_chain_seed0/log.jsonl")
trans_log = Path("results/run_transformer_seed0/log.jsonl")

for i in range(40):  # 40 * 30s = 20 分钟
    time.sleep(30)
    print(f"\n=== [{i+1}/40] {time.strftime('%H:%M:%S')} ===")

    # 检查 python 进程
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU | Format-Table -AutoSize"],
            capture_output=True, text=True, timeout=10
        )
        print("Python 进程:")
        print(result.stdout.strip())
    except Exception as e:
        print(f"  (进程检查失败: {e})")

    # 检查 three_chain log
    if log_file.exists():
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        print(f"three_chain 最新: {lines[-1][:120]}")

    # 检查 single_chain log
    if single_log.exists():
        lines = single_log.read_text(encoding="utf-8").strip().split("\n")
        print(f"single_chain 最新: {lines[-1][:120]}")

    # 检查 transformer log
    if trans_log.exists():
        lines = trans_log.read_text(encoding="utf-8").strip().split("\n")
        print(f"transformer 最新: {lines[-1][:120]}")

    # 检查 summary
    for name in ["three_chain", "single_chain", "transformer"]:
        s = Path(f"results/run_{name}_seed0/summary.json")
        if s.exists():
            print(f"!!! {name} summary 已生成: {s.read_text(encoding='utf-8')[:200]}")
            print(f"!!! 所有训练可能已完成，退出监控")
            import sys
            sys.exit(0)
