"""分析训练日志中的 drift 分布，为 Eagle τ 设置提供依据。

从 log.jsonl 提取所有 step 级 drift 值，统计分布，
找出能让 5%/10%/20% 步触发回滚的 τ 阈值。
"""
import json
from pathlib import Path
import numpy as np

root = Path("results_wsl")
models = ["three_chain", "three_chain_bp_v2"]

for model in models:
    log_path = root / f"run_{model}_seed0" / "log.jsonl"
    if not log_path.exists():
        print(f"{model}: log 不存在")
        continue
    drifts = []
    taus = []
    for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
        if "drift" in rec:
            drifts.append(rec["drift"])
        if "tau" in rec:
            taus.append(rec["tau"])
    drifts = np.array(drifts)
    print(f"\n=== {model} drift 分布 ===")
    print(f"  样本数: {len(drifts)}")
    print(f"  min={drifts.min():.4f}, max={drifts.max():.4f}, mean={drifts.mean():.4f}, std={drifts.std():.4f}")
    print(f"  分位数: 50%={np.percentile(drifts,50):.4f}, 75%={np.percentile(drifts,75):.4f}, "
          f"90%={np.percentile(drifts,90):.4f}, 95%={np.percentile(drifts,95):.4f}, "
          f"99%={np.percentile(drifts,99):.4f}")
    # 找触发回滚的 τ
    for pct in [5, 10, 20, 30]:
        tau = np.percentile(drifts, 100 - pct)
        print(f"  τ={tau:.4f} → 约 {pct}% 步触发回滚")
    if taus:
        print(f"  τ 调度范围: {min(taus):.2f} → {max(taus):.2f}（训练中实际退火）")

# 也看 eval 阶段的 drift（更接近推理场景）
print("\n=== eval 阶段 drift（test 集评估时）===")
for model in models:
    m_path = root / f"run_{model}_seed0" / "metrics.json"
    if m_path.exists():
        d = json.loads(m_path.read_text(encoding="utf-8"))
        print(f"  {model}: drift_mean={d.get('drift_mean', 'N/A')}, rollback_rate={d.get('rollback_rate', 'N/A')}")
