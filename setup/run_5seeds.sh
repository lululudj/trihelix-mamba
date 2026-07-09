#!/bin/bash
# setup/run_5seeds.sh — 5-seed 完整基准 (论文 Table 复现)
#
# 用法 (在 WSL Ubuntu 里, conda env mamba 已激活):
#   cd /mnt/e/three_chain_v3
#   bash setup/run_5seeds.sh
#
# 对 5 个 seed [42, 123, 456, 789, 1024] 各跑一次:
#   1. train.py 训练 three_chain_mamba2 (2000 步)
#   2. eval_ood.py 评估 OOD (T=150)
#   3. 收集 changed_acc 到 ood_metrics.json
# 最后算 5 个 seed 的 changed_acc 均值/标准差
#
# 期望 (来自 README.md Table):
#   ThreeChainMamba2 OOD changed_acc@T=150 = 0.5952 ± 0.0034
#   OOD decay ≈ -0.0042 (≈ 0)
#
# 估计耗时: RTX 4060 ~1-1.5 小时 (5 × ~15 分钟)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 自动激活 conda env mamba
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "mamba" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mamba
fi

echo "=== 5-seed 完整基准 ==="
echo "项目根: $PROJECT_ROOT"
echo

# 确保训练数据
if [ ! -d data_split ] || [ "$(find data_split -name '*.npz' | wc -l)" -eq 0 ]; then
    echo "训练数据不存在, 调用 gen_data.sh..."
    bash setup/gen_data.sh
fi

MODEL="three_chain_mamba2"
CONFIG="configs/matched_mamba2.yaml"
MAX_STEPS=2000
BENCH_DIR="results_wsl/benchmark_mamba2"
SEEDS=(42 123 456 789 1024)

mkdir -p "$BENCH_DIR"

echo "模型: $MODEL (3.26M)"
echo "配置: $CONFIG"
echo "seeds: ${SEEDS[*]}"
echo "输出目录: $BENCH_DIR/"
echo "估计耗时: RTX 4060 ~1-1.5 小时"
echo

# 跑每个 seed
for SEED in "${SEEDS[@]}"; do
    SEED_DIR="$BENCH_DIR/${MODEL}_seed${SEED}"
    echo "=========================================="
    echo "  Seed $SEED → $SEED_DIR/"
    echo "=========================================="

    # 已完成则跳过
    if [ -f "$SEED_DIR/ood_metrics.json" ]; then
        echo "✅ seed=$SEED 已完成, 跳过"
        continue
    fi

    # 训练
    echo "--- 训练 seed=$SEED ---"
    python train.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        --seed "$SEED" \
        --max_steps "$MAX_STEPS" \
        --out_dir "$SEED_DIR"

    # 检查 best.pt
    if [ ! -f "$SEED_DIR/best.pt" ]; then
        echo "❌ seed=$SEED 训练未生成 best.pt"
        continue
    fi

    # OOD 评估
    echo "--- OOD 评估 seed=$SEED ---"
    python eval_ood.py \
        --checkpoint "$SEED_DIR/best.pt" \
        --config "$CONFIG" \
        --data_root ./data/ood_T150 \
        --out "$SEED_DIR/ood_metrics.json"

    echo "✅ seed=$SEED 完成"
    echo
done

# 汇总 5 个 seed 的指标
echo "=========================================="
echo "  5-seed 基准汇总"
echo "=========================================="
python <<EOF
import json
import statistics
from pathlib import Path

bench_dir = Path("$BENCH_DIR")
seeds = [42, 123, 456, 789, 1024]
results = []

for seed in seeds:
    seed_dir = bench_dir / f"${MODEL}_seed{seed}"
    metrics_file = seed_dir / "ood_metrics.json"
    if not metrics_file.exists():
        print(f"  seed={seed}: ❌ ood_metrics.json 不存在")
        continue
    with open(metrics_file) as f:
        m = json.load(f)
    ch = m.get("changed_acc", 0)
    dec = m.get("ood_decay_pct")
    results.append({"seed": seed, "ch_acc": ch, "decay": dec})
    print(f"  seed={seed}: changed_acc={ch:.4f}  decay={dec*100:+.2f}%" if dec is not None else f"  seed={seed}: changed_acc={ch:.4f}")

if len(results) >= 2:
    chs = [r["ch_acc"] for r in results]
    mean_ch = statistics.mean(chs)
    std_ch = statistics.stdev(chs)
    print()
    print(f"  changed_acc 均值: {mean_ch:.4f} ± {std_ch:.4f}")
    print(f"  README 期望:     0.5952 ± 0.0034")
    if abs(mean_ch - 0.5952) < 0.05:
        print()
        print("  ✅ 5-seed 基准与论文一致 (±0.05 内)")
    else:
        print()
        print("  ⚠️  5-seed 基准偏离论文期望 (可能 seed/步数/环境差异)")

# 保存汇总
summary_file = bench_dir / "summary_5seeds.json"
summary = {
    "model": "$MODEL",
    "seeds": seeds,
    "results": results,
    "changed_acc_mean": mean_ch if len(results) >= 2 else None,
    "changed_acc_std": std_ch if len(results) >= 2 else None,
    "expected": {"changed_acc_mean": 0.5952, "changed_acc_std": 0.0034},
}
summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\n  汇总已保存: {summary_file}")
EOF

echo
echo "✅ 5-seed 基准完成"
echo "   结果: $BENCH_DIR/"
echo "   汇总: $BENCH_DIR/summary_5seeds.json"
