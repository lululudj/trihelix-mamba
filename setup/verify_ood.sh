#!/bin/bash
# setup/verify_ood.sh — 秒级 OOD 长程外推评估验证 (核心交付物)
#
# 用法 (在 WSL Ubuntu 里, conda env mamba 已激活):
#   cd /mnt/e/three_chain_v3
#   bash setup/verify_ood.sh
#
# 流程:
#   1. 确认 best.pt 已下载 (否则调 download_weights.sh)
#   2. 用 best.pt 跑 eval_ood.py 评估 data/ood_T150/ (72 样本, T=150)
#   3. 解析 ood_metrics.json, 断言:
#        changed_acc > 0.50  (优于随机 1/16=0.0625)
#        |ood_decay_pct| < 0.05  (t=100→150 衰减 ±5% 内 = 不退化)
#
# 期望输出:
#   ✅ 本机部署验证通过: changed_acc≈0.59, decay≈0%
#
# 期望指标 (来自 README.md):
#   ThreeChainMamba2 OOD changed_acc@T=150 = 0.5952 ± 0.0034
#   OOD decay ≈ -0.0042 (≈ 0)

set -eo pipefail   # pipefail: 让 python | tee 管道能捕获 python 失败

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 自动激活 conda env mamba
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "mamba" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mamba
fi

echo "=== OOD 长程外推评估验证 ==="
echo "项目根: $PROJECT_ROOT"
echo

# 1. 确认 best.pt
if [ ! -f best.pt ]; then
    echo "best.pt 不存在, 调用 download_weights.sh..."
    bash setup/download_weights.sh
fi
[ -f best.pt ] || { echo "❌ best.pt 仍未就绪, 无法验证"; exit 1; }

# 2. 确认 OOD 测试数据
DATA_ROOT="$PROJECT_ROOT/data/ood_T150"
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ OOD 测试数据目录不存在: $DATA_ROOT"
    echo "   data/ood_T150/ 应已含在仓库里 (72 样本 T=150)"
    echo "   如果没有, 请重新 clone 仓库或联系 owner"
    exit 1
fi
N_NPZ=$(find "$DATA_ROOT" -name "*.npz" | wc -l)
echo "OOD 数据: $DATA_ROOT ($N_NPZ 个 .npz 文件)"
[ "$N_NPZ" -gt 0 ] || { echo "❌ OOD 数据目录为空"; exit 1; }

# 3. 跑 eval_ood.py
echo
echo "跑 eval_ood.py (用 best.pt 评估 T=150 OOD 数据)..."
METRICS_JSON="/tmp/ood_metrics.json"
python eval_ood.py \
    --checkpoint best.pt \
    --config configs/matched_mamba2.yaml \
    --data_root ./data/ood_T150 \
    --out "$METRICS_JSON" 2>&1 | tee /tmp/ood_stdout.txt

# 4. 解析 JSON 断言
echo
echo "=== 验证指标 ==="
python <<EOF
import json
import sys

with open("$METRICS_JSON") as f:
    m = json.load(f)

ch_acc = m.get("changed_acc", 0)
decay_pct = m.get("ood_decay_pct")
n_samples = m.get("n_samples", 0)
max_T = m.get("max_T", 0)
model_name = m.get("model", "unknown")

print(f"模型: {model_name}")
print(f"样本数: {n_samples}, max_T: {max_T}")
print(f"changed_acc (t=150): {ch_acc:.4f}")
if decay_pct is not None:
    print(f"ood_decay_pct (t=100→150): {decay_pct*100:+.2f}%")
else:
    print("ood_decay_pct: N/A (数据缺 t=100 或 t=150)")

errors = []
if ch_acc <= 0.50:
    errors.append(f"changed_acc={ch_acc:.4f} <= 0.50 (期望 ≈ 0.59)")
if decay_pct is not None and abs(decay_pct) >= 0.05:
    errors.append(f"|ood_decay_pct|={abs(decay_pct)*100:.2f}% >= 5% (期望 ≈ 0%)")
# 注意: eval_ood.py 的 n_samples 实际是 batch 数 (len(final_accs)),
# 不是真实样本数. 72 样本 / batch_size=32 = 3 batch, 所以 n_samples=3 是正常的.
# 真实样本数从 stdout "OOD 样本数: 72" 拿, 这里只断言至少有 1 个 batch.
if n_samples < 1:
    errors.append(f"n_batches={n_samples} < 1 (无任何 batch, 数据可能损坏)")

if errors:
    print()
    print("❌ 验证失败:")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print()
    print("=" * 50)
    print("✅ 本机部署验证通过!")
    print(f"   changed_acc = {ch_acc:.4f} (期望 ≈ 0.5952)")
    if decay_pct is not None:
        print(f"   ood_decay   = {decay_pct*100:+.2f}% (期望 ≈ 0%)")
    print(f"   样本数 = {n_samples}")
    print("=" * 50)
    sys.exit(0)
EOF
