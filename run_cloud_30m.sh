#!/bin/bash
# ============================================================
# 三链 DNA-Mamba2 30M 云端一键训练 + OOD eval
# 用法: 在 AutoDL JupyterLab Terminal 跑
#       bash run_cloud_30m.sh
# 预计: 训练 5-15 分钟 + eval 2-3 分钟 ≈ 20 分钟
# 成本: 2.29 元/h × 0.3h ≈ 0.7 元
# ============================================================
set -e

echo "=========================================="
echo "  三链 DNA-Mamba2 30M 云端验证"
echo "  (验证 SSM 长程外推优势是否随规模保持)"
echo "=========================================="
echo ""

# ---------- 1. 检查 GPU ----------
echo "[1/6] 检查 GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---------- 2. 定位代码目录 ----------
echo "[2/6] 定位代码目录..."
WORKDIR=""
for d in /root/three_chain_v3 /mnt/workspace/three_chain_v3 /root/autodl-tmp/three_chain_v3 /root/three_chain_v3; do
    if [ -f "$d/train.py" ]; then
        WORKDIR="$d"
        break
    fi
done
if [ -z "$WORKDIR" ]; then
    echo "[ERROR] 找不到 train.py!"
    echo ""
    echo "请确认 three_chain_v3 文件夹已上传到 /root/ 下"
    echo "当前目录: $(pwd)"
    echo "目录内容:"
    ls -la /root/ 2>/dev/null | head -20
    exit 1
fi
cd "$WORKDIR"
echo "代码目录: $WORKDIR"
echo ""

# ---------- 3. 装 mamba_ssm ----------
echo "[3/6] 检查 mamba_ssm..."
if python -c "from mamba_ssm import Mamba2" 2>/dev/null; then
    echo "mamba_ssm 已装好"
else
    echo "mamba_ssm 未装, 开始安装 (约 3-5 分钟编译)..."
    pip install packaging ninja -q
    # --no-build-isolation: 让 pip 用当前环境的 torch (隔离构建找不到 torch 会报错)
    pip install --no-build-isolation mamba-ssm causal-conv1d
    python -c "from mamba_ssm import Mamba2; print('mamba_ssm 装好了')"
fi
echo ""

# ---------- 4. 检查数据 ----------
echo "[4/6] 检查数据..."
if [ ! -d "data_split_m3" ]; then
    echo "[ERROR] 找不到 data_split_m3/ 训练数据!"
    echo "请上传 data_split_m3 文件夹到 $WORKDIR/"
    exit 1
fi
if [ ! -d "data/ood_T150" ]; then
    echo "[ERROR] 找不到 data/ood_T150/ OOD 数据!"
    echo "请上传 data 文件夹到 $WORKDIR/"
    exit 1
fi
echo "训练数据: $(ls data_split_m3/*.npz 2>/dev/null | wc -l) 个样本"
echo "OOD 数据: $(ls data/ood_T150/*.npz 2>/dev/null | wc -l) 个样本"
echo ""

# ---------- 5. 训练 ----------
echo "[5/6] 开始训练 (100 步, ~30M 参数, batch=2)..."
echo "预计 5-15 分钟, 请勿关闭浏览器"
echo ""
python train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_30m.yaml \
    --max_steps 100 \
    --batch_size 2 \
    --data_root ./data_split_m3 \
    --out_dir results/run_three_chain_mamba2_30m_seed0
echo ""

# ---------- 6. OOD eval ----------
echo "[6/6] OOD 长程外推评估..."
python eval_ood.py \
    --checkpoint results/run_three_chain_mamba2_30m_seed0/final.pt \
    --config configs/matched_mamba2_30m.yaml \
    --data_root ./data/ood_T150 \
    --batch_size 4 \
    --out results/run_three_chain_mamba2_30m_seed0/ood_metrics.json
echo ""

# ---------- 结果判断 ----------
echo "=========================================="
echo "  结果判断 (对照 baseline decay +0.7%)"
echo "=========================================="
python3 -c "
import json
with open('results/run_three_chain_mamba2_30m_seed0/ood_metrics.json') as f:
    m = json.load(f)
decay = m['ood_decay_pct'] * 100
ch100 = m['changed_acc_curve']['100']
ch150 = m['changed_acc_curve']['150']
print(f'')
print(f'参数规模: ~29M (9× baseline 3.26M)')
print(f'changed_acc@100 (训练长度): {ch100:.4f}')
print(f'changed_acc@150 (OOD 长度): {ch150:.4f}')
print(f'OOD decay: {decay:+.1f}%')
print(f'')
print(f'对照:')
print(f'  baseline 3.26M:  decay +0.7% (零衰减 ✅)')
print(f'  Mamba3 3.33M:    decay -11.3% (严重退化 ❌)')
print(f'  本实验 30M:      decay {decay:+.1f}%')
print(f'')
if decay > -1.0:
    print(f'✅✅✅ 趋势保持! SSM 优势随规模稳定!')
    print(f'   → 可投 100M / 300M, 有望投顶会主会')
elif decay > -5.0:
    print(f'⚠️ 轻微退化, 但远好于 Mamba3 (-11.3%)')
    print(f'   → 可谨慎扩 100M, 或直接发 workshop')
else:
    print(f'❌ 规模化失效, 退化严重')
    print(f'   → 3.26M 直接发 workshop, 别烧更多钱')
print(f'')
print(f'结果文件: results/run_three_chain_mamba2_30m_seed0/')
print(f'  - final.pt          (模型权重, 下载备用)')
print(f'  - ood_metrics.json  (完整 OOD 曲线)')
print(f'  - summary.json      (训练摘要)')
"
echo ""
echo "=========================================="
echo "  完成! 跑完记得回 AutoDL 控制台'关机'省钱"
echo "=========================================="
