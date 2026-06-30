#!/bin/bash
# 等待 matplotlib 安装完成，然后运行专业测评报告生成
cd /mnt/e/three_chain_v3

echo "=== 等待 matplotlib 可用 ==="
for i in $(seq 1 60); do
    if python3 -c "import matplotlib" 2>/dev/null; then
        echo "✅ matplotlib 可用 (等待 $((i*5))s)"
        break
    fi
    echo "  [$i/60] matplotlib 尚未就绪，等待 5s..."
    sleep 5
done

if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "❌ 等待 300s 后 matplotlib 仍不可用，请检查安装"
    exit 1
fi

echo ""
echo "=== 运行专业测评报告生成 ==="
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
python3 generate_pro_report.py 2>&1
echo ""
echo "=== 完成，检查输出 ==="
ls -la figures/ 2>&1
ls -la PROFESSIONAL_REPORT.md 2>&1
