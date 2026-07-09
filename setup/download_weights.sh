#!/bin/bash
# setup/download_weights.sh — 下载预训练权重 best.pt
#
# 用法 (在 WSL Ubuntu 里):
#   cd /mnt/e/three_chain_v3
#   bash setup/download_weights.sh
#
# 下载源:
#   主: GitHub Release (https://github.com/lululudj/trihelix-mamba/releases)
#   备: GitLink Release (https://www.gitlink.org.cn/lulululudj/ThreeChainMamba/releases)
#
# best.pt = 3.26M GridWorld 主力模型, OOD changed_acc@T=150 = 0.5952 ± 0.0034, decay ≈ 0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== 下载预训练权重 best.pt ==="

# 已存在则跳过
if [ -f best.pt ]; then
    SIZE=$(stat -c%s best.pt 2>/dev/null || stat -f%z best.pt 2>/dev/null)
    if [ "$SIZE" -gt 30000000 ]; then
        echo "✅ best.pt 已存在 ($SIZE bytes), 跳过下载"
        exit 0
    else
        echo "⚠️  best.pt 文件过小 ($SIZE bytes), 可能损坏, 重新下载..."
        rm -f best.pt
    fi
fi

# 主链接: GitHub Release
GITHUB_URL="https://github.com/lululudj/trihelix-mamba/releases/download/v1.0-weights/best.pt"
GITLINK_URL="https://www.gitlink.org.cn/lulululudj/ThreeChainMamba/releases"

echo "下载 best.pt (39 MB, 3.26M GridWorld 主力模型)..."
echo "  尝试 GitHub Release..."
if curl -fL --connect-timeout 15 --max-time 300 -o best.pt "$GITHUB_URL"; then
    echo "  ✅ GitHub Release 下载成功"
else
    echo "  ❌ GitHub Release 下载失败"
    echo "  尝试 GitLink Release (国内镜像)..."
    # GitLink 的具体 release 下载 URL 需要用户从网页获取
    echo "  请手动访问: $GITLINK_URL"
    echo "  下载 best.pt 后放到: $PROJECT_ROOT/best.pt"
    exit 1
fi

# 校验大小 (期望 ~39MB)
SIZE=$(stat -c%s best.pt 2>/dev/null || stat -f%z best.pt 2>/dev/null)
if [ -z "$SIZE" ] || [ "$SIZE" -lt 30000000 ]; then
    echo "❌ best.pt 下载失败或过小 ($SIZE bytes)"
    echo "   期望 ~39 MB, 实际 ${SIZE:-0} bytes"
    echo "   请手动下载:"
    echo "     GitHub:  $GITHUB_URL"
    echo "     GitLink: $GITLINK_URL"
    rm -f best.pt
    exit 1
fi

echo "✅ best.pt 下载完成 ($SIZE bytes, $((SIZE/1024/1024)) MB)"
echo
echo "下一步:"
echo "  bash setup/verify_ood.sh   # 用 best.pt 跑 OOD 长程评估验证"
