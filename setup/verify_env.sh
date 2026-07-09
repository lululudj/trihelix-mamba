#!/bin/bash
# setup/verify_env.sh — 环境自测脚本
#
# 用法 (在 WSL Ubuntu 里, conda env mamba 已激活):
#   cd /mnt/e/three_chain_v3
#   bash setup/verify_env.sh
#
# 断言:
#   1. torch.cuda.is_available() + GPU 名字
#   2. mamba_ssm.Mamba2 能 import + 实例化
#   3. causal_conv1d 能 import
#   4. models.three_chain_mamba2.ThreeChainMamba2 能 import

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 自动激活 conda env mamba (如果没激活)
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "mamba" ]; then
    echo "激活 conda env mamba..."
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mamba
fi

echo "=== ThreeChainMamba2 环境自测 ==="
echo "项目根: $PROJECT_ROOT"
echo "Python: $(which python)"
echo

python <<'EOF'
import sys
import warnings
warnings.filterwarnings("ignore")

THREE_CHAIN_ROOT = "/mnt/e/three_chain_v3"
if THREE_CHAIN_ROOT not in sys.path:
    sys.path.insert(0, THREE_CHAIN_ROOT)

errors = []

# 1. torch + CUDA
try:
    import torch
    assert torch.cuda.is_available(), "CUDA 不可用"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ torch={torch.__version__} cuda={torch.version.cuda} gpu={gpu_name}")
except Exception as e:
    errors.append(f"torch: {e}")
    print(f"❌ torch: {e}")

# 2. triton
try:
    import triton
    print(f"✅ triton={triton.__version__}")
except Exception as e:
    errors.append(f"triton: {e}")
    print(f"❌ triton: {e}")

# 3. mamba_ssm
try:
    import mamba_ssm
    from mamba_ssm import Mamba2
    m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2, headdim=64)
    n = sum(p.numel() for p in m.parameters())
    print(f"✅ mamba_ssm={mamba_ssm.__version__} (Mamba2 实例化 OK, params={n:,})")
except Exception as e:
    errors.append(f"mamba_ssm: {e}")
    print(f"❌ mamba_ssm: {e}")

# 4. causal_conv1d
try:
    import causal_conv1d
    print(f"✅ causal_conv1d")
except Exception as e:
    errors.append(f"causal_conv1d: {e}")
    print(f"❌ causal_conv1d: {e}")

# 5. 项目核心 import
try:
    from models.three_chain_mamba2 import ThreeChainMamba2
    model = ThreeChainMamba2(
        cell_types=16, action_dim=5,
        d_model=256, n_layers=2, max_T=256,
        enable_m3=False, enable_jepa=False,
        use_hta=False, use_checkpoint=False,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"✅ ThreeChainMamba2 import OK (params={n/1e6:.2f}M)")
except Exception as e:
    errors.append(f"ThreeChainMamba2: {e}")
    print(f"❌ ThreeChainMamba2: {e}")

# 6. 基础依赖
for mod_name in ["numpy", "yaml", "matplotlib", "dotenv"]:
    try:
        __import__(mod_name)
        print(f"✅ {mod_name}")
    except Exception as e:
        errors.append(f"{mod_name}: {e}")
        print(f"❌ {mod_name}: {e}")

print()
if errors:
    print(f"❌ 自测失败 ({len(errors)} 项):")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("✅ 环境自测全部通过")
    sys.exit(0)
EOF
