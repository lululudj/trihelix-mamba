#!/bin/bash
set -e
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"

echo "=== Python 依赖检查 ==="
python3 -c "
import sys
print('Python:', sys.version)
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('cuda version:', torch.version.cuda)
try:
    import mamba_ssm
    print('mamba_ssm:', mamba_ssm.__version__)
except Exception as e:
    print('mamba_ssm error:', e)
try:
    from mamba_ssm import Mamba
    print('Mamba module import OK')
except Exception as e:
    print('Mamba import error:', e)
"

echo ""
echo "=== 数据检查 ==="
ls data_split/ | head -10
echo "..."
echo "data_split 文件夹数:"
ls data_split/ | wc -l

echo ""
echo "=== 现有 results ==="
ls results/ | head -20
