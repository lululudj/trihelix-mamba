#!/bin/bash
source "$HOME/mamba3_venv/bin/activate"

echo "=== venv mamba_ssm 当前状态 ==="
python3 -c "import mamba_ssm; print('version:', mamba_ssm.__version__); print('path:', mamba_ssm.__file__)"
echo "=== try Mamba3 ==="
python3 -c "from mamba_ssm import Mamba3; print('✅ Mamba3 import OK')" 2>&1 | head -5
echo "=== check files in venv mamba_ssm ==="
ls "$HOME/mamba3_venv/lib/python3.14/site-packages/mamba_ssm/modules/" 2>/dev/null | grep mamba
echo "=== mamba3.py 在 venv 里吗 ==="
ls -la "$HOME/mamba3_venv/lib/python3.14/site-packages/mamba_ssm/modules/mamba3.py" 2>/dev/null || echo "NO mamba3.py in venv"
