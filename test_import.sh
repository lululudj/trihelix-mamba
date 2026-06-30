#!/bin/bash
cd /mnt/e/three_chain_v3
timeout 15 python3 -c "import sys; sys.path.insert(0,'.'); from models.common import HeteroMamba; print('import ok')" 2>&1
echo "exit: $?"
