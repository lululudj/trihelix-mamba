#!/bin/bash
# 在 WSL 中检查 ThreeChainBP 的门控值（CPU 即可）
set -e
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"
python3 check_bp_gate.py
