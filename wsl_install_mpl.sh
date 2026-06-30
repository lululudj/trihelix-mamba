#!/bin/bash
set -e
python3 -m pip install --break-system-packages --quiet matplotlib 2>&1 | tail -3
python3 -c "import matplotlib; print('matplotlib', matplotlib.__version__)"
