#!/bin/bash
killall -9 python3 2>/dev/null
sleep 1
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cleared')"
echo "done"
