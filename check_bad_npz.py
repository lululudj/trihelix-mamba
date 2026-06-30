"""扫描并删除损坏的 .npz 文件。"""
import sys
from pathlib import Path
import numpy as np

roots = [Path('data_cache'), Path('data_split')]
bad = []
ok = 0
for root in roots:
    if not root.exists():
        continue
    for f in root.rglob('*.npz'):
        try:
            with np.load(f, allow_pickle=True) as d:
                _ = d['S_0'], d['actions'], d['S_t']
            ok += 1
        except Exception as e:
            bad.append((str(f), str(e)))

# 删除损坏文件
for f, _ in bad:
    Path(f).unlink()

# 输出到文件
with open('check_result.txt', 'w', encoding='utf-8') as fp:
    fp.write(f'OK: {ok}\n')
    fp.write(f'BAD: {len(bad)}\n')
    for f, e in bad[:20]:
        fp.write(f'  {f}: {e}\n')
    fp.write(f'已删除 {len(bad)} 个损坏文件\n')
