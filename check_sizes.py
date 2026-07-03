"""快速检查 three_chain_v3 各子目录大小"""
import os

root = r'e:\three_chain_v3'
print("一级目录大小:")
for d in sorted(os.listdir(root)):
    full = os.path.join(root, d)
    if os.path.isdir(full):
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(full) for f in fs)
        nfiles = sum(len(fs) for _, _, fs in os.walk(full))
        print(f"  {d:30s} {size/1024/1024:8.1f} MB  ({nfiles} files)")
    else:
        size = os.path.getsize(full)
        if size > 1024*1024:
            print(f"  {d:30s} {size/1024/1024:8.1f} MB  (file)")

print("\n二级目录 (data/ 下):")
data_dir = os.path.join(root, 'data')
if os.path.isdir(data_dir):
    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if os.path.isdir(full):
            size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(full) for f in fs)
            print(f"  data/{d:27s} {size/1024/1024:8.1f} MB")
