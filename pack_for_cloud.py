"""打包 three_chain_v3 给云端用 (去掉 results/ __pycache__/ .git/ 省体积)"""
import os
import zipfile

src = r'e:\three_chain_v3'
dst = r'e:\three_chain_v3_cloud.zip'

exclude_dirs = {
    'results', 'results_wsl',  # 实验结果 (2.3GB + 264MB, 云端重新生成)
    '__pycache__', '.git', '.trae', '.pytest_cache', '.venv', 'venv',
    'data_cache', 'data_split',  # 旧数据 (用 data_split_m3 即可)
    'm3_snapshots',  # .m3 快照 (云端重新生成)
    '_texlive',  # LaTeX 环境 (92MB)
    '_mamba3_ref', 'paper', 'figures', 'docs',  # 论文/文档相关
    'killer_experiments',  # 旧实验
}
exclude_exts = {'.pyc', '.pyo', '.log'}

count = 0
total_size = 0

with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in exclude_exts:
                continue
            full = os.path.join(root, f)
            arc = os.path.relpath(full, src)
            size = os.path.getsize(full)
            zf.write(full, arc)
            count += 1
            total_size += size

zip_size = os.path.getsize(dst)
print(f"打包完成: {dst}")
print(f"文件数: {count}")
print(f"原始大小: {total_size / 1024 / 1024:.1f} MB")
print(f"压缩后:   {zip_size / 1024 / 1024:.1f} MB")
print(f"\n关键文件检查:")
for must in ['train.py', 'eval_ood.py', 'utils.py',
             'configs/matched_mamba2_30m.yaml',
             'configs/matched_mamba2.yaml',
             'run_cloud_30m.sh',
             'models/three_chain_mamba2.py',
             'models/__init__.py',
             'data/dataset.py']:
    p = os.path.join(src, must)
    exists = '✓' if os.path.exists(p) else '✗ 缺失!'
    print(f"  {exists} {must}")
