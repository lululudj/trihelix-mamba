"""SDD annotations 下载器 v2 (GitHub flclain/StanfordDroneDataset)

vatic2.stanford.edu 服务器全球不可达, 改从 GitHub 镜像仓库下载。

源仓库: https://github.com/flclain/StanfordDroneDataset (纯 annotation 文件, ~60MB)
下载策略:
  1. 用 GitHub API 列出 annotations/{scene}/ 结构
  2. 用 raw.githubusercontent.com 逐个下载 annotations.txt (每个几百KB, 共~10MB)
  3. 若 raw 失败, 回退到 codeload 完整 zip (60MB, 较慢)

输出结构 (与 gen_sdd_grid.py 的 --sdd_root 契约一致):
  {out_dir}/
  └── annotations/
      └── {scene}/
          ├── video0/annotations.txt
          └── ...
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = "flclain/StanfordDroneDataset"
API_BASE = "https://api.github.com/repos"
RAW_BASE = "https://raw.githubusercontent.com"
CODELOAD_URL = "https://codeload.github.com/flclain/StanfordDroneDataset/zip/refs/heads/master"

ALL_SCENES = [
    "bookstore", "coupa", "deathCircle", "gates",
    "hyang", "little", "nexus", "quad",
]


def github_api(path, retries=3):
    """调用 GitHub API, 带 User-Agent 和重试。"""
    url = f"{API_BASE}/{REPO}/{path}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "sdd-downloader"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"    [api retry {attempt+1}/{retries}] {type(e).__name__}: {e}")
            time.sleep(2)
    return None


def list_dir(path=""):
    """列出仓库某路径下的内容。返回 [(name, type, download_url), ...]"""
    api_path = f"contents/{path}" if path else "contents"
    items = github_api(api_path)
    if items is None:
        return []
    result = []
    for it in items:
        result.append((
            it.get("name", ""),
            it.get("type", ""),
            it.get("download_url"),
        ))
    return result


def download_raw(url, dest_path, retries=3):
    """从 raw.githubusercontent.com 下载单个文件。"""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sdd-downloader"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(data)
            return len(data)
        except Exception as e:
            print(f"      [retry {attempt+1}/{retries}] {type(e).__name__}: {e}")
            time.sleep(2)
    return 0


def download_via_codeload(out_dir):
    """回退: 下载完整 zip 并解压。"""
    import subprocess
    import zipfile
    zip_path = os.path.join(out_dir, "sdd_repo.zip")
    print(f"  [codeload] 下载 {CODELOAD_URL} (可能需要几分钟)...")
    r = subprocess.run(
        ["wget", "--tries=3", "--timeout=300", "-q", CODELOAD_URL, "-O", zip_path],
        timeout=600,
    )
    if r.returncode != 0 or not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
        print(f"  [codeload] 失败")
        return False
    print(f"  [codeload] zip 大小: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
    print(f"  [codeload] 解压...")
    extract_dir = os.path.join(out_dir, "_extracted")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        print(f"  [codeload] 解压失败: {e}")
        return False
    # 找到 annotations 目录并移动
    for root, dirs, files in os.walk(extract_dir):
        if os.path.basename(root) == "annotations" and any(
            d in ALL_SCENES for d in dirs
        ):
            # 移动到 out_dir/annotations
            dest = os.path.join(out_dir, "annotations")
            if os.path.exists(dest):
                import shutil
                shutil.rmtree(dest)
            import shutil
            shutil.move(root, dest)
            print(f"  [codeload] annotations 移动到 {dest}")
            os.remove(zip_path)
            return True
    print(f"  [codeload] 未找到 annotations 目录")
    return False


def download_scene(scene, out_dir):
    """下载单个场景的 annotations。"""
    print(f"\n--- {scene} ---")
    # 先列出 annotations/ 下是否有该场景
    ann_items = list_dir("annotations")
    if not ann_items:
        print(f"  [warn] 仓库根目录无 annotations/, 尝试顶层...")
        # 看顶层结构
        top = list_dir("")
        print(f"  顶层: {top[:10]}")
        return 0

    # 找场景目录
    scene_entry = None
    for name, typ, url in ann_items:
        if name == scene and typ == "dir":
            scene_entry = (name, typ, url)
            break
    if not scene_entry:
        print(f"  [warn] annotations/ 下无 {scene}/, 可用: {[n for n,t,_ in ann_items if t=='dir'][:10]}")
        return 0

    # 列出 scene 下的 video 目录
    scene_items = list_dir(f"annotations/{scene}")
    if not scene_items:
        print(f"  [warn] annotations/{scene}/ 为空或不可访问")
        return 0

    video_dirs = [(n, u) for n, t, u in scene_items if t == "dir"]
    print(f"  {scene}: {len(video_dirs)} 个 video 目录")
    if not video_dirs:
        # 可能直接是文件
        files = [(n, u) for n, t, u in scene_items if t == "file" and n == "annotations.txt"]
        if files:
            name, url = files[0]
            dest = os.path.join(out_dir, "annotations", scene, "video0", "annotations.txt")
            sz = download_raw(url, dest)
            print(f"  ✓ {name}: {sz} bytes")
            return 1 if sz > 0 else 0
        return 0

    # 逐个 video 下载 annotations.txt
    count = 0
    for vname, _ in video_dirs:
        # 列 video 目录找 annotations.txt
        vitems = list_dir(f"annotations/{scene}/{vname}")
        for fname, ftype, furl in vitems:
            if fname == "annotations.txt" and ftype == "file" and furl:
                dest = os.path.join(out_dir, "annotations", scene, vname, "annotations.txt")
                sz = download_raw(furl, dest)
                if sz > 0:
                    count += 1
                    print(f"  ✓ {scene}/{vname}/annotations.txt ({sz} bytes)")
                else:
                    print(f"  ✗ {scene}/{vname}/annotations.txt 下载失败")
                break
    return count


def main():
    parser = argparse.ArgumentParser(description="SDD annotations 下载器 (GitHub 镜像)")
    parser.add_argument("--out_dir", default="./SDD", help="输出根目录 (默认 ./SDD)")
    parser.add_argument(
        "--scenes", nargs="+", default=["bookstore"],
        help="场景名列表 (默认 bookstore); 'all' = 全 8 场景",
    )
    parser.add_argument("--use_codeload", action="store_true",
                        help="强制用 codeload zip 下载 (跳过 raw 逐个下载)")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenes = ALL_SCENES if args.scenes == ["all"] else args.scenes
    print(f"=== 下载 SDD annotations: {scenes} → {args.out_dir} ===")
    print(f"    源仓库: github.com/{REPO}")

    if args.use_codeload:
        ok = download_via_codeload(args.out_dir)
        if not ok:
            print("[fatal] codeload 下载失败")
            sys.exit(1)
        # 统计
        ann_dir = out_path / "annotations"
        total = sum(1 for _ in ann_dir.rglob("annotations.txt"))
        print(f"\n=== codeload 完成: {total} 个 annotations.txt ===")
    else:
        total = 0
        for scene in scenes:
            n = download_scene(scene, args.out_dir)
            total += n
        print(f"\n=== 完成: 共 {total} 个 annotations.txt ===")
        # 若 raw 方式失败, 回退到 codeload
        if total == 0:
            print("\n[warn] raw 下载失败, 回退到 codeload zip...")
            ok = download_via_codeload(args.out_dir)
            if ok:
                ann_dir = out_path / "annotations"
                total = sum(1 for _ in ann_dir.rglob("annotations.txt"))
                print(f"=== codeload 回退成功: {total} 个 annotations.txt ===")

    # 列出最终结构
    ann_dir = out_path / "annotations"
    if ann_dir.exists():
        print("\n目录结构:")
        for scene_dir in sorted(ann_dir.iterdir()):
            if scene_dir.is_dir():
                ann_count = sum(
                    1 for v in scene_dir.iterdir()
                    if v.is_dir() and (v / "annotations.txt").exists()
                )
                print(f"  {scene_dir.name}/: {ann_count} videos")


if __name__ == "__main__":
    main()
