"""C500部署工具: 用plink SSH上传文件+执行命令

用法:
    python c500_plink.py upload     # 上传所有文件
    python c500_plink.py check      # 检查环境
    python c500_plink.py launch     # 启动实验
    python c500_plink.py progress   # 检查进度
    python c500_plink.py download   # 下载结果

连接配置说明:
    所有 SSH 连接参数通过环境变量提供, 不再硬编码在源码中,
    确保本文件可安全推送到公开仓库。可通过以下两种方式配置:

    方式1: 使用 .env 文件 (推荐, 需要 python-dotenv, 已在 requirements.txt 中)
        在项目根目录创建 .env 文件 (.env 不会被提交到 git), 填入:
            C500_SSH_HOST=your.host.ip
            C500_SSH_PORT=32222
            C500_SSH_USER=your_username
            C500_SSH_PASS=your_password
            C500_SSH_HOSTKEY=SHA256:xxxxx

    方式2: 直接设置环境变量
        Linux/macOS:
            export C500_SSH_HOST=your.host.ip
            export C500_SSH_PORT=32222
            export C500_SSH_USER=your_username
            export C500_SSH_PASS=your_password
            export C500_SSH_HOSTKEY=SHA256:xxxxx
        Windows PowerShell:
            $env:C500_SSH_HOST="your.host.ip"
            $env:C500_SSH_PORT="32222"
            $env:C500_SSH_USER="your_username"
            $env:C500_SSH_PASS="your_password"
            $env:C500_SSH_HOSTKEY="SHA256:xxxxx"
"""
import base64
import subprocess
import os
import sys
import time

# 尝试加载 .env 文件 (可选; 未安装 python-dotenv 时回退到系统环境变量)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ========== 环境变量校验 ==========
_REQUIRED_ENV = ["C500_SSH_HOST", "C500_SSH_USER", "C500_SSH_PASS", "C500_SSH_HOSTKEY"]
_MISSING_ENV = [k for k in _REQUIRED_ENV if not os.environ.get(k)]

if _MISSING_ENV:
    print("[ERROR] 缺少必要的 SSH 连接环境变量, 请先配置以下变量:")
    for _k in _MISSING_ENV:
        print(f"  - {_k}")
    print("\n可选环境变量:")
    print("  - C500_SSH_PORT  (默认 32222)")
    print("\n配置方式:")
    print("  1. 在项目根目录创建 .env 文件 (推荐, .env 不会被提交到 git):")
    print("       C500_SSH_HOST=your.host.ip")
    print("       C500_SSH_PORT=32222")
    print("       C500_SSH_USER=your_username")
    print("       C500_SSH_PASS=your_password")
    print("       C500_SSH_HOSTKEY=SHA256:xxxxx")
    print("  2. 或直接设置系统环境变量 (Linux/macOS: export; Windows: set / $env:)")
    print("\n提示: python-dotenv 可选; 若未安装, 请直接设置系统环境变量。")
    sys.exit(1)

# ========== C500 plink连接参数 (全部来自环境变量) ==========
PLINK_ARGS = [
    'plink', '-batch', '-ssh',
    '-hostkey', os.environ['C500_SSH_HOSTKEY'],
    '-P', os.environ.get('C500_SSH_PORT', '32222'),
    '-l', os.environ['C500_SSH_USER'],
    '-pw', os.environ['C500_SSH_PASS'],
    os.environ['C500_SSH_HOST'],
]
REMOTE_DIR = "/root/three_chain_v3"

LOCAL_BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FILES = [
    "battle32_c500_gpu.py",
    "stats_analysis.py",
    "models/three_chain_mamba3.py",
    "models/__init__.py",
    "models/baselines.py",
    "models/common.py",
    "models/three_chain_mamba2.py",
    "models/three_chain_mamba2_bp.py",
    "models/three_chain_mamba2_bpv2.py",
    "models/three_chain.py",
    "models/mamba3_ref.py",
    "data/gen_grid_world.py",
    "data/dataset.py",
    "data/__init__.py",
    "c500_smoke.py",
    "c500_run_battle32_v2.sh",
]


def run_remote(cmd, timeout=300, stdin_data=None):
    """通过plink执行远程命令"""
    full_cmd = f"source /opt/maca/env.sh 2>/dev/null; cd {REMOTE_DIR}; {cmd}"
    args = PLINK_ARGS + [full_cmd]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate(input=stdin_data, timeout=timeout)
    return out.decode('utf-8', errors='replace'), err.decode('utf-8', errors='replace'), proc.returncode


def upload_file(local_path, remote_path):
    """通过base64编码+stdin管道上传文件"""
    with open(local_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    # 确保远程目录存在
    remote_dir = os.path.dirname(remote_path)
    mkdir_cmd = f"mkdir -p {remote_dir}"
    run_remote(mkdir_cmd, timeout=10)

    # 通过stdin传递base64编码的文件内容
    cmd = f"python3 -c \"import base64,sys; open('{remote_path}','wb').write(base64.b64decode(sys.stdin.read()))\""
    out, err, rc = run_remote(cmd, timeout=30, stdin_data=b64.encode())
    if rc == 0:
        print(f"  OK: {os.path.basename(local_path)} ({len(data)} bytes)")
    else:
        print(f"  FAIL: {os.path.basename(local_path)}: {err[:200]}")
    return rc == 0


def upload_all():
    """上传所有文件"""
    print(f"\n=== Uploading {len(UPLOAD_FILES)} files to C500 ===")
    ok = 0
    for fname in UPLOAD_FILES:
        local = os.path.join(LOCAL_BASE, fname.replace('/', os.sep))
        remote = f"{REMOTE_DIR}/{fname}"
        if not os.path.exists(local):
            print(f"  [SKIP] {fname} not found")
            continue
        if upload_file(local, remote):
            ok += 1
    print(f"\nUploaded {ok}/{len(UPLOAD_FILES)} files")


def check_env():
    """检查C500环境"""
    print("\n=== Checking C500 Environment ===")
    out, err, rc = run_remote("python3 c500_smoke.py", timeout=120)
    print(out)
    if err.strip():
        print(f"STDERR: {err[:1000]}")
    ok = "Mamba3 OK" in out and "PairwiseBasePairMamba3 OK" in out
    print(f"\nEnvironment check: {'PASS' if ok else 'FAIL'}")
    return ok


def launch():
    """启动32场景实验(断点续跑,不清除已有结果)"""
    print("\n=== Launching 32-Scenario Experiment (v2 hardened, resume) ===")
    # 不删除已有结果,支持断点续跑
    out, err, rc = run_remote(
        "chmod +x c500_run_battle32_v2.sh && nohup bash c500_run_battle32_v2.sh > /dev/null 2>&1 & echo PID=$!",
        timeout=10
    )
    print(out.strip())
    if err.strip():
        print(f"STDERR: {err[:200]}")


def progress():
    """检查实验进度"""
    out, err, rc = run_remote("tail -30 battle32_c500_log.txt 2>/dev/null", timeout=10)
    print(out)
    # 统计已完成数
    out2, _, _ = run_remote("grep -c 'val_ch=' battle32_c500_log.txt 2>/dev/null || echo 0", timeout=10)
    count = int(out2.strip()) if out2.strip().isdigit() else 0
    print(f"\nCompleted: {count}/384")
    return count


def download():
    """下载结果文件"""
    print("\n=== Downloading Results ===")
    for remote_name, local_suffix in [
        ("battle32_results.json", "battle32_c500_results.json"),
        ("battle32_stats.json", "battle32_c500_stats.json"),
        ("battle32_c500_log.txt", "battle32_c500_log.txt"),
    ]:
        out, err, rc = run_remote(f"cat {REMOTE_DIR}/{remote_name}", timeout=30)
        if rc == 0 and out.strip():
            local_path = os.path.join(LOCAL_BASE, local_suffix)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(out)
            print(f"  Downloaded: {remote_name} -> {local_suffix} ({len(out)} bytes)")
        else:
            print(f"  [SKIP] {remote_name}: not ready or empty")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action == "all":
        upload_all()
        ok = check_env()
        if ok:
            launch()
        else:
            print("\n[WARNING] Environment check failed, not launching yet.")
    elif action == "upload":
        upload_all()
    elif action == "check":
        check_env()
    elif action == "launch":
        launch()
    elif action == "progress":
        progress()
    elif action == "download":
        download()
    else:
        print(f"Usage: python {sys.argv[0]} [all|upload|check|launch|progress|download]")
