"""从多种渠道下载 mamba-ssm + causal-conv1d 源码
方案A: ghproxy.com 镜像 git clone
方案B: kkgithub.com 镜像 git clone
方案C: codeload.github.com 下 zip
方案D: 本地下好源码 SFTP 上传
"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
INSTALL_SH = "/root/install_mamba_v2.sh"

# 这个脚本尝试多个镜像下载, 谁先用谁
INSTALL_SCRIPT = """#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS=4

# 工具函数: 试多个 URL git clone
try_clone() {
    local name=$1
    local dst=$2
    shift 2
    for url in "$@"; do
        echo "  尝试: $url"
        if timeout 60 git clone --depth 1 "$url" "$dst" 2>&1; then
            echo "  ✓ 成功: $url"
            return 0
        fi
        echo "  ✗ 失败, 试下一个"
        rm -rf "$dst"
    done
    return 1
}

# 工具函数: 试多个 URL 下 zip
try_zip() {
    local name=$1
    local dst_zip=$2
    shift 2
    for url in "$@"; do
        echo "  尝试 wget: $url"
        if timeout 60 wget -q --tries=2 --timeout=30 "$url" -O "$dst_zip"; then
            if [ -s "$dst_zip" ]; then
                echo "  ✓ 下载成功: $url ($(du -h $dst_zip | cut -f1))"
                return 0
            fi
        fi
        echo "  ✗ 失败, 试下一个"
        rm -f "$dst_zip"
    done
    return 1
}

echo "=== 1. 装 causal-conv1d ==="
rm -rf /root/causal-conv1d-src /root/cc1d.zip

# 方案 A: ghproxy 镜像 git clone
echo "--- 方案 A: ghproxy git clone ---"
if try_clone "causal-conv1d" "/root/causal-conv1d-src" \\
    "https://gh-proxy.com/https://github.com/Dao-AILab/causal-conv1d.git" \\
    "https://ghproxy.net/https://github.com/Dao-AILab/causal-conv1d.git" \\
    "https://mirror.ghproxy.com/https://github.com/Dao-AILab/causal-conv1d.git"; then
    cd /root/causal-conv1d-src
elif try_clone "causal-conv1d" "/root/causal-conv1d-src" \\
    "https://kkgithub.com/Dao-AILab/causal-conv1d.git" \\
    "https://gitclone.com/github.com/Dao-AILab/causal-conv1d.git"; then
    cd /root/causal-conv1d-src
else
    # 方案 C: 下 zip
    echo "--- 方案 C: 下 zip ---"
    if try_zip "causal-conv1d" "/root/cc1d.zip" \\
        "https://codeload.github.com/Dao-AILab/causal-conv1d/zip/refs/heads/main" \\
        "https://gh-proxy.com/https://github.com/Dao-AILab/causal-conv1d/archive/refs/heads/main.zip"; then
        cd /root && unzip -q cc1d.zip && mv causal-conv1d-main causal-conv1d-src
        cd /root/causal-conv1d-src
    else
        echo "✗ causal-conv1d 所有镜像都失败"
        exit 1
    fi
fi

echo "=== HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo no-git) ==="
echo "=== 装 causal-conv1d ==="
pip install --no-build-isolation -e . 2>&1 | tail -20

echo ""
echo "=== 2. 装 mamba-ssm v2.2.4 ==="
rm -rf /root/mamba-src /root/mamba.zip

echo "--- 方案 A: ghproxy git clone ---"
if try_clone "mamba" "/root/mamba-src" \\
    "https://gh-proxy.com/https://github.com/state-spaces/mamba.git" \\
    "https://ghproxy.net/https://github.com/state-spaces/mamba.git" \\
    "https://mirror.ghproxy.com/https://github.com/state-spaces/mamba.git"; then
    cd /root/mamba-src && git checkout v2.2.4 2>/dev/null || echo "checkout v2.2.4 失败, 用 main"
elif try_clone "mamba" "/root/mamba-src" \\
    "https://kkgithub.com/state-spaces/mamba.git" \\
    "https://gitclone.com/github.com/state-spaces/mamba.git"; then
    cd /root/mamba-src && git checkout v2.2.4 2>/dev/null || echo "checkout v2.2.4 失败, 用 main"
else
    echo "--- 方案 C: 下 zip (tag v2.2.4) ---"
    if try_zip "mamba" "/root/mamba.zip" \\
        "https://codeload.github.com/state-spaces/mamba/zip/refs/tags/v2.2.4" \\
        "https://gh-proxy.com/https://github.com/state-spaces/mamba/archive/refs/tags/v2.2.4.zip"; then
        cd /root && unzip -q mamba.zip && mv mamba-2.2.4 mamba-src
        cd /root/mamba-src
    else
        echo "✗ mamba 所有镜像都失败"
        exit 1
    fi
fi

echo "=== HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo no-git) ==="
echo "=== ls csrc/selective_scan/ ==="
ls csrc/selective_scan/ 2>/dev/null | head -5

echo "=== 装 mamba-ssm 2.2.4 ==="
pip install --no-build-isolation --no-deps -e . 2>&1 | tail -30

echo ""
echo "=== 3. 验证 ==="
python -c "from mamba_ssm import Mamba2; import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__, 'OK')"
python -c "import causal_conv1d; print('causal_conv1d OK')"

echo "=== INSTALL_DONE ==="
"""


def run_cmd_streaming(cli, cmd, max_wait=1200):
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.exec_command(cmd + " 2>&1")
    out_buf = []
    last_print = time.time()
    start = time.time()
    while True:
        if chan.recv_ready():
            data = chan.recv(4096).decode("utf-8", errors="replace")
            out_buf.append(data)
            for line in data.splitlines():
                print("    " + line.rstrip())
            last_print = time.time()
        elif chan.exit_status_ready():
            while chan.recv_ready():
                data = chan.recv(4096).decode("utf-8", errors="replace")
                out_buf.append(data)
                for line in data.splitlines():
                    print("    " + line.rstrip())
            break
        else:
            time.sleep(0.5)
            elapsed = int(time.time() - start)
            if time.time() - last_print > 30:
                print(f"    [心跳 {elapsed}s] 还在跑...")
                last_print = time.time()
            if elapsed > max_wait:
                print(f"    ✗ 总超时 {max_wait}s")
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    print("\n[2] 上传 install_mamba_v2.sh ...")
    sftp = cli.open_sftp()
    with sftp.open(INSTALL_SH, "w") as f:
        f.write(INSTALL_SCRIPT)
    sftp.chmod(INSTALL_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {INSTALL_SH}")

    print("\n[3] 跑安装 (试多个镜像, 总共约 5-15 分钟) ...")
    out, rc = run_cmd_streaming(cli, f"bash {INSTALL_SH}", max_wait=1200)

    print(f"\n[4] 完成, rc={rc}")
    if "INSTALL_DONE" in out:
        print("\n✓✓✓ mamba_ssm + causal-conv1d 装好了!")
        return 0
    else:
        print("\n✗ 安装失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
