"""从 GitHub git clone 安装 mamba-ssm 2.2.4 + causal-conv1d
PyPI 上的 tarball 缺 .cpp 源文件, 必须用 GitHub 完整源码
"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
INSTALL_SH = "/root/install_mamba_github.sh"

INSTALL_SCRIPT = """#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS=4  # 防 OOM

echo "=== 1. clone causal-conv1d (小, ~1MB) ==="
rm -rf /root/causal-conv1d-src
git clone --depth 1 https://github.com/Dao-AILab/causal-conv1d.git /root/causal-conv1d-src
cd /root/causal-conv1d-src
echo "HEAD: $(git rev-parse --short HEAD)"

echo "=== 2. 装 causal-conv1d (编译约 1-3 分钟) ==="
pip install --no-build-isolation -e .

echo "=== 3. clone mamba-ssm v2.2.4 (含完整 csrc/) ==="
rm -rf /root/mamba-src
git clone --branch v2.2.4 --depth 1 https://github.com/state-spaces/mamba.git /root/mamba-src
cd /root/mamba-src
echo "HEAD: $(git rev-parse --short HEAD)"
echo "ls csrc/selective_scan/:"
ls csrc/selective_scan/ | head -10

echo "=== 4. 装 mamba-ssm==2.2.4 (编译约 3-8 分钟) ==="
pip install --no-build-isolation --no-deps -e .

echo "=== 5. 验证 ==="
python -c "from mamba_ssm import Mamba2; import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__, 'OK')"
python -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__ if hasattr(causal_conv1d, '__version__') else '?', 'OK')"

echo "=== INSTALL_DONE ==="
"""


def run_cmd_streaming(cli, cmd, timeout=900):
    """流式打印输出, 永久等待 (timeout 仅用于 read 单次)"""
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.exec_command(cmd + " 2>&1")
    out_buf = []
    last_print = time.time()
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
            if time.time() - last_print > 30:
                print(f"    [心跳] 编译中... ({int(time.time() - last_print)}s since last output)")
                last_print = time.time()
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    # 上传 install 脚本
    print("\n[2] 上传 install_mamba_github.sh ...")
    sftp = cli.open_sftp()
    with sftp.open(INSTALL_SH, "w") as f:
        f.write(INSTALL_SCRIPT)
    sftp.chmod(INSTALL_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {INSTALL_SH}")

    # 跑安装
    print("\n[3] 跑安装 (git clone + 编译, 总共约 5-12 分钟) ...")
    out, rc = run_cmd_streaming(cli, f"bash {INSTALL_SH}", timeout=1200)

    print(f"\n[4] 完成, rc={rc}")
    if "INSTALL_DONE" in out:
        print("\n✓✓✓ mamba_ssm 2.2.4 + causal-conv1d 装好了! 可启动训练")
        return 0
    else:
        print("\n✗ 安装失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
