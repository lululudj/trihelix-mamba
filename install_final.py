"""最终安装脚本: 设 CUDA_HOME + MAMBA_FORCE_BUILD=TRUE, 强制源码编译"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
INSTALL_SH = "/root/install_mamba.sh"

INSTALL_SCRIPT = """#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# 关键: 设置 CUDA 环境 (云机器 nvcc 不在 PATH)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 关键: 强制源码编译, 不尝试下载 GitHub 预编译 wheel
export MAMBA_FORCE_BUILD=TRUE

echo "=== torch 版本 ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "=== nvcc 版本 ==="
nvcc --version | tail -2
echo "=== 装 mamba-ssm==2.2.4 (源码编译, 约 3-8 分钟) ==="
pip install --no-build-isolation --no-deps \\
    -i https://mirrors.aliyun.com/pypi/simple/ \\
    mamba-ssm==2.2.4
echo "=== 装 causal-conv1d (源码编译) ==="
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation \\
    -i https://mirrors.aliyun.com/pypi/simple/ \\
    causal-conv1d
echo "=== 验证 mamba_ssm ==="
python -c "from mamba_ssm import Mamba2; import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__, 'OK')"
echo "=== INSTALL_DONE ==="
"""


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    # [2] 上传最新 install_mamba.sh
    print("\n[2] 上传 install_mamba.sh (CUDA_HOME + MAMBA_FORCE_BUILD) ...")
    sftp = cli.open_sftp()
    with sftp.open(INSTALL_SH, "w") as f:
        f.write(INSTALL_SCRIPT)
    sftp.chmod(INSTALL_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {INSTALL_SH}")

    # [3] 前台跑 (不后台, paramiko 永久等待)
    print("\n[3] 前台跑 install_mamba.sh (mamba-ssm 编译 3-8 分钟, 请耐心等) ...")
    transport = cli.get_transport()
    channel = transport.open_session()
    channel.exec_command(f"bash {INSTALL_SH} 2>&1")

    out_buf = []
    last_print = time.time()
    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode("utf-8", errors="replace")
            out_buf.append(data)
            for line in data.splitlines():
                print("    " + line.rstrip())
            last_print = time.time()
        elif channel.exit_status_ready():
            while channel.recv_ready():
                data = channel.recv(4096).decode("utf-8", errors="replace")
                out_buf.append(data)
                for line in data.splitlines():
                    print("    " + line.rstrip())
            break
        else:
            time.sleep(0.5)
            # 每 30 秒打印心跳, 让用户知道还在跑
            if time.time() - last_print > 30:
                print(f"    [心跳] 还在编译中... 已等 {int(time.time() - last_print)}s")
                last_print = time.time()

    rc = channel.recv_exit_status()
    out = "".join(out_buf)
    print(f"\n[4] 完成, rc={rc}")

    if "INSTALL_DONE" in out:
        print("\n✓✓✓ mamba_ssm 装好了! 可启动训练")
        return 0
    elif "OK" in out and "mamba_ssm" in out:
        print("\n✓✓✓ mamba_ssm 验证 OK!")
        return 0
    else:
        print(f"\n✗ 安装失败, 末尾输出已显示")
        return 1


if __name__ == "__main__":
    sys.exit(main())
