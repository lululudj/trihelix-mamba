"""测 GitHub 可达性, 然后从 GitHub 源码装 mamba-ssm 2.2.4"""
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
export MAX_JOBS=4

echo "=== 测 GitHub 可达性 ==="
curl -sI --max-time 15 https://github.com/state-spaces/mamba | head -3 || echo "GitHub 直连失败"
echo ""
echo "=== 测 ghproxy.com 镜像 ==="
curl -sI --max-time 15 https://ghproxy.com/https://github.com/state-spaces/mamba | head -3 || echo "ghproxy 失败"
echo ""
echo "=== 测 kkgithub.com 镜像 ==="
curl -sI --max-time 15 https://kkgithub.com/state-spaces/mamba | head -3 || echo "kkgithub 失败"
echo ""
echo "=== 测 gitee 镜像 (state-spaces/mamba 是否有镜像) ==="
curl -sI --max-time 15 https://gitee.com/gh-mirror/mamba | head -3 || echo "gitee 失败"
echo ""
echo "=== DONE ==="
"""


def run_cmd(cli, cmd, timeout=60):
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.exec_command(cmd + " 2>&1")
    out_buf = []
    while True:
        if chan.recv_ready():
            out_buf.append(chan.recv(4096).decode("utf-8", errors="replace"))
        elif chan.exit_status_ready():
            while chan.recv_ready():
                out_buf.append(chan.recv(4096).decode("utf-8", errors="replace"))
            break
        else:
            time.sleep(0.2)
    return "".join(out_buf)


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    # 上传 + 跑测试脚本
    print("\n[2] 上传 + 跑 GitHub 可达性测试 ...")
    sftp = cli.open_sftp()
    with sftp.open(INSTALL_SH, "w") as f:
        f.write(INSTALL_SCRIPT)
    sftp.chmod(INSTALL_SH, 0o755)
    sftp.close()

    out = run_cmd(cli, f"bash {INSTALL_SH}", timeout=120)
    print(out)

    cli.close()


if __name__ == "__main__":
    main()
