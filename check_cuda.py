"""查云机器 CUDA 编译环境 + 修安装脚本"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    CONDA = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && "

    print("\n[2] 查 nvcc + CUDA 环境 ...")
    transport = cli.get_transport()
    for cmd in [
        "which nvcc 2>&1; nvcc --version 2>&1 | head -5",
        "ls /usr/local/cuda* 2>&1 | head -5",
        CONDA + "python -c 'import torch; print(torch.utils.cpp_extension.CUDA_HOME)'",
        CONDA + "pip list 2>/dev/null | grep -iE 'torch|triton|ninja|packaging|einops'",
    ]:
        print(f"\n  $ {cmd}")
        chan = transport.open_session()
        chan.exec_command(cmd + " 2>&1")
        out = b""
        while True:
            if chan.recv_ready():
                out += chan.recv(4096)
            elif chan.exit_status_ready():
                while chan.recv_ready():
                    out += chan.recv(4096)
                break
            else:
                time.sleep(0.2)
        for line in out.decode("utf-8", errors="replace").splitlines():
            print(f"    {line}")

    cli.close()


if __name__ == "__main__":
    main()
