"""修复 mamba_ssm 安装 - 改用 2.2.4 版本
原因: 默认装最新版 2.3.x, 依赖 triton>=3.5 + cuda-python>=12.8, 会拖一堆大包并可能换 torch
本地项目用 mamba_ssm 2.2.4, 云机器应保持一致
"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"
LOG_FILE = "/root/three_chain_v3/cloud_train.log"
CONDA_INIT = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && "


def connect():
    print(f"[1] 连接 {HOST}:{PORT} ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")
    return cli


def run(cli, cmd, timeout=600, show=True):
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=True)
    out_buf = []
    while not stdout.channel.exit_status_ready():
        line = stdout.readline()
        if line:
            out_buf.append(line)
            if show:
                print("    " + line.rstrip())
        else:
            time.sleep(0.2)
    err = stderr.read().decode("utf-8", errors="replace")
    if err and show:
        for ln in err.splitlines():
            print("    [stderr] " + ln)
    rc = stdout.channel.recv_exit_status()
    return rc, "".join(out_buf), err


def main():
    cli = connect()

    # [2] 杀掉可能残留的 pip 进程
    print("\n[2] 杀掉残留 pip 进程 ...")
    run(cli, "pkill -f 'pip install' 2>/dev/null; pkill -f 'setup.py' 2>/dev/null; sleep 1; echo done", show=False)

    # [3] 查 torch 版本 (决定 mamba_ssm 兼容版本)
    print("\n[3] 查 torch 版本 ...")
    rc, out, _ = run(cli, CONDA_INIT + "python -c 'import torch; print(\"torch=\", torch.__version__); print(\"cuda=\", torch.version.cuda); print(\"avail=\", torch.cuda.is_available())' 2>&1")

    # [4] 装 mamba-ssm==2.2.4 + causal-conv1d (匹配本地环境)
    print("\n[4] 装 mamba-ssm==2.2.4 + causal-conv1d (匹配本地) ...")
    # 注意: causal-conv1d 需要匹配 torch + cuda 版本, 用阿里云镜像加速
    run(cli, CONDA_INIT + "pip install packaging ninja -q")
    # 关键: 指定版本 + 不升级依赖 + 用国内镜像
    cmd_install = (
        CONDA_INIT +
        "pip install --no-build-isolation --no-deps "
        "-i https://mirrors.aliyun.com/pypi/simple/ "
        "mamba-ssm==2.2.4 2>&1"
    )
    rc, out, _ = run(cli, cmd_install, timeout=600)
    if rc != 0:
        print(f"    ✗ mamba-ssm 安装失败 (rc={rc})")
        sys.exit(1)

    # 装 causal-conv1d (mamba-ssm 2.2.4 的依赖)
    print("\n[5] 装 causal-conv1d ...")
    rc, out, _ = run(cli, CONDA_INIT + "pip install --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ causal-conv1d 2>&1", timeout=600)
    if rc != 0:
        print(f"    ✗ causal-conv1d 安装失败 (rc={rc})")
        # 不致命, mamba-ssm 可能自带 fallback
    else:
        print("    ✓ causal-conv1d 装好")

    # [6] 验证 mamba_ssm 可用
    print("\n[6] 验证 mamba_ssm ...")
    rc, out, _ = run(cli, CONDA_INIT + "python -c 'from mamba_ssm import Mamba2; print(\"mamba_ssm 装好了, version=\", __import__(\"mamba_ssm\").__version__)' 2>&1")
    if "装好了" not in out:
        print(f"    ✗ 验证失败: {out}")
        sys.exit(1)
    print("    ✓ mamba_ssm 可用")

    cli.close()
    print("\n[完成] 环境就绪，可启动训练")


if __name__ == "__main__":
    main()
