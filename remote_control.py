"""远程接管 AutoDL 云机器 - 修正版
云机器用 conda, python 在 /root/miniconda3/bin/
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
# conda 初始化前缀 (每次 exec_command 都是新 shell)
CONDA_INIT = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && "


def connect():
    print(f"[1] 连接 {HOST}:{PORT} ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")
    return cli


def run(cli, cmd, timeout=600, show=True):
    """执行命令并流式打印输出"""
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

    # [2] 环境检查 - 用 conda activate
    print("\n[2] 环境检查 ...")
    run(cli, CONDA_INIT + "which python && python --version")
    run(cli, CONDA_INIT + "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    run(cli, CONDA_INIT + "python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'")

    # [3] 装 mamba_ssm (关键: --no-build-isolation)
    print("\n[3] 检查 mamba_ssm ...")
    rc, out, _ = run(cli, CONDA_INIT + "python -c 'from mamba_ssm import Mamba2; print(\"mamba_ssm 已装好\")' 2>&1", show=False)
    if "已装好" in out:
        print("    ✓ mamba_ssm 已装好，跳过安装")
    else:
        print("    mamba_ssm 未装，开始安装 (--no-build-isolation, 约 2-5 分钟) ...")
        run(cli, CONDA_INIT + "pip install packaging ninja -q")
        rc, _, _ = run(cli, CONDA_INIT + "pip install --no-build-isolation mamba-ssm causal-conv1d 2>&1", timeout=600)
        if rc != 0:
            print(f"    ✗ 安装失败 (rc={rc})")
            sys.exit(1)
        rc, out, _ = run(cli, CONDA_INIT + "python -c 'from mamba_ssm import Mamba2; print(\"mamba_ssm 装好了\")' 2>&1", show=False)
        if "装好了" not in out:
            print(f"    ✗ 二次验证失败: {out}")
            sys.exit(1)
        print("    ✓ mamba_ssm 装好了")

    # [4] 检查代码目录
    print("\n[4] 检查代码目录 ...")
    run(cli, f"ls -la {WORKDIR}/train.py {WORKDIR}/run_cloud_30m.sh {WORKDIR}/configs/matched_mamba2_30m.yaml 2>&1")
    run(cli, f"ls {WORKDIR}/data_split_m3/ 2>&1 | head -5")
    run(cli, f"ls {WORKDIR}/data/ood_T150/ 2>&1 | head -5")

    # [5] 后台启动训练 (nohup)
    print("\n[5] 后台启动训练 (nohup, 日志: cloud_train.log) ...")
    run(cli, "pkill -f 'train.py.*three_chain_mamba2' 2>/dev/null; sleep 1", show=False)
    # nohup 后台跑完整脚本 (需 conda activate 才能 python 可用)
    cmd = (
        f"cd {WORKDIR} && "
        f"nohup bash -c '{CONDA_INIT} bash run_cloud_30m.sh' > {LOG_FILE} 2>&1 &"
    )
    run(cli, cmd, show=False)
    time.sleep(3)
    rc, out, _ = run(cli, "ps aux | grep -E 'train.py|run_cloud' | grep -v grep", show=False)
    if "train.py" in out or "run_cloud" in out:
        print("    ✓ 训练进程已启动 (后台运行)")
    else:
        print("    ⚠ 未发现训练进程，检查日志:")
        run(cli, f"tail -30 {LOG_FILE}")
        sys.exit(1)

    print(f"\n[完成] 训练已在后台运行")
    print(f"  日志: {LOG_FILE}")
    print(f"  下一步: 运行 poll_results.py 查看进度")

    cli.close()


if __name__ == "__main__":
    main()
