"""上传 install.sh 到云机器 + nohup 执行 + 轮询
避免 shell 嵌套引号问题: 把要执行的命令写成脚本文件
"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"
INSTALL_LOG = "/root/install_mamba.log"
INSTALL_SH = "/root/install_mamba.sh"

# 安装脚本内容 (写到云机器 /root/install_mamba.sh)
INSTALL_SCRIPT = """#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
echo "=== torch 版本 ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "=== 装 mamba-ssm==2.2.4 + causal-conv1d ==="
pip install --no-build-isolation --no-deps \\
    -i https://mirrors.aliyun.com/pypi/simple/ \\
    mamba-ssm==2.2.4
pip install --no-build-isolation \\
    -i https://mirrors.aliyun.com/pypi/simple/ \\
    causal-conv1d
echo "=== 验证 ==="
python -c "from mamba_ssm import Mamba2; import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__, 'OK')"
echo "=== INSTALL_DONE ==="
"""


def connect():
    print(f"[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")
    return cli


def run(cli, cmd, timeout=60, show=True):
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=True)
    out_buf = []
    end_time = time.time() + timeout
    while time.time() < end_time:
        if stdout.channel.exit_status_ready():
            break
        line = stdout.readline()
        if line:
            out_buf.append(line)
            if show:
                print("    " + line.rstrip())
        else:
            time.sleep(0.2)
    rc = stdout.channel.recv_exit_status() if stdout.channel.exit_status_ready() else -1
    return rc, "".join(out_buf)


def main():
    cli = connect()

    # [2] 杀残留 pip
    print("\n[2] 杀残留 pip ...")
    run(cli, "pkill -9 -f 'pip install' 2>/dev/null; pkill -9 -f 'setup.py' 2>/dev/null; sleep 2; echo killed", show=False)

    # [3] 上传 install.sh
    print("\n[3] 上传 install_mamba.sh ...")
    sftp = cli.open_sftp()
    with sftp.open(INSTALL_SH, "w") as f:
        f.write(INSTALL_SCRIPT)
    sftp.chmod(INSTALL_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {INSTALL_SH}")

    # [4] setsid 完全脱离会话执行 (避免 PTY 关闭杀死子进程)
    print("\n[4] setsid 后台执行 install_mamba.sh ...")
    # setsid + & + < /dev/null 让进程完全脱离控制终端
    run(cli, f"rm -f {INSTALL_LOG}; setsid bash -c 'bash {INSTALL_SH} > {INSTALL_LOG} 2>&1 < /dev/null' &", timeout=10, show=False)
    time.sleep(5)

    # 确认在跑
    rc, out = run(cli, "ps aux | grep -E 'pip|install_mamba' | grep -v grep", show=False)
    if "pip" in out or "install_mamba" in out:
        print("    ✓ 后台运行中")
    else:
        print("    ✗ 未启动, 查日志:")
        run(cli, f"cat {INSTALL_LOG}")
        sys.exit(1)

    # [5] 轮询 (最多 15 分钟)
    print(f"\n[5] 轮询日志 (最多 15 分钟) ...")
    max_wait = 900
    waited = 0
    while waited < max_wait:
        time.sleep(30)
        waited += 30
        rc, out = run(cli, f"tail -3 {INSTALL_LOG} 2>&1; echo '---SEP---'; ps aux | grep -E 'pip install|install_mamba.sh' | grep -v grep | wc -l", show=False)
        parts = out.split("---SEP---")
        log_tail = parts[0].strip().splitlines() if parts else []
        proc_count = parts[1].strip() if len(parts) > 1 else "0"
        last_log = log_tail[-1] if log_tail else "(空)"
        print(f"  [{waited}s] 进程={proc_count}, 末尾: {last_log[:120]}")

        if "INSTALL_DONE" in out:
            print("\n    ✓✓✓ 安装完成!")
            break
        if "Error" in out or "error:" in out.lower():
            if proc_count == "0":
                print("\n    ✗ 安装出错, 完整日志:")
                run(cli, f"cat {INSTALL_LOG}")
                sys.exit(1)
        if proc_count == "0":
            # 进程结束但没看到 DONE 标志
            rc, tail = run(cli, f"tail -10 {INSTALL_LOG}", show=False)
            if "INSTALL_DONE" in tail:
                print("\n    ✓✓✓ 安装完成!")
                break
            print(f"\n    ⚠ 进程结束, 末尾:")
            print(tail)
            if "OK" in tail and "mamba_ssm" in tail:
                print("    ✓ 看到验证 OK, 视为成功")
                break
            sys.exit(1)
    else:
        print(f"\n    ✗ 等待超时")
        run(cli, f"tail -30 {INSTALL_LOG}")
        sys.exit(1)

    cli.close()
    print("\n[完成] mamba_ssm 就绪, 可启动训练")


if __name__ == "__main__":
    main()
