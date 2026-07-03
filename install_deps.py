"""补装 mamba-ssm 缺失的运行时依赖 (einops 等), 不动 torch"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"


INSTALL_SCRIPT = """#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== 1. 装缺失的运行时依赖 (不动 torch) ==="
# einops 是 mamba_ssm 的硬依赖
# transformers 仅用于 huggingface 集成, 不必装完整版, 但 mamba_ssm 模块 import 时可能需要
# 用 --upgrade-strategy only-if-needed 防止 torch 被升级
pip install --upgrade-strategy only-if-needed \\
    -i https://mirrors.aliyun.com/pypi/simple/ \\
    einops 2>&1 | tail -5

echo ""
echo "=== 2. 试 import, 看缺啥 ==="
python -c "
import sys
try:
    from mamba_ssm import Mamba2
    import mamba_ssm
    print('mamba_ssm', mamba_ssm.__version__, 'OK')
except ImportError as e:
    print('MISSING:', e)
    sys.exit(1)
" 2>&1

echo ""
echo "=== 3. 装 causal-conv1d 验证 ==="
python -c "import causal_conv1d; print('causal_conv1d OK')" 2>&1 || echo "causal-conv1d 仍缺失"

echo ""
echo "=== 4. torch 仍是 2.1.2 ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1

echo "=== DONE ==="
"""


def run_cmd_streaming(cli, cmd, max_wait=300):
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
            if time.time() - last_print > 30:
                print(f"    [心跳 {int(time.time() - start)}s] ...")
                last_print = time.time()
            if time.time() - start > max_wait:
                print(f"    ✗ 超时")
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    print("\n[2] 上传 install_deps.sh ...")
    sftp = cli.open_sftp()
    with sftp.open("/root/install_deps.sh", "w") as f:
        f.write(INSTALL_SCRIPT)
    sftp.chmod("/root/install_deps.sh", 0o755)
    sftp.close()

    print("\n[3] 装依赖 + 验证 ...")
    out, rc = run_cmd_streaming(cli, "bash /root/install_deps.sh", max_wait=300)

    print(f"\n[4] rc={rc}")
    if "mamba_ssm" in out and "OK" in out and "MISSING" not in out:
        print("\n✓✓✓ mamba_ssm 可用! 可启动训练")
        return 0
    else:
        print("\n✗ 仍有缺失")
        return 1


if __name__ == "__main__":
    sys.exit(main())
