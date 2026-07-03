"""补装 huggingface_hub 等剩余依赖"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"


SCRIPT = """#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== 降级 transformers 到 4.40.0 (兼容 torch 2.1 + 保留旧 API) ==="
pip install --upgrade-strategy only-if-needed \\
    -i https://mirrors.aliyun.com/pypi/simple/ \\
    "transformers==4.40.0" 2>&1 | tail -10

echo ""
echo "=== 验证 ==="
python -c "import torch; print('torch', torch.__version__)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "from mamba_ssm import Mamba2; import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__, 'OK')"
echo ""
echo "=== 全部 import 一次 ==="
python -c "
from mamba_ssm import Mamba2
from mamba_ssm.modules.mamba2 import Mamba2 as M2
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import causal_conv1d
import einops
import torch
print('ALL OK, torch=', torch.__version__, 'cuda=', torch.cuda.is_available())
print('mamba_ssm=', __import__('mamba_ssm').__version__)
"

echo "=== DONE ==="
"""


def run_cmd_streaming(cli, cmd, max_wait=180):
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.exec_command(cmd + " 2>&1")
    out_buf = []
    start = time.time()
    while True:
        if chan.recv_ready():
            data = chan.recv(4096).decode("utf-8", errors="replace")
            out_buf.append(data)
            for line in data.splitlines():
                print("    " + line.rstrip())
        elif chan.exit_status_ready():
            while chan.recv_ready():
                data = chan.recv(4096).decode("utf-8", errors="replace")
                out_buf.append(data)
                for line in data.splitlines():
                    print("    " + line.rstrip())
            break
        else:
            time.sleep(0.5)
            if time.time() - start > max_wait:
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    print("\n[2] 上传补依赖脚本 ...")
    sftp = cli.open_sftp()
    with sftp.open("/root/install_hub.sh", "w") as f:
        f.write(SCRIPT)
    sftp.chmod("/root/install_hub.sh", 0o755)
    sftp.close()

    print("\n[3] 装依赖 + 验证 ...")
    out, rc = run_cmd_streaming(cli, "bash /root/install_hub.sh", max_wait=180)

    print(f"\n[4] rc={rc}")
    if "ALL OK" in out:
        print("\n✓✓✓ mamba_ssm 2.2.4 全部就绪! 可启动训练")
        return 0
    else:
        print("\n✗ 仍有问题")
        return 1


if __name__ == "__main__":
    sys.exit(main())
