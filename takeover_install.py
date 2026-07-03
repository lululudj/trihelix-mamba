"""远程接管: 清理错误依赖 + 装 v2.2.4/v1.4.0 + 降级 transformers
之前在云上跑的 PyPI 默认安装会拉 mamba-ssm 2.3.2.post1 (要 torch>=2.4) 失败。
本脚本走 gh-proxy.com 克隆 v2.2.4/v1.4.0 标签, 兼容 torch 2.1.2。
"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
INSTALL_SH = "/root/takeover_install.sh"

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

echo "=== 0. 清理之前装错的版本 (mamba-ssm 2.3.x / causal-conv1d 1.6.x / transformers 5.x) ==="
pip uninstall -y mamba-ssm causal-conv1d transformers tilelang quack-kernels triton 2>&1 | grep -E "(Successfully|Skipping|not installed)" | head -20 || true

echo ""
echo "=== 1. clone causal-conv1d v1.4.0 (兼容 torch 2.1.2) ==="
rm -rf /root/causal-conv1d-src
for url in \\
    "https://gh-proxy.com/https://github.com/Dao-AILab/causal-conv1d.git" \\
    "https://ghproxy.net/https://github.com/Dao-AILab/causal-conv1d.git" \\
    "https://mirror.ghproxy.com/https://github.com/Dao-AILab/causal-conv1d.git" \\
    "https://kkgithub.com/Dao-AILab/causal-conv1d.git"; do
    for tag in v1.4.0 1.4.0 v1.5.0 1.5.0; do
        echo "  试: $url @ $tag"
        if timeout 90 git clone --branch $tag --depth 1 "$url" /root/causal-conv1d-src 2>&1 | tail -3; then
            if [ -d /root/causal-conv1d-src/csrc ]; then
                echo "  ✓ 成功"
                break 2
            fi
        fi
        rm -rf /root/causal-conv1d-src
    done
done
if [ ! -d /root/causal-conv1d-src/csrc ]; then
    echo "✗ causal-conv1d clone 全失败"
    exit 1
fi

cd /root/causal-conv1d-src
echo "HEAD: $(git log --oneline -1)"

echo ""
echo "=== 2. 装 causal-conv1d (编译 1-3 分钟) ==="
pip install --no-build-isolation -e . 2>&1 | tail -15

echo ""
echo "=== 3. clone mamba v2.2.4 (兼容 torch 2.1.2) ==="
rm -rf /root/mamba-src
for url in \\
    "https://gh-proxy.com/https://github.com/state-spaces/mamba.git" \\
    "https://ghproxy.net/https://github.com/state-spaces/mamba.git" \\
    "https://mirror.ghproxy.com/https://github.com/state-spaces/mamba.git" \\
    "https://kkgithub.com/state-spaces/mamba.git"; do
    for tag in v2.2.4 2.2.4; do
        echo "  试: $url @ $tag"
        if timeout 120 git clone --branch $tag --depth 1 "$url" /root/mamba-src 2>&1 | tail -3; then
            if [ -d /root/mamba-src/csrc/selective_scan ]; then
                echo "  ✓ 成功"
                break 2
            fi
        fi
        rm -rf /root/mamba-src
    done
done
if [ ! -d /root/mamba-src/csrc/selective_scan ]; then
    echo "✗ mamba clone 全失败"
    exit 1
fi

cd /root/mamba-src
echo "HEAD: $(git log --oneline -1)"

echo ""
echo "=== 4. 装 mamba-ssm 2.2.4 (编译 3-8 分钟) ==="
pip install --no-build-isolation --no-deps -e . 2>&1 | tail -30

echo ""
echo "=== 5. 装运行时依赖 + 降级 transformers 到 4.40.0 ==="
pip install --upgrade-strategy only-if-needed \\
    -i https://mirrors.aliyun.com/pypi/simple/ \\
    einops "transformers==4.40.0" huggingface_hub 2>&1 | tail -10

echo ""
echo "=== 6. 验证 ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "from mamba_ssm import Mamba2; import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__, 'OK')"
python -c "import transformers; print('transformers', transformers.__version__)"

echo ""
echo "=== 全部 import 一次 ==="
python -c "
from mamba_ssm import Mamba2
from mamba_ssm.modules.mamba2 import Mamba2 as M2
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import causal_conv1d
import einops
import torch
import transformers
print('ALL OK, torch=', torch.__version__, 'transformers=', transformers.__version__)
print('mamba_ssm=', __import__('mamba_ssm').__version__)
"

echo "=== TAKEOVER_INSTALL_DONE ==="
"""


def run_cmd_streaming(cli, cmd, max_wait=1800):
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
                print(f"    ✗ 总超时 {max_wait}s")
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接云服务器 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    except Exception as e:
        print(f"    ✗ 连接失败: {e}")
        print("    可能 SSH 信息变了。请告诉我新的端口/密码, 或在 AutoDL 控制台确认实例还在运行。")
        return 1
    print("    ✓ 连上了")

    print("\n[2] 上传合并安装脚本 ...")
    sftp = cli.open_sftp()
    with sftp.open(INSTALL_SH, "w") as f:
        f.write(INSTALL_SCRIPT)
    sftp.chmod(INSTALL_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {INSTALL_SH}")

    print("\n[3] 跑安装 (清理 + clone v2.2.4/v1.4.0 + 编译 + 降级 transformers, 约 8-15 分钟) ...")
    out, rc = run_cmd_streaming(cli, f"bash {INSTALL_SH}", max_wait=1800)

    print(f"\n[4] rc={rc}")
    if "TAKEOVER_INSTALL_DONE" in out:
        print("\n✓✓✓ mamba_ssm 2.2.4 + transformers 4.40.0 全部就绪! 可以启动训练")
        return 0
    else:
        print("\n✗ 安装未完成, 看上面日志")
        return 1


if __name__ == "__main__":
    sys.exit(main())
