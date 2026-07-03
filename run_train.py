"""启动训练: wrapper 脚本激活 conda, nohup 后台跑, 轮询日志"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"
TRAIN_LOG = "/root/cloud_train.log"
WRAPPER_SH = "/root/run_train_wrapper.sh"

# Wrapper: 激活 conda + 跑 run_cloud_30m.sh
WRAPPER_SCRIPT = """#!/bin/bash
# 激活 conda 环境 (云机器默认 PATH 没有python)
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# 设 CUDA 环境 (mamba_ssm 编译时已用, 运行时也需)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /root/three_chain_v3
echo "=== 环境检查 ==="
echo "python: $(which python)"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "mamba_ssm: $(python -c 'import mamba_ssm; print(mamba_ssm.__version__)')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""
echo "=== 启动 run_cloud_30m.sh ==="
bash run_cloud_30m.sh
echo ""
echo "=== TRAIN_DONE ==="
"""


def run_cmd_streaming(cli, cmd, max_wait=1800):
    """流式打印 + 永久等待 (训练 20 分钟)"""
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.exec_command(cmd + " 2>&1")
    out_buf = []
    last_print = time.time()
    start = time.time()
    last_heartbeat = 0
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
            time.sleep(1)
            elapsed = int(time.time() - start)
            # 每 60 秒打一次心跳
            if elapsed - last_heartbeat >= 60:
                print(f"    [心跳 {elapsed}s] 训练中... (正常, 100 步 ~10-15 分钟)")
                last_heartbeat = elapsed
            if elapsed > max_wait:
                print(f"    ✗ 总超时 {max_wait}s")
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    print("\n[2] 上传 run_train_wrapper.sh ...")
    sftp = cli.open_sftp()
    with sftp.open(WRAPPER_SH, "w") as f:
        f.write(WRAPPER_SCRIPT)
    sftp.chmod(WRAPPER_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {WRAPPER_SH}")

    print("\n[3] 启动训练 (前台流式打印, 100 步训练 + OOD eval, 约 15-20 分钟) ...")
    print("    " + "=" * 60)
    out, rc = run_cmd_streaming(cli, f"bash {WRAPPER_SH}", max_wait=1800)
    print("    " + "=" * 60)
    print(f"\n[4] 训练完成, rc={rc}")

    if "TRAIN_DONE" in out:
        print("\n🎉🎉🎉 训练 + OOD eval 全部完成!")
        # 提取关键结果
        if "OOD decay" in out:
            for line in out.splitlines():
                if "decay" in line.lower() or "changed_acc" in line.lower() or "趋势" in line or "退化" in line or "✅" in line or "❌" in line or "⚠" in line:
                    print(f"  {line.strip()}")
        return 0
    else:
        print("\n✗ 训练中断")
        return 1


if __name__ == "__main__":
    sys.exit(main())
