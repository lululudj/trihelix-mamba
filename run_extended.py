"""补跑实验: 300M 加长到 2000 步 + 100M seed0 加长到 500 步
解决两个问题:
1. 300M seed0 之前 100 步未收敛 (ch@100=0.0557),加长看能否收敛
2. 100M seed0 ch@100 偏低 (0.4645),加长过 warmup (500步)
"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"
RUNNER_SH = "/root/run_extended.sh"

# 补跑实验列表: (exp_name, config, model, seed, batch, max_steps)
EXTENDED_EXPERIMENTS = [
    # 300M 加长到 2000 步 (warmup 500 + 训练 1500), 验证大模型能否收敛并保持零衰减
    ("300m_seed0_2k", "matched_mamba2_300m.yaml", "three_chain_mamba2", 0, 1, 2000),
    # 100M seed0 加长到 500 步 (刚好过 warmup), 解决 ch@100 偏低
    ("100m_seed0_500", "matched_mamba2_100m.yaml", "three_chain_mamba2", 0, 2, 500),
    # 100M HTA seed0 加长到 500 步, 看是否同样随训练步数上升
    ("100m_hta_seed0_500", "matched_mamba2_100m.yaml", "three_chain_mamba2_hta", 0, 2, 500),
]


def build_runner_script():
    calls = []
    for exp_name, config, model, seed, batch, max_steps in EXTENDED_EXPERIMENTS:
        calls.append(f'run_experiment "{exp_name}" "{config}" "{model}" {seed} {batch} {max_steps}')

    calls_str = "\n".join(calls)

    return f"""#!/bin/bash
# 三链 DNA-Mamba2 补跑实验 (加长训练步数)
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/three_chain_v3

echo "=========================================="
echo "  补跑实验 (加长训练步数)"
echo "  实验数: {len(EXTENDED_EXPERIMENTS)}"
echo "  开始时间: $(date)"
echo "=========================================="

# 实验函数: 训练 + eval, 失败不中断后续
run_experiment() {{
    local exp_name=$1
    local config=$2
    local model=$3
    local seed=$4
    local batch=$5
    local max_steps=$6
    local exp_dir="results/run_${{exp_name}}"

    echo ""
    echo "=========================================="
    echo "  实验: $exp_name (model=$model seed=$seed batch=$batch max_steps=$max_steps)"
    echo "  $(date)"
    echo "=========================================="
    mkdir -p "$exp_dir"

    # 训练
    echo "[训练] 开始 ($max_steps 步, batch=$batch)..."
    python -u train.py \\
        --model $model \\
        --config configs/$config \\
        --max_steps $max_steps \\
        --batch_size $batch \\
        --seed $seed \\
        --data_root ./data_split_m3 \\
        --out_dir "$exp_dir" 2>&1 | tail -25
    local train_rc=${{PIPESTATUS[0]}}
    echo "[训练] rc=$train_rc"

    if [ $train_rc -ne 0 ]; then
        echo "✗ $exp_name 训练失败"
        echo "$exp_name: TRAIN_FAILED rc=$train_rc" >> /root/extended_summary.txt
        return
    fi

    # 读取 summary
    if [ -f "$exp_dir/summary.json" ]; then
        echo "[summary]:"
        cat "$exp_dir/summary.json"
    fi

    # eval_ood
    echo ""
    echo "[OOD 评估] ..."
    python -u eval_ood.py \\
        --ckpt "$exp_dir/final.pt" \\
        --config configs/$config \\
        --model $model \\
        --out "$exp_dir/ood_metrics.json" 2>&1 | tail -20
    local eval_rc=${{PIPESTATUS[0]}}
    echo "[OOD] rc=$eval_rc"

    # 读 OOD 结果
    if [ -f "$exp_dir/ood_metrics.json" ]; then
        echo "[OOD 结果]:"
        python -c "
import json
with open('$exp_dir/ood_metrics.json') as f:
    m = json.load(f)
ch100 = m.get('changed_acc', {{}}).get('t=100', None)
ch150 = m.get('changed_acc', {{}}).get('t=150', None)
decay = m.get('ood_decay_pct', None)
zr = m.get('zero_ratio', None)
if ch100 is not None and ch150 is not None and decay is not None:
    print(f'  ch@100={{ch100:.4f}} ch@150={{ch150:.4f}} decay={{decay*100:+.1f}}% zero_ratio={{zr:.3f}}')
    line='$exp_name: decay={{decay*100:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}'
    echo $line >> /root/extended_summary.txt
else:
    print('  [解析失败]', list(m.keys()))
"
    fi

    echo ""
    echo "[实验 $exp_name 完成] $(date)"
}}

rm -f /root/extended_summary.txt
touch /root/extended_summary.txt

{calls_str}

echo ""
echo "=========================================="
echo "  全部补跑实验完成"
echo "  结束时间: $(date)"
echo "=========================================="
echo ""
echo "=== 汇总 ==="
cat /root/extended_summary.txt

echo "=== EXTENDED_DONE ==="
"""


def run_cmd_streaming(cli, cmd, max_wait=3600):
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
                elapsed = int(time.time() - start)
                print(f"    [心跳 {elapsed}s] ...")
                last_print = time.time()
            if time.time() - start > max_wait:
                print(f"    ✗ 总超时 {max_wait}s")
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    print("\n[2] 上传补跑脚本 ...")
    sftp = cli.open_sftp()
    with sftp.open(RUNNER_SH, "w") as f:
        f.write(build_runner_script())
    sftp.chmod(RUNNER_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {RUNNER_SH}")

    print("\n[3] 跑补跑实验 (300M 2k步 + 100M seed0 500步 + 100M HTA 500步, 约 20-40 分钟) ...")
    out, rc = run_cmd_streaming(cli, f"bash {RUNNER_SH}", max_wait=3600)

    print(f"\n[4] rc={rc}")
    if "EXTENDED_DONE" in out:
        print("\n✓✓✓ 补跑实验全部完成!")
        # 打印汇总
        if "extended_summary.txt" in out:
            idx = out.rfind("=== 汇总 ===")
            if idx >= 0:
                print("\n最终汇总:")
                print(out[idx:])
        return 0
    else:
        print("\n✗ 补跑实验未完成, 看上面日志")
        return 1


if __name__ == "__main__":
    sys.exit(main())
