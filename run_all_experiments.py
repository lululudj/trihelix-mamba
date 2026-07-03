"""多实验自动运行 (修正版, 用 bash 函数)"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"
RUNNER_SH = "/root/run_all_experiments.sh"

# 实验列表: (exp_name, config, model, seed, batch)
EXPERIMENTS = [
    ("100m_seed0",    "matched_mamba2_100m.yaml", "three_chain_mamba2",     0, 2),
    ("100m_seed1",    "matched_mamba2_100m.yaml", "three_chain_mamba2",     1, 2),
    ("100m_seed2",    "matched_mamba2_100m.yaml", "three_chain_mamba2",     2, 2),
    ("30m_seed1",     "matched_mamba2_30m.yaml",  "three_chain_mamba2",     1, 2),
    ("30m_seed2",     "matched_mamba2_30m.yaml",  "three_chain_mamba2",     2, 2),
    ("300m_seed0",    "matched_mamba2_300m.yaml", "three_chain_mamba2",     0, 1),
    ("100m_hta_seed0","matched_mamba2_100m.yaml", "three_chain_mamba2_hta", 0, 2),
]


def build_runner_script():
    """用 bash 函数 + 顺序调用, 清晰可靠"""
    # 实验调用列表
    calls = []
    for exp_name, config, model, seed, batch in EXPERIMENTS:
        calls.append(f'run_experiment "{exp_name}" "{config}" "{model}" {seed} {batch}')

    calls_str = "\n".join(calls)

    return f"""#!/bin/bash
# 三链 DNA-Mamba2 多实验自动运行 (7 实验, 约 1.5-2 小时)
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /root/three_chain_v3

echo "=========================================="
echo "  三链 DNA-Mamba2 多实验自动运行"
echo "  实验数: {len(EXPERIMENTS)} (预计 1.5-2 小时)"
echo "  开始时间: $(date)"
echo "=========================================="
rm -f /root/experiments_summary.txt
touch /root/experiments_summary.txt

# 实验函数: 训练 + eval, 失败不中断后续
run_experiment() {{
    local exp_name=$1
    local config=$2
    local model=$3
    local seed=$4
    local batch=$5
    local exp_dir="results/run_${{exp_name}}"

    echo ""
    echo "=========================================="
    echo "  实验: $exp_name (model=$model seed=$seed batch=$batch config=$config)"
    echo "  $(date)"
    echo "=========================================="
    mkdir -p "$exp_dir"

    # 训练
    echo "[训练] 开始 (100 步, batch=$batch)..."
    python train.py \\
        --model $model \\
        --config configs/$config \\
        --max_steps 100 \\
        --batch_size $batch \\
        --seed $seed \\
        --data_root ./data_split_m3 \\
        --out_dir "$exp_dir" 2>&1 | tail -15
    local train_rc=${{PIPESTATUS[0]}}
    echo "[训练] rc=$train_rc"

    if [ $train_rc -ne 0 ]; then
        echo "[跳过 eval] 训练失败 (可能 OOM 或错误)"
        echo "$exp_name: TRAIN_FAILED (rc=$train_rc)" >> /root/experiments_summary.txt
        return 1
    fi

    # OOD eval
    echo "[OOD eval] 开始..."
    python eval_ood.py \\
        --checkpoint "$exp_dir/final.pt" \\
        --config configs/$config \\
        --data_root ./data/ood_T150 \\
        --batch_size 4 \\
        --seed $seed \\
        --out "$exp_dir/ood_metrics.json" 2>&1 | tail -10

    if [ -f "$exp_dir/ood_metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('$exp_dir/ood_metrics.json'))
ch100 = m['changed_acc_curve']['100']
ch150 = m['changed_acc_curve']['150']
decay = m['ood_decay_pct'] * 100
print(f'  $exp_name: decay={{decay:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}')
with open('/root/experiments_summary.txt', 'a') as f:
    f.write(f'$exp_name: decay={{decay:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}\\n')
" 2>&1
        echo "$exp_name: OK"
        return 0
    else
        echo "$exp_name: EVAL_FAILED" >> /root/experiments_summary.txt
        return 1
    fi
}}

# 顺序跑所有实验 (失败不中断)
{calls_str}

# 汇总
echo ""
echo "=========================================="
echo "  全部实验完成: $(date)"
echo "=========================================="
echo ""
echo "=== 结果汇总 ==="
cat /root/experiments_summary.txt
echo ""
echo "=== ALL_DONE ==="
"""


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    # [2] 上传 configs
    print("\n[2] 上传 configs ...")
    sftp = cli.open_sftp()
    for cfg in ["matched_mamba2_100m.yaml", "matched_mamba2_300m.yaml"]:
        local = f"e:/three_chain_v3/configs/{cfg}"
        remote = f"{WORKDIR}/configs/{cfg}"
        sftp.put(local, remote)
        print(f"    ✓ {cfg}")

    # [3] 上传 runner
    print("\n[3] 上传 run_all_experiments.sh ...")
    script = build_runner_script()
    with sftp.open(RUNNER_SH, "w") as f:
        f.write(script)
    sftp.chmod(RUNNER_SH, 0o755)
    sftp.close()
    print(f"    ✓ {RUNNER_SH}")
    print(f"    实验数: {len(EXPERIMENTS)}")
    for exp_name, config, model, seed, batch in EXPERIMENTS:
        print(f"      - {exp_name} (model={model}, seed={seed}, batch={batch})")

    # [4] 启动 (前台流式, 不限时)
    print(f"\n[4] 启动全部实验 (预计 1.5-2 小时) ...")
    print("    " + "=" * 60)
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.exec_command(f"bash {RUNNER_SH} 2>&1")

    out_buf = []
    last_print = time.time()
    start = time.time()
    last_heartbeat = 0
    while True:
        if chan.recv_ready():
            data = chan.recv(8192).decode("utf-8", errors="replace")
            out_buf.append(data)
            for line in data.splitlines():
                print("    " + line.rstrip())
            last_print = time.time()
        elif chan.exit_status_ready():
            while chan.recv_ready():
                data = chan.recv(8192).decode("utf-8", errors="replace")
                out_buf.append(data)
                for line in data.splitlines():
                    print("    " + line.rstrip())
            break
        else:
            time.sleep(1)
            elapsed = int(time.time() - start)
            if elapsed - last_heartbeat >= 180:
                print(f"    [心跳 {elapsed//60}min] 还在跑...")
                last_heartbeat = elapsed
    print("    " + "=" * 60)
    total_min = int(time.time() - start) // 60
    print(f"\n[5] 全部实验完成, 总耗时 {total_min} 分钟")

    out = "".join(out_buf)
    if "ALL_DONE" in out:
        print("\n🎉🎉🎉 全部实验完成!")
        return 0
    else:
        print("\n⚠ 可能未全部完成")
        return 1


if __name__ == "__main__":
    sys.exit(main())
