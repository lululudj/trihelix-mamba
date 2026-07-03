"""扩展 seed 实验: 300M 多 seed + 100M/30M 补 seed + HTA 多 seed
修复了 eval_ood 参数名 (--checkpoint 不是 --ckpt)。
"""
import sys
import time
import json
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"
RUNNER_SH = "/root/run_all_seeds.sh"

# (exp_name, config, model, seed, batch, max_steps)
EXPERIMENTS = [
    # 300M 多 seed (验证大模型零衰减的统计可靠性)
    ("300m_seed1_2k",     "matched_mamba2_300m.yaml", "three_chain_mamba2",  1, 1, 2000),
    ("300m_seed2_2k",     "matched_mamba2_300m.yaml", "three_chain_mamba2",  2, 1, 2000),
    # 100M HTA 多 seed (HTA 在 100M 规模的多 seed)
    ("100m_hta_seed1_500","matched_mamba2_100m.yaml", "three_chain_mamba2_hta", 1, 2, 500),
    ("100m_hta_seed2_500","matched_mamba2_100m.yaml", "three_chain_mamba2_hta", 2, 2, 500),
    # 100M 补 seed3/4 (扩展统计)
    ("100m_seed3_500",    "matched_mamba2_100m.yaml", "three_chain_mamba2",  3, 2, 500),
    ("100m_seed4_500",    "matched_mamba2_100m.yaml", "three_chain_mamba2",  4, 2, 500),
    # 30M 补 seed3/4 (小模型统计)
    ("30m_seed3",         "matched_mamba2_30m.yaml",  "three_chain_mamba2",  3, 2, 100),
    ("30m_seed4",         "matched_mamba2_30m.yaml",  "three_chain_mamba2",  4, 2, 100),
]


def build_runner_script():
    calls = []
    for exp_name, config, model, seed, batch, max_steps in EXPERIMENTS:
        calls.append(f'run_experiment "{exp_name}" "{config}" "{model}" {seed} {batch} {max_steps}')
    calls_str = "\n".join(calls)

    return f"""#!/bin/bash
# 多 seed 扩展实验
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/three_chain_v3

echo "=========================================="
echo "  多 seed 扩展实验 ({len(EXPERIMENTS)} 实验)"
echo "  开始: $(date)"
echo "=========================================="

run_experiment() {{
    local exp_name=$1 config=$2 model=$3 seed=$4 batch=$5 max_steps=$6
    local exp_dir="results/run_${{exp_name}}"
    echo ""
    echo "=========================================="
    echo "  $exp_name (model=$model seed=$seed batch=$batch steps=$max_steps)"
    echo "  $(date)"
    echo "=========================================="
    mkdir -p "$exp_dir"

    # 训练
    echo "[训练] $max_steps 步..."
    python -u train.py --model $model --config configs/$config \\
        --max_steps $max_steps --batch_size $batch --seed $seed \\
        --data_root ./data_split_m3 --out_dir "$exp_dir" 2>&1 | tail -8
    local rc=${{PIPESTATUS[0]}}
    echo "[训练] rc=$rc"
    if [ $rc -ne 0 ]; then
        echo "$exp_name: TRAIN_FAILED" >> /root/all_seeds_summary.txt
        return
    fi

    # eval_ood (正确参数!)
    echo "[OOD eval]..."
    python -u eval_ood.py --checkpoint "$exp_dir/final.pt" \\
        --config configs/$config --data_root ./data/ood_T150 \\
        --batch_size 4 --seed $seed --out "$exp_dir/ood_metrics.json" 2>&1 | tail -8
    echo "[OOD] rc=${{PIPESTATUS[0]}}"

    # 提取关键指标
    if [ -f "$exp_dir/ood_metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('$exp_dir/ood_metrics.json'))
curve = m.get('changed_acc_curve', {{}})
ch100 = curve.get('100', 0)
ch150 = curve.get('150', 0)
decay = m.get('ood_decay_pct', 0)
zr = m.get('zero_ratio', 0)
print(f'  $exp_name: decay={{decay*100:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}, zr={{zr:.3f}}')
with open('/root/all_seeds_summary.txt', 'a') as f:
    f.write(f'$exp_name: decay={{decay*100:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}\\n')
"
    else
        echo "$exp_name: EVAL_FAILED" >> /root/all_seeds_summary.txt
    fi
    echo "[完成] $(date)"
}}

rm -f /root/all_seeds_summary.txt
touch /root/all_seeds_summary.txt

{calls_str}

echo ""
echo "=========================================="
echo "  全部完成: $(date)"
echo "=========================================="
echo "=== 汇总 ==="
cat /root/all_seeds_summary.txt
echo "=== ALL_SEEDS_DONE ==="
"""


def run_streaming(cli, cmd, max_wait=3600):
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
            if time.time() - last_print > 60:
                print(f"    [心跳 {int(time.time()-start)}s] ...")
                last_print = time.time()
            if time.time() - start > max_wait:
                print(f"    ✗ 超时 {max_wait}s")
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    print("\n[2] 上传脚本 ...")
    sftp = cli.open_sftp()
    with sftp.open(RUNNER_SH, "w") as f:
        f.write(build_runner_script())
    sftp.chmod(RUNNER_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {RUNNER_SH}")

    print(f"\n[3] 跑 {len(EXPERIMENTS)} 个 seed 扩展实验 (约 30-40 分钟) ...")
    out, rc = run_streaming(cli, f"bash {RUNNER_SH}", max_wait=3600)

    print(f"\n[4] rc={rc}")
    if "ALL_SEEDS_DONE" in out:
        print("\n✓✓✓ 全部完成!")
        idx = out.rfind("=== 汇总 ===")
        if idx >= 0:
            print("\n最终汇总:")
            print(out[idx:])
    else:
        print("\n✗ 未完成, 看日志")
    return 0


if __name__ == "__main__":
    sys.exit(main())
