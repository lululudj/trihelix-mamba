"""等当前 seed 实验完成后, 跑 700M + 1000M 极限实验
700M (d_model=4096, ~728M params) 应该能跑
1000M (d_model=5120, ~1.1B params) 可能 OOM, 失败也可接受
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
RUNNER_SH = "/root/run_700m_1000m.sh"

CONFIGS = {
    "matched_mamba2_700m.yaml": "e:/three_chain_v3/configs/matched_mamba2_700m.yaml",
    "matched_mamba2_1000m.yaml": "e:/three_chain_v3/configs/matched_mamba2_1000m.yaml",
}

# (exp_name, config, seed, batch, max_steps)
EXPERIMENTS = [
    ("700m_seed0_2k",  "matched_mamba2_700m.yaml",  0, 1, 2000),
    ("1000m_seed0_2k", "matched_mamba2_1000m.yaml", 0, 1, 2000),
]


def build_runner_script():
    calls = []
    for exp_name, config, seed, batch, max_steps in EXPERIMENTS:
        calls.append(f'run_experiment "{exp_name}" "{config}" {seed} {batch} {max_steps}')
    calls_str = "\n".join(calls)

    return f"""#!/bin/bash
# 700M + 1000M 极限实验
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/three_chain_v3

echo "=========================================="
echo "  700M + 1000M 极限实验"
echo "  开始: $(date)"
echo "=========================================="

run_experiment() {{
    local exp_name=$1 config=$2 seed=$3 batch=$4 max_steps=$5
    local exp_dir="results/run_${{exp_name}}"
    echo ""
    echo "=========================================="
    echo "  $exp_name (seed=$seed batch=$batch steps=$max_steps)"
    echo "  $(date)"
    echo "=========================================="
    mkdir -p "$exp_dir"

    # 训练 (OOM 不中断)
    echo "[训练] $max_steps 步..."
    python -u train.py --model three_chain_mamba2 --config configs/$config \\
        --max_steps $max_steps --batch_size $batch --seed $seed \\
        --data_root ./data_split_m3 --out_dir "$exp_dir" 2>&1 | tail -15
    local rc=${{PIPESTATUS[0]}}
    echo "[训练] rc=$rc"
    if [ $rc -ne 0 ]; then
        echo "$exp_name: TRAIN_FAILED (可能 OOM)" >> /root/big_scale_summary.txt
        return
    fi

    # eval_ood
    echo "[OOD eval]..."
    python -u eval_ood.py --checkpoint "$exp_dir/final.pt" \\
        --config configs/$config --data_root ./data/ood_T150 \\
        --batch_size 4 --seed $seed --out "$exp_dir/ood_metrics.json" 2>&1 | tail -10
    echo "[OOD] rc=${{PIPESTATUS[0]}}"

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
with open('/root/big_scale_summary.txt', 'a') as f:
    f.write(f'$exp_name: decay={{decay*100:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}\\n')
"
    else
        echo "$exp_name: EVAL_FAILED" >> /root/big_scale_summary.txt
    fi
    echo "[完成] $(date)"
}}

rm -f /root/big_scale_summary.txt
touch /root/big_scale_summary.txt

{calls_str}

echo ""
echo "=========================================="
echo "  极限实验完成: $(date)"
echo "=========================================="
echo "=== 汇总 ==="
cat /root/big_scale_summary.txt
echo "=== BIG_SCALE_DONE ==="
"""


def run(cli, cmd):
    _, stdout, _ = cli.exec_command(cmd)
    return stdout.read().decode("utf-8", errors="replace")


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

    print("\n[2] 上传 700M/1000M 配置 ...")
    sftp = cli.open_sftp()
    for name, local_path in CONFIGS.items():
        sftp.put(local_path, f"{WORKDIR}/configs/{name}")
        print(f"    ✓ {name}")
    # 上传 runner
    with sftp.open(RUNNER_SH, "w") as f:
        f.write(build_runner_script())
    sftp.chmod(RUNNER_SH, 0o755)
    sftp.close()
    print(f"    ✓ 写入 {RUNNER_SH}")

    # 等待当前 seed 实验完成 (最多等 50 分钟)
    print("\n[3] 等待 run_all_seeds.sh 完成 ...")
    waited = 0
    while waited < 3000:
        ps = run(cli, "ps aux | grep -E 'run_all_seeds.sh|train.py' | grep -v grep | head -3")
        if not ps.strip():
            print(f"    ✓ run_all_seeds.sh 已结束 (等待了 {waited}s)")
            break
        if waited % 120 == 0:
            # 看下当前进度
            summary = run(cli, "cat /root/all_seeds_summary.txt 2>&1")
            print(f"    [等待 {waited}s] 已完成:\n{summary[:300]}")
        time.sleep(20)
        waited += 20
    else:
        print(f"    ⚠ 等了 50 分钟, 强制继续")

    # 跑 700M + 1000M
    print("\n[4] 跑 700M + 1000M 极限实验 (700M ~12分钟, 1000M 可能OOM) ...")
    out, rc = run_streaming(cli, f"bash {RUNNER_SH}", max_wait=3000)

    print(f"\n[5] rc={rc}")
    if "BIG_SCALE_DONE" in out:
        print("\n✓✓✓ 极限实验完成!")
        idx = out.rfind("=== 汇总 ===")
        if idx >= 0:
            print("\n汇总:")
            print(out[idx:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
