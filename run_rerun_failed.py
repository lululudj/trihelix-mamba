"""只补跑失败实验 (300M seed2 + 100M HTA seed1)
等 700M/1000M 完成后跑这个, 不重跑 700M/1000M
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
RUNNER_SH = "/root/run_rerun_failed.sh"

EXPERIMENTS = [
    ("300m_seed2_2k",      "matched_mamba2_300m.yaml", "three_chain_mamba2",     2, 1, 2000),
    ("100m_hta_seed1_500", "matched_mamba2_100m.yaml", "three_chain_mamba2_hta", 1, 2, 500),
]


def build_runner_script():
    calls = []
    for exp_name, config, model, seed, batch, max_steps in EXPERIMENTS:
        calls.append(f'run_experiment "{exp_name}" "{config}" "{model}" {seed} {batch} {max_steps}')
    calls_str = "\n".join(calls)
    return f"""#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/three_chain_v3
echo "=== 补跑失败实验 ==="
run_experiment() {{
    local exp_name=$1 config=$2 model=$3 seed=$4 batch=$5 max_steps=$6
    local exp_dir="results/run_${{exp_name}}"
    echo "  $exp_name (seed=$seed steps=$max_steps) $(date)"
    mkdir -p "$exp_dir"
    python -u train.py --model $model --config configs/$config \\
        --max_steps $max_steps --batch_size $batch --seed $seed \\
        --data_root ./data_split_m3 --out_dir "$exp_dir" 2>&1 | tail -8
    local rc=${{PIPESTATUS[0]}}
    if [ $rc -ne 0 ]; then echo "$exp_name: TRAIN_FAILED" >> /root/rerun_summary.txt; return; fi
    rm -f "$exp_dir/best.pt"
    python -u eval_ood.py --checkpoint "$exp_dir/final.pt" \\
        --config configs/$config --data_root ./data/ood_T150 \\
        --batch_size 4 --seed $seed --out "$exp_dir/ood_metrics.json" 2>&1 | tail -8
    if [ -f "$exp_dir/ood_metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('$exp_dir/ood_metrics.json'))
curve = m.get('changed_acc_curve', {{}})
ch100 = curve.get('100', 0)
ch150 = curve.get('150', 0)
decay = m.get('ood_decay_pct', 0)
print(f'  $exp_name: decay={{decay*100:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}')
with open('/root/rerun_summary.txt', 'a') as f:
    f.write(f'$exp_name: decay={{decay*100:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}\\n')
"
    fi
}}
rm -f /root/rerun_summary.txt
touch /root/rerun_summary.txt
{calls_str}
echo "=== 汇总 ==="
cat /root/rerun_summary.txt
echo "=== RERUN_DONE ==="
"""


def run(cli, cmd):
    _, stdout, _ = cli.exec_command(cmd)
    return stdout.read().decode("utf-8", errors="replace")


def run_streaming(cli, cmd, max_wait=1800):
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
                print(f"    [心跳 {int(time.time()-start)}s]")
                last_print = time.time()
            if time.time() - start > max_wait:
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("✓ 连上")

    sftp = cli.open_sftp()
    with sftp.open(RUNNER_SH, "w") as f:
        f.write(build_runner_script())
    sftp.chmod(RUNNER_SH, 0o755)
    sftp.close()

    # 等 train.py 结束 (700M/1000M 完成)
    print("\n等待 700M/1000M 完成 ...")
    waited = 0
    while waited < 3000:
        ps = run(cli, "ps aux | grep train.py | grep -v grep | head -3")
        if not ps.strip():
            print(f"✓ 训练都结束了 (等了 {waited}s)")
            break
        if waited % 120 == 0:
            print(f"[等待 {waited}s] {ps[:150]}")
        time.sleep(20)
        waited += 20

    print("\n跑补跑实验 ...")
    out, rc = run_streaming(cli, f"bash {RUNNER_SH}", max_wait=1800)
    if "RERUN_DONE" in out:
        print("\n✓✓✓ 补跑完成!")
        idx = out.rfind("=== 汇总 ===")
        if idx >= 0:
            print(out[idx:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
