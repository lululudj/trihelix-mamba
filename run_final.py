"""最终合并脚本: 等 seed 完成 → 补跑失败实验 → 700M/1000M
处理磁盘空间问题: 每个实验后删 best.pt (只留 final.pt)
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
RUNNER_SH = "/root/run_final.sh"

# (exp_name, config, model, seed, batch, max_steps)
EXPERIMENTS = [
    # 补跑失败实验 (磁盘满导致)
    ("300m_seed2_2k",      "matched_mamba2_300m.yaml",  "three_chain_mamba2",     2, 1, 2000),
    ("100m_hta_seed1_500", "matched_mamba2_100m.yaml",  "three_chain_mamba2_hta",  1, 2, 500),
    # 极限实验
    ("700m_seed0_2k",      "matched_mamba2_700m.yaml",  "three_chain_mamba2",     0, 1, 2000),
    ("1000m_seed0_2k",     "matched_mamba2_1000m.yaml", "three_chain_mamba2",     0, 1, 2000),
]


def build_runner_script():
    calls = []
    for exp_name, config, model, seed, batch, max_steps in EXPERIMENTS:
        calls.append(f'run_experiment "{exp_name}" "{config}" "{model}" {seed} {batch} {max_steps}')
    calls_str = "\n".join(calls)

    return f"""#!/bin/bash
# 最终实验: 补跑失败 + 700M/1000M
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/three_chain_v3

echo "=========================================="
echo "  最终实验 ({len(EXPERIMENTS)} 实验)"
echo "  开始: $(date)"
echo "=========================================="

run_experiment() {{
    local exp_name=$1 config=$2 model=$3 seed=$4 batch=$5 max_steps=$6
    local exp_dir="results/run_${{exp_name}}"
    echo ""
    echo "=========================================="
    echo "  $exp_name (model=$model seed=$seed steps=$max_steps)"
    echo "  $(date) | 磁盘: $(df -h /root | tail -1 | awk '{{print $4}}') free"
    echo "=========================================="
    mkdir -p "$exp_dir"

    # 训练
    echo "[训练] $max_steps 步..."
    python -u train.py --model $model --config configs/$config \\
        --max_steps $max_steps --batch_size $batch --seed $seed \\
        --data_root ./data_split_m3 --out_dir "$exp_dir" 2>&1 | tail -12
    local rc=${{PIPESTATUS[0]}}
    echo "[训练] rc=$rc"
    if [ $rc -ne 0 ]; then
        echo "$exp_name: TRAIN_FAILED (可能 OOM/磁盘)" >> /root/final_summary.txt
        return
    fi

    # 清理 best.pt 省空间 (final.pt 就够 eval)
    rm -f "$exp_dir/best.pt"
    echo "  [清理 best.pt] 磁盘: $(df -h /root | tail -1 | awk '{{print $4}}') free"

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
with open('/root/final_summary.txt', 'a') as f:
    f.write(f'$exp_name: decay={{decay*100:+.1f}}%, ch@100={{ch100:.4f}}, ch@150={{ch150:.4f}}\\n')
"
    else
        echo "$exp_name: EVAL_FAILED" >> /root/final_summary.txt
    fi
    echo "[完成] $(date)"
}}

rm -f /root/final_summary.txt
touch /root/final_summary.txt

{calls_str}

echo ""
echo "=========================================="
echo "  全部完成: $(date)"
echo "=========================================="
echo "=== 最终汇总 ==="
cat /root/final_summary.txt
echo ""
echo "=== 全部 summary 合并 ==="
cat /root/all_seeds_summary.txt /root/final_summary.txt 2>/dev/null
echo "=== FINAL_DONE ==="
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

    print("\n[2] 上传配置 (700M/1000M 已上传过, 再传一次保险) ...")
    sftp = cli.open_sftp()
    for name, local in [("matched_mamba2_700m.yaml", "e:/three_chain_v3/configs/matched_mamba2_700m.yaml"),
                         ("matched_mamba2_1000m.yaml", "e:/three_chain_v3/configs/matched_mamba2_1000m.yaml")]:
        sftp.put(local, f"{WORKDIR}/configs/{name}")
        print(f"    ✓ {name}")
    with sftp.open(RUNNER_SH, "w") as f:
        f.write(build_runner_script())
    sftp.chmod(RUNNER_SH, 0o755)
    sftp.close()

    # 杀掉可能残留的 run_700m_1000m.sh
    print("\n[3] 清理可能残留的 run_700m_1000m.sh 进程 ...")
    run(cli, "pkill -f run_700m_1000m.sh 2>/dev/null; echo done")

    # 等 run_all_seeds.sh 完成
    print("\n[4] 等待 run_all_seeds.sh 完成 ...")
    waited = 0
    while waited < 3000:
        ps = run(cli, "ps aux | grep -E 'run_all_seeds.sh|train.py' | grep -v grep | head -3")
        if not ps.strip():
            print(f"    ✓ run_all_seeds.sh 已结束 (等待 {waited}s)")
            break
        if waited % 120 == 0:
            summary = run(cli, "cat /root/all_seeds_summary.txt 2>&1")
            print(f"    [等待 {waited}s] 进度:\n{summary[:400]}")
        time.sleep(20)
        waited += 20

    # 跑最终实验
    print("\n[5] 跑最终实验 (补跑 + 700M/1000M, 约 25-40 分钟) ...")
    out, rc = run_streaming(cli, f"bash {RUNNER_SH}", max_wait=3600)

    print(f"\n[6] rc={rc}")
    if "FINAL_DONE" in out:
        print("\n✓✓✓ 全部完成!")
        idx = out.rfind("=== 最终汇总 ===")
        if idx >= 0:
            print("\n" + out[idx:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
