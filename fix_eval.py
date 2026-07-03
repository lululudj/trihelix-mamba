"""修复: 对 3 个补跑实验重新跑 eval_ood (之前 --ckpt 参数名错)
等当前 run_extended.sh 跑完后, 对已有 final.pt 重新 eval。
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

# (exp_name, config, seed)
EXPERIMENTS = [
    ("300m_seed0_2k",     "matched_mamba2_300m.yaml", 0),
    ("100m_seed0_500",    "matched_mamba2_100m.yaml", 0),
    ("100m_hta_seed0_500","matched_mamba2_100m.yaml", 0),
]


def run(cli, cmd):
    _, stdout, _ = cli.exec_command(cmd)
    return stdout.read().decode("utf-8", errors="replace")


def run_streaming(cli, cmd, max_wait=600):
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
                print(f"    ✗ 超时 {max_wait}s")
                break
    return "".join(out_buf), chan.recv_exit_status()


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    # 等待 run_extended.sh 完成 (最多等 30 分钟)
    print("\n[2] 等待 run_extended.sh 完成 ...")
    waited = 0
    while waited < 1800:
        ps = run(cli, "ps aux | grep -E 'run_extended.sh|train.py' | grep -v grep | head -3")
        if not ps.strip():
            print(f"    ✓ run_extended.sh 已结束 (等待了 {waited}s)")
            break
        if waited % 60 == 0:
            print(f"    [等待 {waited}s] 还有进程:\n{ps[:200]}")
        time.sleep(15)
        waited += 15
    else:
        print(f"    ✗ 等了 30 分钟还没结束, 强制继续")

    # 对每个实验重新跑 eval_ood
    print("\n[3] 重新跑 eval_ood ...")
    results = []
    for exp_name, config, seed in EXPERIMENTS:
        exp_dir = f"{WORKDIR}/results/run_{exp_name}"
        print(f"\n--- {exp_name} ---")
        # 确认 final.pt 存在
        check = run(cli, f"ls -la {exp_dir}/final.pt 2>&1")
        if "No such file" in check:
            print(f"    ✗ final.pt 不存在, 跳过")
            continue
        print(f"    final.pt 存在")

        # 跑 eval_ood (正确参数)
        cmd = f"""cd {WORKDIR} && \\
source /root/miniconda3/etc/profile.d/conda.sh && \\
conda activate base && \\
export CUDA_HOME=/usr/local/cuda && \\
export PATH=$CUDA_HOME/bin:$PATH && \\
export PYTHONUNBUFFERED=1 && \\
python -u eval_ood.py \\
    --checkpoint {exp_dir}/final.pt \\
    --config configs/{config} \\
    --data_root ./data/ood_T150 \\
    --batch_size 4 \\
    --seed {seed} \\
    --out {exp_dir}/ood_metrics.json"""
        out, rc = run_streaming(cli, cmd, max_wait=600)
        print(f"    rc={rc}")

        # 读结果
        result_out = run(cli, f"cat {exp_dir}/ood_metrics.json 2>&1")
        try:
            m = json.loads(result_out)
            zr = m.get('zero_ratio', None)
            decay = m.get('ood_decay_pct', None)
            curve = m.get('changed_acc_curve', {})
            ch100 = curve.get('100', None)
            ch150 = curve.get('150', None)
            print(f"    zero_ratio={zr:.4f}, decay={decay*100:+.1f}%, ch@100={ch100:.4f}, ch@150={ch150:.4f}")
            results.append((exp_name, decay, ch100, ch150, zr))
        except Exception as e:
            print(f"    解析失败: {e}")
            print(f"    原始: {result_out[:300]}")

    # 汇总
    print("\n==========================================")
    print("  补跑实验 OOD 结果汇总")
    print("==========================================")
    for exp_name, decay, ch100, ch150, zr in results:
        print(f"  {exp_name}: decay={decay*100:+.1f}%, ch@100={ch100:.4f}, ch@150={ch150:.4f}, zero_ratio={zr:.3f}")

    # 写到 extended_summary.txt
    summary_text = "\n".join(
        f"{n}: decay={d*100:+.1f}%, ch@100={c1:.4f}, ch@150={c2:.4f}"
        for n, d, c1, c2, _ in results
    )
    run(cli, f"cat > /root/extended_summary.txt << 'EOF'\n{summary_text}\nEOF")
    print(f"\n汇总已写入 /root/extended_summary.txt")
    print("\n=== FIX_EVAL_DONE ===")


if __name__ == "__main__":
    main()
