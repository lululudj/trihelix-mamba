"""综合诊断: 看云端当前真实状态"""
import paramiko, sys, time

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

def run(client, cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err

def main():
    print("[1] 连接 ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)
    except Exception as e:
        print(f"  连接失败: {e}")
        sys.exit(1)
    print("    ✓ 连上了")

    print("\n[2] GPU 状态 ...")
    out, _ = run(client, "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv")
    print(out.strip())

    print("\n[3] 训练进程 ...")
    out, _ = run(client, "ps aux | grep -E 'python|train' | grep -v grep")
    print(out.strip() if out.strip() else "(无 python/train 进程)")

    print("\n[4] run_700m_1000m.sh 进程 ...")
    out, _ = run(client, "ps aux | grep -E 'run_700m|run_final|run_all_seeds|run_rerun' | grep -v grep")
    print(out.strip() if out.strip() else "(无)")

    print("\n[5] 磁盘 ...")
    out, _ = run(client, "df -h / /root 2>/dev/null | head -5")
    print(out.strip())

    print("\n[6] results 目录 ...")
    out, _ = run(client, "ls -la /root/three_chain_v3/results/ 2>/dev/null")
    print(out.strip())

    print("\n[7] 各实验 ood_metrics.json 状态 ...")
    cmd = "cd /root/three_chain_v3/results && for d in run_*; do " \
          "if [ -f \"$d/ood_metrics.json\" ]; then " \
          "decay=$(python3 -c \"import json; d=json.load(open('$d/ood_metrics.json')); print(f'ch@100={d.get(\\\"changed_acc_curve\\\",{}).get(\\\"100\\\")}, ch@150={d.get(\\\"changed_acc_curve\\\",{}).get(\\\"150\\\")}')\" 2>/dev/null); " \
          "echo \"$d: $decay\"; " \
          "elif [ -f \"$d/summary.json\" ]; then echo \"$d: NO_OOD\"; " \
          "else echo \"$d: INCOMPLETE\"; fi; done"
    out, _ = run(client, cmd, timeout=60)
    print(out.strip())

    print("\n[8] 700M 训练日志最后 20 行 ...")
    out, _ = run(client, "tail -20 /root/three_chain_v3/results/run_700m_seed0_2k/log.jsonl 2>/dev/null || echo '(无日志)'")
    print(out.strip())

    print("\n[9] 700M final.pt 是否存在 ...")
    out, _ = run(client, "ls -lh /root/three_chain_v3/results/run_700m_seed0_2k/*.pt 2>/dev/null || echo '(无 .pt)'")
    print(out.strip())

    print("\n[10] 1000M 目录 ...")
    out, _ = run(client, "ls -la /root/three_chain_v3/results/run_1000m_seed0_2k/ 2>/dev/null || echo '(无 1000M 目录)'")
    print(out.strip())

    client.close()
    print("\n✓ 诊断完成")

if __name__ == "__main__":
    main()
