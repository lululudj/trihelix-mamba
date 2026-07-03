"""读 300M 2k 步的 OOD 结果"""
import paramiko
import json

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"


def run(cli, cmd):
    _, stdout, _ = cli.exec_command(cmd)
    return stdout.read().decode("utf-8", errors="replace")


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

    # 300M 2k 的 ood_metrics.json
    print("=== 300m_seed0_2k OOD 结果 ===")
    out = run(cli, "cat /root/three_chain_v3/results/run_300m_seed0_2k/ood_metrics.json 2>&1")
    if "No such file" in out or not out.strip():
        print("(还未生成, eval 还在跑)")
    else:
        try:
            m = json.loads(out)
            print(f"zero_ratio: {m.get('zero_ratio', 'N/A')}")
            print(f"ood_decay_pct: {m.get('ood_decay_pct', 'N/A')}")
            ca = m.get('changed_acc', {})
            print("changed_acc 各时间点:")
            for k in sorted(ca.keys()):
                print(f"  {k}: {ca[k]:.4f}")
            # 算 decay
            ch100 = ca.get('t=100')
            ch150 = ca.get('t=150')
            if ch100 is not None and ch150 is not None:
                decay = (ch150 - ch100) / ch100 * 100 if ch100 > 0 else 0
                print(f"\n>>> decay (ch150-ch100)/ch100 = {decay:+.1f}%")
        except Exception as e:
            print(f"解析失败: {e}")
            print(out[:500])

    # summary.json
    print("\n=== 300m_seed0_2k summary ===")
    print(run(cli, "cat /root/three_chain_v3/results/run_300m_seed0_2k/summary.json 2>&1"))

    # 看进程
    print("\n=== 当前进程 ===")
    print(run(cli, "ps aux | grep -E 'train.py|eval_ood' | grep -v grep | head -3"))

    # GPU
    print("\n=== GPU ===")
    print(run(cli, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>&1"))


if __name__ == "__main__":
    main()
