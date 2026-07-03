"""检查 final_summary.txt 和当前 700M 进度"""
import paramiko

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

    print("=== final_summary.txt (补跑+700M/1000M) ===")
    print(run(cli, "cat /root/final_summary.txt 2>&1"))

    print("\n=== 700M 训练进度 (log.jsonl) ===")
    print(run(cli, "tail -5 /root/three_chain_v3/results/run_700m_seed0_2k/log.jsonl 2>&1"))

    print("\n=== 当前进程 ===")
    print(run(cli, "ps aux | grep -E 'train.py|eval_ood|run_final' | grep -v grep | head -3"))

    print("\n=== GPU ===")
    print(run(cli, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>&1"))

    print("\n=== 磁盘 ===")
    print(run(cli, "df -h /root | tail -2"))


if __name__ == "__main__":
    main()
