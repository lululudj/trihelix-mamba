"""紧急清理: 删除所有已完成实验的 best.pt 释放磁盘 (保留 final.pt)
磁盘只剩 1.3G, 700M 训练可能保存失败
"""
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"


def run(cli, cmd):
    _, stdout, _ = cli.exec_command(cmd)
    return stdout.read().decode("utf-8", errors="replace")


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("✓ 连上\n")

    print("=== 清理前磁盘 ===")
    print(run(cli, "df -h /root | tail -2"))

    # 删除所有 results 目录下的 best.pt (除了正在训练的 700M/1000M)
    print("\n=== 删除已完成实验的 best.pt ===")
    out = run(cli, f"find {WORKDIR}/results -name 'best.pt' -not -path '*700m*' -not -path '*1000m*' 2>/dev/null")
    files = [f.strip() for f in out.splitlines() if f.strip()]
    print(f"找到 {len(files)} 个 best.pt:")
    total_freed = 0
    for f in files:
        size_out = run(cli, f"stat -c %s '{f}' 2>/dev/null")
        size = int(size_out.strip()) if size_out.strip().isdigit() else 0
        if size > 0:
            print(f"  删 {f} ({size//1024//1024}MB)")
            total_freed += size
        run(cli, f"rm -f '{f}'")

    print(f"\n总计释放: {total_freed//1024//1024}MB")

    print("\n=== 清理后磁盘 ===")
    print(run(cli, "df -h /root | tail -2"))

    # 确认 700M 还在训练
    print("\n=== 当前进程 ===")
    print(run(cli, "ps aux | grep -E 'train.py|run_final|run_700m' | grep -v grep | head -5"))


if __name__ == "__main__":
    main()
