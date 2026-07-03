"""清理被替换的旧实验 .pt 文件释放磁盘空间
这些实验已被加长版本替代, .pt 可删:
- run_300m_seed0 (100步, 被 run_300m_seed0_2k 替代)
- run_100m_seed0 (100步, 被 run_100m_seed0_500 替代)
- run_100m_hta_seed0 (100步, 被 run_100m_hta_seed0_500 替代)
保留 JSON 不删 (核心数据已下载本地)
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

    # 1. 看磁盘
    print("=== 清理前磁盘 ===")
    print(run(cli, "df -h /root | tail -2"))

    # 2. 要删的旧实验 .pt (被加长版本替代)
    to_delete = [
        "run_300m_seed0/final.pt",       # 被 run_300m_seed0_2k 替代
        "run_300m_seed0/best.pt",
        "run_100m_seed0/final.pt",       # 被 run_100m_seed0_500 替代
        "run_100m_seed0/best.pt",
        "run_100m_hta_seed0/final.pt",   # 被 run_100m_hta_seed0_500 替代
        "run_100m_hta_seed0/best.pt",
    ]
    print("\n=== 删除被替代的旧 .pt ===")
    for f in to_delete:
        path = f"{WORKDIR}/results/{f}"
        # 先看大小
        size_out = run(cli, f"ls -la {path} 2>&1 | awk '{{print $5}}'")
        size = size_out.strip()
        if size and size.isdigit():
            print(f"  删 {f} ({int(size)//1024//1024}MB)")
        run(cli, f"rm -f {path}")

    # 3. 也删 300M seed1 和 seed2 的 best.pt (只留 final.pt, best 和 final 几乎一样)
    # 不,先不删,万一下次实验还要用

    # 4. 看清理后磁盘
    print("\n=== 清理后磁盘 ===")
    print(run(cli, "df -h /root | tail -2"))

    # 5. 看当前实验状态
    print("\n=== 当前训练进程 ===")
    print(run(cli, "ps aux | grep train.py | grep -v grep | head -3"))

    print("\n✓ 清理完成")


if __name__ == "__main__":
    main()
