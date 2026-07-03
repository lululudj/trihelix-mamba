"""下载所有实验结果到本地"""
import os
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"
LOCAL_RESULTS = "e:/three_chain_v3/results_cloud"

# 要下载的实验列表
EXPERIMENTS = [
    "run_100m_seed0", "run_100m_seed1", "run_100m_seed2",
    "run_30m_seed1", "run_30m_seed2",
    "run_300m_seed0", "run_100m_hta_seed0",
    "run_three_chain_mamba2_30m_seed0",  # 第一次 30M 实验
]
# 每个实验要下载的文件 (先只下 JSON, .pt 大文件后续统一打包)
FILES = ["ood_metrics.json", "summary.json", "log.jsonl"]


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    sftp = cli.open_sftp()
    os.makedirs(LOCAL_RESULTS, exist_ok=True)

    print(f"\n[2] 下载结果到 {LOCAL_RESULTS}/ ...")
    total_files = 0
    total_bytes = 0
    for exp in EXPERIMENTS:
        remote_dir = f"{WORKDIR}/results/{exp}"
        local_dir = os.path.join(LOCAL_RESULTS, exp)
        os.makedirs(local_dir, exist_ok=True)

        # 检查远程目录是否存在
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            print(f"  ⚠ {exp}: 远程目录不存在, 跳过")
            continue

        # 下载每个文件
        for fname in FILES:
            remote_path = f"{remote_dir}/{fname}"
            local_path = os.path.join(local_dir, fname)
            try:
                remote_stat = sftp.stat(remote_path)
                size = remote_stat.st_size
                print(f"  {exp}/{fname} ({size//1024}KB)... ", end="", flush=True)
                sftp.get(remote_path, local_path)
                print("✓")
                total_files += 1
                total_bytes += size
            except FileNotFoundError:
                print(f"  {exp}/{fname}: 不存在, 跳过")
            except Exception as e:
                print(f"  {exp}/{fname}: 错误 {e}")

    # 下载汇总文件
    print("\n[3] 下载汇总文件 ...")
    for summary_file in ["/root/experiments_summary.txt"]:
        try:
            local_path = os.path.join(LOCAL_RESULTS, os.path.basename(summary_file))
            sftp.get(summary_file, local_path)
            print(f"  ✓ {summary_file}")
            # 打印内容
            with open(local_path, "r") as f:
                print(f"  内容:")
                print(f.read())
        except Exception as e:
            print(f"  ⚠ {summary_file}: {e}")

    sftp.close()
    cli.close()

    print(f"\n[4] 完成!")
    print(f"  下载文件数: {total_files}")
    print(f"  总大小: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  本地路径: {LOCAL_RESULTS}/")


if __name__ == "__main__":
    main()
