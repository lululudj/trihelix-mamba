"""下载 seed 扩展实验的 JSON"""
import os
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"
LOCAL = "e:/three_chain_v3/results_cloud"

NEW_EXPS = [
    "run_300m_seed1_2k",
    "run_100m_hta_seed2_500",
    "run_100m_seed3_500",
    "run_100m_seed4_500",
    "run_30m_seed3",
    "run_30m_seed4",
]
FILES = ["ood_metrics.json", "summary.json", "log.jsonl"]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("✓ 连上\n")

    sftp = cli.open_sftp()
    for exp in NEW_EXPS:
        remote_dir = f"{WORKDIR}/results/{exp}"
        local_dir = os.path.join(LOCAL, exp)
        os.makedirs(local_dir, exist_ok=True)
        print(f"--- {exp} ---")
        for fname in FILES:
            try:
                size = sftp.stat(f"{remote_dir}/{fname}").st_size
                sftp.get(f"{remote_dir}/{fname}", os.path.join(local_dir, fname))
                print(f"  ✓ {fname} ({size}B)")
            except Exception as e:
                print(f"  ⚠ {fname}: {e}")

    # 也下 all_seeds_summary.txt
    try:
        sftp.get("/root/all_seeds_summary.txt", os.path.join(LOCAL, "all_seeds_summary.txt"))
        print("\n✓ all_seeds_summary.txt:")
        with open(os.path.join(LOCAL, "all_seeds_summary.txt")) as f:
            print(f.read())
    except Exception as e:
        print(f"⚠ {e}")

    sftp.close()
    cli.close()
    print(f"\n✓ 下载完成到 {LOCAL}/")


if __name__ == "__main__":
    main()
