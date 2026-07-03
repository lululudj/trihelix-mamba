"""
monitor_final2.py - 监控 final_rerun.sh, 完成后自动下载所有结果
每 60s 检查一次, 最长等 40 分钟
"""
import paramiko, os, json, time

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_DIR = r"e:\three_chain_v3\results_cloud"

ALL_EXPS = [
    "run_100m_hta_seed0_500", "run_100m_hta_seed1_500", "run_100m_hta_seed2_500",
    "run_100m_seed0_500", "run_100m_seed1", "run_100m_seed2",
    "run_100m_seed3_500", "run_100m_seed4_500",
    "run_300m_seed0_2k", "run_300m_seed1_2k", "run_300m_seed2_2k",
    "run_30m_seed1", "run_30m_seed2", "run_30m_seed3", "run_30m_seed4",
    "run_three_chain_mamba2_30m_seed0",
    "run_700m_seed0_2k", "run_1000m_seed0_2k",
]

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)
    return c

def run(c, cmd, t=30):
    i, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")

def parse_ood(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        curve = d.get("changed_acc_curve", {})
        c100 = curve.get("100")
        c150 = curve.get("150")
        if c100 is not None and c150 is not None:
            return c100, c150, (c150 - c100) / c100 * 100
    except:
        pass
    return None, None, None

def download_all():
    c = connect()
    sftp = c.open_sftp()
    n = 0
    for exp in ALL_EXPS:
        rd = f"/root/three_chain_v3/results/{exp}"
        ld = os.path.join(LOCAL_DIR, exp)
        os.makedirs(ld, exist_ok=True)
        for fname in ["ood_metrics.json", "summary.json", "log.jsonl"]:
            try:
                sftp.get(f"{rd}/{fname}", os.path.join(ld, fname))
                n += 1
            except:
                pass
    sftp.close()
    c.close()
    return n

def print_summary():
    print(f"\n{'实验':<35} {'ch@100':>8} {'ch@150':>8} {'decay':>8}")
    print("-" * 65)
    done = 0
    for exp in ALL_EXPS:
        p = os.path.join(LOCAL_DIR, exp, "ood_metrics.json")
        if os.path.exists(p):
            c100, c150, decay = parse_ood(p)
            if c100 is not None:
                print(f"{exp:<35} {c100:>8.4f} {c150:>8.4f} {decay:>+7.1f}%")
                done += 1
            else:
                print(f"{exp:<35} {'?':>8} {'?':>8} {'?':>8}")
        else:
            print(f"{exp:<35} {'-':>8} {'-':>8} {'-':>8}")
    print(f"\n共 {done} 个实验有 OOD 结果")

def main():
    print("开始监控 final_rerun.sh, 每 60s 检查一次")
    start = time.time()
    max_wait = 2400  # 40 分钟

    while time.time() - start < max_wait:
        try:
            c = connect()
            log, _ = run(c, "cat /root/final_rerun.log 2>/dev/null")
            proc, _ = run(c, "pgrep -af 'bash /root/final_rerun.sh' | grep -v grep | head -1")
            gpu, _ = run(c, "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null")
            c.close()

            elapsed = int(time.time() - start)
            lines = log.strip().split("\n")
            last_lines = lines[-4:] if len(lines) > 4 else lines

            print(f"\n[{elapsed}s] proc={'运行中' if proc.strip() else '已结束'} GPU={gpu.strip()}")
            for l in last_lines:
                print(f"  {l.strip()}")

            if "ALL_FINAL_DONE" in log:
                print("\n" + "=" * 60)
                print("ALL_FINAL_DONE! 开始下载所有结果...")
                print("=" * 60)
                n = download_all()
                print(f"下载 {n} 个文件到 {LOCAL_DIR}")
                print_summary()
                print("\n提醒: 请关闭 AutoDL 实例以停止计费!")
                return

        except Exception as e:
            print(f"[{int(time.time()-start)}s] 检查出错: {e}")

        time.sleep(60)

    print("\n超时 40 分钟, 强制下载已有结果")
    n = download_all()
    print(f"下载 {n} 个文件")
    print_summary()

if __name__ == "__main__":
    main()
