"""
monitor_tail.py - 后台监控 tail_rerun 进度, 完成后自动下载所有结果
每 90s 检查一次, 最长等 60 分钟
"""
import paramiko, os, json, time, sys

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
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

def parse_ood(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        curve = d.get("changed_acc_curve", {})
        c100 = curve.get("100"); c150 = curve.get("150")
        if c100 is not None and c150 is not None:
            return c100, c150, (c150-c100)/c100*100
    except: pass
    return None, None, None

def download_all():
    """下载所有实验的 JSON"""
    c = connect()
    sftp = c.open_sftp()
    n = 0
    for exp in ALL_EXPS:
        rd = f"/root/three_chain_v3/results/{exp}"
        ld = os.path.join(LOCAL_DIR, exp)
        os.makedirs(ld, exist_ok=True)
        for f in ["ood_metrics.json", "summary.json", "log.jsonl"]:
            try:
                sftp.get(f"{rd}/{f}", os.path.join(ld, f))
                n += 1
            except: pass
    sftp.close()
    c.close()
    return n

def print_summary():
    print(f"\n{'实验':<35} {'ch@100':>8} {'ch@150':>8} {'decay':>8}")
    print("-" * 65)
    for exp in ALL_EXPS:
        p = os.path.join(LOCAL_DIR, exp, "ood_metrics.json")
        if os.path.exists(p):
            c100, c150, decay = parse_ood(p)
            if c100 is not None:
                print(f"{exp:<35} {c100:>8.4f} {c150:>8.4f} {decay:>+7.1f}%")
            else:
                print(f"{exp:<35} {'?':>8} {'?':>8} {'?':>8}")
        else:
            print(f"{exp:<35} {'-':>8} {'-':>8} {'-':>8}")

def main():
    print(f"开始监控 tail_rerun, 每 90s 检查一次")
    start = time.time()
    max_wait = 3600  # 60 分钟

    while time.time() - start < max_wait:
        try:
            c = connect()
            # 检查 tail_rerun.log
            out, _ = run(c, "cat /root/tail_rerun.log 2>/dev/null")
            # 700M 进度
            out7, _ = run(c, "tail -1 /root/three_chain_v3/results/run_700m_seed0_2k/log.jsonl 2>/dev/null")
            # 1000M 进度
            out10, _ = run(c, "ls /root/three_chain_v3/results/run_1000m_seed0_2k/ 2>/dev/null | head -3")
            # tail_rerun 进程
            outp, _ = run(c, "pgrep -af 'bash /root/tail_rerun.sh' | grep -v grep | head -1")
            # run_700m_1000m.sh 进程
            outr, _ = run(c, "pgrep -af run_700m_1000m.sh | grep -v grep | head -1")
            c.close()

            elapsed = int(time.time() - start)
            print(f"\n[已等 {elapsed}s]")

            # 700M 状态
            if "step" in out7:
                import re
                m = re.search(r'"step": (\d+).*"changed_acc": ([\d.]+)', out7)
                if m:
                    print(f"  700M: step {m.group(1)}/2000, ch_acc={m.group(2)[:6]}")
            elif "val" in out7:
                print(f"  700M: 训练完成 (val)")

            # 1000M 状态
            if out10.strip():
                print(f"  1000M: 已开始 ({out10.strip()[:50]})")
            else:
                print(f"  1000M: 未开始")

            # run_700m_1000m.sh 状态
            if outr.strip():
                print(f"  run_700m_1000m.sh: 运行中")
            else:
                print(f"  run_700m_1000m.sh: 已结束")

            # tail_rerun 状态
            if outp.strip():
                print(f"  tail_rerun.sh: 运行中")
            else:
                print(f"  tail_rerun.sh: 已结束")

            # tail_rerun 日志关键行
            for line in out.strip().split("\n"):
                if any(k in line for k in ["完成", "结束", "开始", "训练", "OOD", "ALL_TAIL_DONE", "OOM", "失败"]):
                    print(f"  日志: {line.strip()}")

            if "ALL_TAIL_DONE" in out:
                print("\n✓✓✓ tail_rerun 全部完成!")
                print("\n[下载最终结果 ...]")
                n = download_all()
                print(f"  ✓ 下载 {n} 个文件")
                print_summary()
                return

        except Exception as e:
            print(f"  检查出错: {e}")

        time.sleep(90)

    print("超时 (60 分钟), 强制下载已有结果")
    n = download_all()
    print(f"下载 {n} 个文件")
    print_summary()

if __name__ == "__main__":
    main()
