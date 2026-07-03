"""
综合监控: 等待 final_rerun.sh + rerun_700m.sh 全部完成
完成后自动下载所有结果, 然后关机 AutoDL 节省费用
"""
import paramiko, time, os, json, sys
from pathlib import Path

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_DIR = Path(r"e:\three_chain_v3\results_cloud")

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"开始综合监控, 结果将下载到 {LOCAL_DIR}")
    print(f"每 90s 检查一次, 最多等 90 分钟\n")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)

    start = time.time()
    MAX_WAIT = 5400  # 90 分钟
    all_done = False

    while time.time() - start < MAX_WAIT:
        elapsed = int(time.time() - start)

        # 检查进程状态
        out, _ = run(c, "pgrep -af 'bash /root/final_rerun.sh' | grep -v grep || echo NONE")
        final_running = "NONE" not in out
        out2, _ = run(c, "pgrep -af 'bash /root/rerun_700m.sh' | grep -v grep || echo NONE")
        rerun_running = "NONE" not in out2

        # 检查日志标记
        final_done_mark, _ = run(c, "grep -c 'ALL_FINAL_DONE' /root/final_rerun.log 2>/dev/null || echo 0")
        rerun_done_mark, _ = run(c, "grep -c 'RERUN_700M_DONE' /root/rerun_700m.log 2>/dev/null || echo 0")

        final_done = final_done_mark.strip() != "0"
        rerun_done = rerun_done_mark.strip() != "0"

        # 当前进度
        final_tail, _ = run(c, "tail -3 /root/final_rerun.log 2>/dev/null")
        rerun_tail, _ = run(c, "tail -3 /root/rerun_700m.log 2>/dev/null")

        # 300m_seed2 / 100m_hta_seed1 结果
        r300m2, _ = run(c, "cat /root/three_chain_v3/results/run_300m_seed2_2k/ood_metrics.json 2>/dev/null | head -1")
        hta1, _ = run(c, "cat /root/three_chain_v3/results/run_100m_hta_seed1_500/ood_metrics.json 2>/dev/null | head -1")
        r700m, _ = run(c, "cat /root/three_chain_v3/results/run_700m_seed0_2k/ood_metrics.json 2>/dev/null | head -1")

        print(f"[已等 {elapsed}s]")
        print(f"  final_rerun: {'运行中' if final_running else '已结束'} (done={final_done})")
        if final_tail.strip():
            for line in final_tail.strip().split("\n")[-2:]:
                print(f"    {line}")
        print(f"  rerun_700m:  {'运行中' if rerun_running else '已结束'} (done={rerun_done})")
        if rerun_tail.strip():
            for line in rerun_tail.strip().split("\n")[-2:]:
                print(f"    {line}")
        print(f"  300m_seed2: {'有结果' if r300m2.strip() else '无'}")
        print(f"  hta_seed1:  {'有结果' if hta1.strip() else '无'}")
        print(f"  700m:       {'有结果' if r700m.strip() else '无'}")
        print()

        # 全部完成?
        if final_done and rerun_done:
            print("✓✓✓ final_rerun + rerun_700m 全部完成!")
            all_done = True
            break

        time.sleep(90)
        c.close()
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)

    if not all_done:
        print("⚠ 超时未完成, 下载已有结果")
    else:
        print("\n[下载所有结果 ...]")
        sftp = c.open_sftp()
        count = 0
        remote_results = "/root/three_chain_v3/results"
        for exp_dir in sftp.listdir(remote_results):
            remote_exp = f"{remote_results}/{exp_dir}"
            local_exp = LOCAL_DIR / exp_dir
            local_exp.mkdir(parents=True, exist_ok=True)
            try:
                for fname in sftp.listdir(remote_exp):
                    if fname.endswith(".json"):
                        remote_file = f"{remote_exp}/{fname}"
                        local_file = local_exp / fname
                        sftp.get(remote_file, str(local_file))
                        count += 1
            except IOError:
                pass
        sftp.close()
        print(f"  ✓ 下载 {count} 个文件")

        # 打印最终汇总
        print("\n=== 最终实验汇总 ===")
        print(f"{'实验':<35} {'ch@100':>8} {'ch@150':>8} {'decay':>8}")
        print("-" * 65)
        for exp_dir in sorted(LOCAL_DIR.iterdir()):
            if not exp_dir.is_dir():
                continue
            ood_file = exp_dir / "ood_metrics.json"
            if not ood_file.exists():
                continue
            try:
                m = json.loads(ood_file.read_text(encoding="utf-8"))
                ch = m.get("changed_acc_curve", {})
                a100 = ch.get("100", None)
                a150 = ch.get("150", None)
                if a100 is not None and a150 is not None:
                    decay = (a150 - a100) / a100 * 100 if a100 > 0 else 0
                    print(f"{exp_dir.name:<35} {a100:>8.4f} {a150:>8.4f} {decay:>+7.1f}%")
                else:
                    print(f"{exp_dir.name:<35} {'?':>8} {'?':>8} {'?':>8}")
            except Exception as ex:
                print(f"{exp_dir.name:<35} ERROR: {ex}")

    c.close()
    print("\n✓ 监控结束")

if __name__ == "__main__":
    main()
