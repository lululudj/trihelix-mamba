"""
监控额外实验: 等 ALL_EXTRA_DONE 后下载全部结果并打印最终汇总
"""
import paramiko, time, json
from pathlib import Path

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_DIR = Path(r"e:\three_chain_v3\results_cloud")

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15, banner_timeout=15)
    return c

def main():
    print("=== 监控额外实验 ===")
    print(f"每 120s 检查一次, 最多等 120 分钟\n")

    c = connect()
    start = time.time()
    MAX_WAIT = 7200  # 120 分钟

    while time.time() - start < MAX_WAIT:
        elapsed = int(time.time() - start)

        # 检查 ALL_EXTRA_DONE
        out, _ = run(c, "grep 'ALL_EXTRA_DONE' /root/extra_experiments.log 2>/dev/null || echo NOTFOUND")
        extra_done = "NOTFOUND" not in out

        # 检查当前实验
        out, _ = run(c, "grep 'RERUN_700M_V2_DONE' /root/run_700m_v2.log 2>/dev/null || echo NOTFOUND")
        m700_done = "NOTFOUND" not in out
        out, _ = run(c, "grep 'HTA_SEED1_DONE' /root/run_hta_seed1.log 2>/dev/null || echo NOTFOUND")
        hta1_done = "NOTFOUND" not in out

        # 额外实验进度
        out, _ = run(c, "tail -15 /root/extra_experiments.log 2>/dev/null")
        extra_log = out.strip()

        # 各实验结果检查
        results = {}
        for exp in ["run_30m_hta_seed0_500", "run_300m_hta_seed0_2k",
                     "run_700m_seed1_2k", "run_700m_hta_seed0_2k",
                     "run_700m_seed0_2k", "run_100m_hta_seed1_500"]:
            out, _ = run(c, f"cat /root/three_chain_v3/results/{exp}/ood_metrics.json 2>/dev/null | head -1")
            results[exp] = bool(out.strip())

        print(f"\n[已等 {elapsed}s] ({elapsed//60}min)")
        print(f"  700M seed0: {'✓' if m700_done else '🔄'} | OOD: {'有' if results.get('run_700m_seed0_2k') else '无'}")
        print(f"  hta_seed1: {'✓' if hta1_done else '🔄'} | OOD: {'有' if results.get('run_100m_hta_seed1_500') else '无'}")
        print(f"  额外实验:  {'✓全部完成' if extra_done else '🔄运行中'}")
        print(f"    30M HTA:  {'✓' if results.get('run_30m_hta_seed0_500') else '⏳'}")
        print(f"    300M HTA: {'✓' if results.get('run_300m_hta_seed0_2k') else '⏳'}")
        print(f"    700M s1:  {'✓' if results.get('run_700m_seed1_2k') else '⏳'}")
        print(f"    700M HTA: {'✓' if results.get('run_700m_hta_seed0_2k') else '⏳'}")
        if extra_log:
            for line in extra_log.split("\n")[-5:]:
                print(f"    {line}")

        if extra_done:
            print("\n✓✓✓ 所有实验全部完成!")
            break

        # 也检查是否所有进程都结束了 (可能脚本崩溃)
        out, _ = run(c, "pgrep -af 'extra_experiments|run_700m_v2|run_hta_seed1|train.py|eval_ood' | grep -v grep || echo NONE")
        if "NONE" in out and not extra_done:
            print("\n⚠ 所有进程已结束但未找到 ALL_EXTRA_DONE 标记, 检查是否出错 ...")
            # 再等一轮确认
            time.sleep(30)
            out2, _ = run(c, "grep 'ALL_EXTRA_DONE' /root/extra_experiments.log 2>/dev/null || echo NOTFOUND")
            if "NOTFOUND" not in out2:
                print("✓ 确认完成")
                break
            else:
                print("⚠ 确认未完成, 可能脚本崩溃, 下载已有结果")
                break

        time.sleep(120)
        c.close()
        c = connect()

    # 下载所有结果
    print("\n[下载所有结果 ...]")
    sftp = c.open_sftp()
    remote_results = "/root/three_chain_v3/results"
    count = 0
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

    # 最终汇总
    print("\n" + "=" * 75)
    print("=== 最终实验汇总 (全部) ===")
    print("=" * 75)
    print(f"{'实验':<40} {'ch@100':>8} {'ch@150':>8} {'decay':>8}")
    print("-" * 75)

    all_results = []
    for exp_dir in sorted(LOCAL_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        ood_file = exp_dir / "ood_metrics.json"
        if not ood_file.exists() or ood_file.stat().st_size == 0:
            continue
        try:
            m = json.loads(ood_file.read_text(encoding="utf-8"))
            ch = m.get("changed_acc_curve", {})
            a100 = ch.get("100")
            a150 = ch.get("150")
            if a100 is not None and a150 is not None:
                decay = (a150 - a100) / a100 * 100 if a100 > 0 else 0
                all_results.append((exp_dir.name, a100, a150, decay))
                print(f"{exp_dir.name:<40} {a100:>8.4f} {a150:>8.4f} {decay:>+7.1f}%")
        except:
            pass

    # 按规模分组 (排除旧的 100 步结果)
    print("\n=== 按规模汇总 ===")
    groups = {
        "baseline (3.26M)": [],
        "30M (27M)": [],
        "30M HTA (27M)": [],
        "100M (72M)": [],
        "100M HTA (72M)": [],
        "300M (182M)": [],
        "300M HTA (182M)": [],
        "700M (~725M)": [],
        "700M HTA (~725M)": [],
    }
    for name, a100, a150, decay in all_results:
        # 排除旧的 100 步结果 (没有 _500 或 _2k 后缀的)
        if not (name.endswith("_500") or name.endswith("_2k") or "seed0" in name and "_2k" in name):
            if name in ("run_100m_seed1", "run_100m_seed2"):
                pass  # 这些是有效的
            elif not (name.endswith("_500") or name.endswith("_2k")):
                continue  # 跳过旧结果

        if "700m_hta" in name:
            groups["700M HTA (~725M)"].append(decay)
        elif "700m" in name:
            groups["700M (~725M)"].append(decay)
        elif "300m_hta" in name:
            groups["300M HTA (182M)"].append(decay)
        elif "300m" in name:
            groups["300M (182M)"].append(decay)
        elif "hta" in name and "100m" in name:
            groups["100M HTA (72M)"].append(decay)
        elif "hta" in name and "30m" in name:
            groups["30M HTA (27M)"].append(decay)
        elif "100m" in name:
            groups["100M (72M)"].append(decay)
        elif "30m" in name:
            groups["30M (27M)"].append(decay)
        else:
            groups["baseline (3.26M)"].append(decay)

    for label, decays in groups.items():
        if decays:
            avg = sum(decays) / len(decays)
            print(f"  {label}: {len(decays)} seeds, avg decay={avg:+.1f}%")
        else:
            print(f"  {label}: 无结果")

    c.close()
    print("\n✓ 全部完成! 可以关机了")

if __name__ == "__main__":
    main()
