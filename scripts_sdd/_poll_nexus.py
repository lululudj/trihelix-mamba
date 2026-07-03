"""后台轮询 stage2_nexus 完成, 完成后下载 + 对比 bookstore vs nexus.

每 5 分钟检查一次 /root/STAGE2_NEXUS_DONE marker (或关键 metrics 都已生成).
完成后下载 SSM 10k + shuffle 3k 的 ood_metrics + ood_metrics_advanced + negative_shadow,
计算 window decay, 打印 bookstore vs nexus 完整对比.
"""
import os
import sys
import time
import json
import paramiko
from pathlib import Path

# 强制 unbuffered, 防 stdout 缓冲 (后台任务看不到进度)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

HOST = "123.127.15.155"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_ROOT = Path(r"e:\three_chain_v3")
REMOTE_DIR = "/root/three_chain_v3"
CHECK_INTERVAL = 300  # 5 min


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    return c


def run(c, cmd, timeout=30):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', errors='replace'), e.read().decode('utf-8', errors='replace')


def window_avg(curve, center, half=5):
    """±half 邻域均值, 规避单点抖动 (ch@100 异常深坑)."""
    vals = [curve.get(str(t)) for t in range(center - half, center + half + 1)
            if curve.get(str(t)) is not None]
    return sum(vals) / len(vals) if vals else None


def download_file(sftp, remote_path, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sftp.get(remote_path, str(local_path))
        return True
    except IOError as e:
        print(f"  [下载失败] {remote_path}: {e}")
        return False


def main():
    print(f"[{time.strftime('%H:%M:%S')}] 开始轮询 stage2_nexus 完成...", flush=True)
    start = time.time()
    poll_count = 0

    while True:
        poll_count += 1
        try:
            c = connect()
            # 1. 检查 marker
            _, o, _ = c.exec_command(f'test -f /root/STAGE2_NEXUS_DONE && echo DONE || echo NOT')
            done = o.read().decode().strip() == 'DONE'

            # 2. 检查关键 metrics 都已生成 (即使 marker 未 touch, 也能检测完成)
            _, o, _ = c.exec_command(
                f'ls {REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/ood_metrics_advanced.json '
                f'{REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/negative_shadow.json '
                f'{REMOTE_DIR}/results_stage2/nexus_shuffle_3k/ood_metrics_advanced.json 2>/dev/null | wc -l'
            )
            metrics_count = int(o.read().decode().strip() or '0')

            # 3. 检查训练进度
            _, o, _ = c.exec_command(
                f'tail -3 {REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/train.log 2>/dev/null;'
                f'grep -oE "step [0-9]+/10000" {REMOTE_DIR}/../stage2_nexus.log 2>/dev/null | tail -1'
            )
            progress = o.read().decode().strip()

            elapsed = (time.time() - start) / 60
            print(f"[{time.strftime('%H:%M:%S')}] poll#{poll_count} elapsed={elapsed:.0f}min "
                  f"marker={'DONE' if done else 'NO'} metrics_files={metrics_count}/3 "
                  f"progress={progress or '(empty)'}", flush=True)

            if done or metrics_count >= 3:
                print(f"\n[{time.strftime('%H:%M:%S')}] !!! nexus 完成 (marker={'DONE' if done else 'NO'}, "
                      f"metrics={metrics_count}/3)\n", flush=True)
                sftp = c.open_sftp()

                # 下载所有 metrics
                downloads = [
                    (f"{REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/ood_metrics.json",
                     f"{LOCAL_ROOT}/results_stage2/nexus_30m_seed0_10k/ood_metrics.json"),
                    (f"{REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/ood_metrics_advanced.json",
                     f"{LOCAL_ROOT}/results_stage2/nexus_30m_seed0_10k/ood_metrics_advanced.json"),
                    (f"{REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/negative_shadow.json",
                     f"{LOCAL_ROOT}/results_stage2/nexus_30m_seed0_10k/negative_shadow.json"),
                    (f"{REMOTE_DIR}/results_stage2/nexus_shuffle_3k/ood_metrics.json",
                     f"{LOCAL_ROOT}/results_stage2/nexus_shuffle_3k/ood_metrics.json"),
                    (f"{REMOTE_DIR}/results_stage2/nexus_shuffle_3k/ood_metrics_advanced.json",
                     f"{LOCAL_ROOT}/results_stage2/nexus_shuffle_3k/ood_metrics_advanced.json"),
                ]
                for r, l in downloads:
                    ok = download_file(sftp, r, l)
                    if ok:
                        print(f"  下载 {Path(l).name} OK", flush=True)

                sftp.close()
                c.close()
                print_comparison()
                return

            c.close()
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [轮询异常] {e}", flush=True)

        if elapsed > 360:  # 6h 超时
            print(f"\n[{time.strftime('%H:%M:%S')}] !!! 超时 6h, 退出", flush=True)
            return
        time.sleep(CHECK_INTERVAL)


def print_comparison():
    """打印 bookstore vs nexus 完整对比."""
    print("\n" + "=" * 70, flush=True)
    print("=== Bookstore vs Nexus 完整对比 ===", flush=True)
    print("=" * 70, flush=True)

    base = LOCAL_ROOT / "results_stage2"
    files = {
        "bookstore_ssm": base / "sdd_30m_seed0_10k" / "ood_metrics_advanced.json",
        "bookstore_shuffle": base / "sdd_shuffle_3k" / "ood_metrics_advanced.json",
        "bookstore_shadow": base / "sdd_30m_seed0_10k" / "negative_shadow.json",
        "nexus_ssm": base / "nexus_30m_seed0_10k" / "ood_metrics_advanced.json",
        "nexus_shuffle": base / "nexus_shuffle_3k" / "ood_metrics_advanced.json",
        "nexus_shadow": base / "nexus_30m_seed0_10k" / "negative_shadow.json",
    }
    data = {}
    for k, p in files.items():
        if p.exists():
            try:
                data[k] = json.loads(p.read_text(encoding='utf-8'))
            except Exception as e:
                print(f"  [读 {k} 失败] {e}", flush=True)

    # === 表 1: SSM OOD position_iou decay (核心指标) ===
    print("\n--- 表 1: SSM position_iou OOD decay (核心) ---", flush=True)
    print(f"{'场景':<12} {'pos_iou@100':<14} {'pos_iou@150':<14} {'decay%':<10}", flush=True)
    for scene, key in [("bookstore", "bookstore_ssm"), ("nexus", "nexus_ssm")]:
        m = data.get(key)
        if m and "pos_iou_curve" in m:
            curve = m["pos_iou_curve"]
            v100 = window_avg(curve, 100)
            v150 = window_avg(curve, 150)
            decay_pct = (v150 - v100) / v100 * 100 if v100 and v100 > 0 else float('nan')
            print(f"{scene:<12} {v100:<14.4f} {v150:<14.4f} {decay_pct:<+10.2f}", flush=True)

    # === 表 2: shuffle sanity (position_iou 应接近 0) ===
    print("\n--- 表 2: shuffle_labels sanity (pos_iou 应 ≈ 0) ---", flush=True)
    print(f"{'场景':<12} {'SSM pos_iou':<14} {'shuffle pos_iou':<18} {'SSM/shuffle':<12}", flush=True)
    for scene, sk, shk in [("bookstore", "bookstore_ssm", "bookstore_shuffle"),
                           ("nexus", "nexus_ssm", "nexus_shuffle")]:
        ssm = data.get(sk)
        sh = data.get(shk)
        ssm_v = window_avg(ssm["pos_iou_curve"], 100) if ssm and "pos_iou_curve" in ssm else None
        sh_v = window_avg(sh["pos_iou_curve"], 100) if sh and "pos_iou_curve" in sh else None
        ratio = ssm_v / sh_v if ssm_v and sh_v and sh_v > 0 else float('inf')
        ssm_str = f"{ssm_v:.4f}" if ssm_v else "N/A"
        sh_str = f"{sh_v:.4f}" if sh_v else "N/A"
        ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "∞"
        print(f"{scene:<12} {ssm_str:<14} {sh_str:<18} {ratio_str:<12}", flush=True)

    # === 表 3: 负面分身 (ablate_all / normal pos_iou ratio) ===
    print("\n--- 表 3: 负面分身 (ablate_all pos_iou 降多少 → 三链贡献) ---", flush=True)
    print(f"{'场景':<12} {'normal':<10} {'ablate_s':<10} {'ablate_t':<10} {'ablate_c':<10} {'ablate_all':<10} {'保留%':<8}", flush=True)
    for scene, key in [("bookstore", "bookstore_shadow"), ("nexus", "nexus_shadow")]:
        m = data.get(key)
        if not m or "modes" not in m:
            print(f"{scene:<12} (无数据)", flush=True)
            continue
        modes = m["modes"]
        vals = {}
        for mode in ["normal", "ablate_s", "ablate_t", "ablate_c", "ablate_all"]:
            v = modes.get(mode, {}).get("pos_iou")
            vals[mode] = v
        normal = vals.get("normal") or 0
        all_v = vals.get("ablate_all") or 0
        keep = all_v / normal * 100 if normal > 0 else 0
        row = f"{scene:<12} "
        for mode in ["normal", "ablate_s", "ablate_t", "ablate_c", "ablate_all"]:
            v = vals.get(mode)
            row += f"{(f'{v:.4f}' if v else 'N/A'):<10} "
        row += f"{keep:<8.1f}"
        print(row, flush=True)

    # === 表 4: changed_acc (失效指标, 对比验证) ===
    print("\n--- 表 4: changed_acc (失效指标, 仅对比) ---", flush=True)
    print(f"{'场景':<12} {'SSM ch@100':<14} {'SSM ch@150':<14} {'decay%':<10} {'shuffle ch@100':<18}", flush=True)
    for scene, sk, shk in [("bookstore", "bookstore_ssm", "bookstore_shuffle"),
                           ("nexus", "nexus_ssm", "nexus_shuffle")]:
        ssm = data.get(sk)
        sh = data.get(shk)
        ch_curve = ssm.get("changed_acc_curve") if ssm else None
        ch100 = window_avg(ch_curve, 100) if ch_curve else None
        ch150 = window_avg(ch_curve, 150) if ch_curve else None
        decay = (ch150 - ch100) / ch100 * 100 if ch100 and ch100 > 0 else float('nan')
        sh_curve = sh.get("changed_acc_curve") if sh else None
        sh100 = window_avg(sh_curve, 100) if sh_curve else None
        print(f"{scene:<12} {ch100 if ch100 else 0:<14.4f} {ch150 if ch150 else 0:<14.4f} "
              f"{decay:<+10.2f} {sh100 if sh100 else 0:<18.4f}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("=== 全部完成 ===", flush=True)


if __name__ == "__main__":
    main()
