"""阶段 3.2 消融实验编排脚本

跑 4 个 ablation 模型 (no_spatial / no_temporal / no_causal / no_all) × seed=2 × 500步 × batch=2
每个模型: train.py → eval_ood.py (用对应 ablation config, 确保 ablate 开关在 eval 时生效)

用法:
    python3 run_stage3_ablation.py             # 跑全部 4 个
    python3 run_stage3_ablation.py no_spatial   # 只跑指定一个

断点续跑: 已完成的模型 (ood_metrics.json 存在) 会自动跳过。

输出: results_stage3/ablation/{no_spatial,no_temporal,no_causal,no_all}/
"""
import subprocess
import sys
import time
import json
import shutil
from pathlib import Path

BASE = Path(__file__).parent.resolve()
OUT_BASE = BASE / "results_stage3" / "ablation"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# (name, config, ablate_s, ablate_t, ablate_c)
ABLATIONS = [
    ("no_spatial",  "ablation_no_spatial.yaml",  True,  False, False),
    ("no_temporal", "ablation_no_temporal.yaml", False, True,  False),
    ("no_causal",   "ablation_no_causal.yaml",   False, False, True),
    ("no_all",      "ablation_no_all.yaml",      True,  True,  True),
]
SEED = 2
MAX_STEPS = 500
BATCH = 2
OOD_BATCH = 8  # eval_ood batch_size (30M 在 24G 上 eval 显存压力小, 可用 8)


def run(cmd, log_file):
    """运行子进程并把 stdout/stderr 追加写到 log_file。"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== CMD: {' '.join(str(c) for c in cmd)} ===\n")
        f.flush()
    result = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n" + result.stderr)
    return result.returncode


def main():
    # 支持命令行指定只跑某些 ablation
    target = sys.argv[1] if len(sys.argv) > 1 else None
    ablations = [a for a in ABLATIONS if target is None or a[0] == target]
    if target and not ablations:
        print(f"[ERROR] 未知 ablation 名: {target}")
        print(f"  可选: {[a[0] for a in ABLATIONS]}")
        sys.exit(1)

    progress_log = OUT_BASE / "progress.log"
    ood_data = BASE / "data" / "ood_T150"

    # 训练数据路径: 云端用 data_split_m3, 本地用 data_split (自动检测)
    data_split_m3 = BASE / "data_split_m3"
    data_split_local = BASE / "data_split"
    if data_split_m3.exists():
        data_split = str(data_split_m3)
        print(f"[INFO] 检测到云端数据: {data_split}")
    elif data_split_local.exists():
        data_split = str(data_split_local)
        print(f"[INFO] 检测到本地数据: {data_split}")
    else:
        data_split = str(data_split_local)
        print(f"[WARN] 未找到训练数据目录, 默认用: {data_split}")

    # 检查 OOD 数据存在
    if not ood_data.exists():
        print(f"[ERROR] OOD 数据不存在: {ood_data}")
        print("  请先生成 OOD 评估数据 (data/ood_T150/)")
        sys.exit(1)

    print(f"=== 阶段 3.2 消融实验 START: {time.ctime()} ===")
    print(f"将跑 {len(ablations)} 个模型: {[a[0] for a in ablations]}")
    print(f"seed={SEED}, max_steps={MAX_STEPS}, batch={BATCH}")
    print(f"OOD 数据: {ood_data}")
    print(f"输出目录: {OUT_BASE}")
    print()

    for name, config, abl_s, abl_t, abl_c in ablations:
        out_dir = OUT_BASE / name
        ood_file = out_dir / "ood_metrics.json"

        # 断点续跑: 已完成的跳过
        if ood_file.exists():
            msg = f"[SKIP] {name} (ood_metrics.json 已存在)"
            print(msg)
            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            continue

        # 清理旧目录重建
        if out_dir.exists():
            shutil.rmtree(str(out_dir))
        out_dir.mkdir(parents=True)
        run_log = out_dir / "run.log"

        msg = (f"=== [{name}] START: {time.ctime()} "
               f"(ablate_s={abl_s}, ablate_t={abl_t}, ablate_c={abl_c}) ===")
        print(msg)
        with open(progress_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        # --- 训练 ---
        train_cmd = [
            sys.executable, str(BASE / "train.py"),
            "--model", "three_chain_mamba2",
            "--config", str(BASE / "configs" / config),
            "--seed", str(SEED),
            "--max_steps", str(MAX_STEPS),
            "--batch_size", str(BATCH),
            "--eval_every", "200",
            "--log_every", "50",
            "--out_dir", str(out_dir),
            "--data_root", str(data_split),
        ]
        rc = run(train_cmd, run_log)

        # --- eval_ood (必须传 --config 让 ablate 开关在 eval 时生效) ---
        best_pt = out_dir / "best.pt"
        if best_pt.exists():
            eval_cmd = [
                sys.executable, str(BASE / "eval_ood.py"),
                "--checkpoint", str(best_pt),
                "--config", str(BASE / "configs" / config),
                "--data_root", str(ood_data),
                "--out", str(out_dir / "ood_metrics.json"),
                "--batch_size", str(OOD_BATCH),
            ]
            run(eval_cmd, run_log)

            # 读 decay 汇报
            try:
                m = json.loads((out_dir / "ood_metrics.json").read_text(encoding="utf-8"))
                decay_pct = m.get("ood_decay_pct", 0) * 100
                ch_final = m.get("changed_acc", 0)
                msg = (f"[OK] {name}: decay={decay_pct:+.2f}%, "
                       f"changed_acc@150={ch_final:.4f}")
            except Exception as e:
                msg = f"[WARN] {name} 读取 decay 失败: {e}"
            print(msg)
            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        else:
            msg = f"[WARN] No best.pt for {name} (train rc={rc})"
            print(msg)
            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        msg = f"=== [{name}] DONE: {time.ctime()} ==="
        print(msg)
        with open(progress_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print()

    # --- 汇总 ---
    print("=== ALL ABLATION DONE ===")
    print("\n=== 消融结果汇总 ===")
    print(f"{'模型':<15} {'decay%':<10} {'changed_acc@150':<15}")
    print("-" * 40)
    for name, _, _, _, _ in ablations:
        ood_file = OUT_BASE / name / "ood_metrics.json"
        if ood_file.exists():
            try:
                m = json.loads(ood_file.read_text(encoding="utf-8"))
                decay = m.get("ood_decay_pct", 0) * 100
                ch = m.get("changed_acc", 0)
                print(f"{name:<15} {decay:+.2f}%    {ch:.4f}")
            except Exception:
                print(f"{name:<15} [读取失败]")
        else:
            print(f"{name:<15} [未完成]")
    with open(progress_log, "a", encoding="utf-8") as f:
        f.write(f"=== ALL ABLATION DONE: {time.ctime()} ===\n")


if __name__ == "__main__":
    main()
