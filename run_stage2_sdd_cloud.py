"""阶段 2 SDD 真实数据验证 — Phase A MVP 云端编排器

流程:
  1. 上传 gen_sdd_grid.py + sdd_mamba2_30m.yaml + download_sdd_annotations.py
  2. 云端下载 SDD annotations (remotezip, 只取 bookstore ~50MB)
  3. 生成 .npz 训练数据 (bookstore, T=100) + OOD 数据 (T=150)
  4. 划分 train/val/test
  5. 训练 smoke test 10k 步 (batch=2)
  6. OOD eval T150
  7. 下载结果 (ood_metrics.json + summary.json + 日志)

零模型改动: 不碰 models/three_chain_mamba2.py, 复用现有 train.py / eval_ood.py / dataset.py
断点续跑: 检查 /root/STAGE2_MVP_DONE marker

预计云端耗时: ~30-60 min (下载2min + 生成1min + 训练10k步~20min + eval 2min)
"""
import os
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
REMOTE_ROOT = "/root/three_chain_v3"
LOCAL_BASE = r"e:\three_chain_v3"

# 云端 shell 脚本: 下载 + 生成 + 训练 + eval
RUN_SH = r"""#!/bin/bash
LOG=/root/stage2_mvp.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date +%H:%M:%S)] Stage 2 SDD MVP Start ===" >> $LOG

# ============================================================
# 1. 下载 SDD annotations (remotezip, 只取 bookstore)
# ============================================================
if [ -d "./SDD/annotations/bookstore" ]; then
    echo "[$(date +%H:%M:%S)] SDD annotations 已存在, 跳过下载" >> $LOG
else
    echo "[$(date +%H:%M:%S)] === 1. Download SDD annotations (bookstore) ===" >> $LOG
    pip install remotezip -q >> $LOG 2>&1
    python scripts_sdd/download_sdd_annotations.py --out_dir ./SDD --scenes bookstore >> $LOG 2>&1
    # 验证下载
    N=$(find ./SDD/annotations/bookstore -name "annotations.txt" 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] Downloaded: $N annotations.txt files" >> $LOG
    if [ "$N" -lt 3 ]; then
        echo "[$(date +%H:%M:%S)] [FATAL] SDD 下载失败 (<3 files), 退出" >> $LOG
        touch /root/STAGE2_MVP_FAILED
        exit 1
    fi
fi

# ============================================================
# 2. 生成训练数据 (bookstore, T=100)
# ============================================================
if [ -d "./data_sdd_raw" ] && [ "$(ls -A ./data_sdd_raw 2>/dev/null)" ]; then
    echo "[$(date +%H:%M:%S)] data_sdd_raw 已存在, 跳过生成" >> $LOG
else
    echo "[$(date +%H:%M:%S)] === 2. Generate train data (T=100) ===" >> $LOG
    python data/gen_sdd_grid.py --sdd_root ./SDD --out_dir ./data_sdd_raw \
        --scenes bookstore --N 24 --K 8 --T 100 --fps_stride 6 >> $LOG 2>&1
    N=$(ls ./data_sdd_raw/*.npz 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] Generated: $N train windows" >> $LOG
fi

# ============================================================
# 3. 划分 train/val/test (8:1:1)
# ============================================================
if [ -d "./data_sdd_split/train" ]; then
    echo "[$(date +%H:%M:%S)] data_sdd_split 已存在, 跳过划分" >> $LOG
else
    echo "[$(date +%H:%M:%S)] === 3. Split train/val/test ===" >> $LOG
    python -c "from data.dataset import split_dataset; split_dataset('./data_sdd_raw', './data_sdd_split', ratios=(0.8,0.1,0.1), seed=0)" >> $LOG 2>&1
    for split in train val test; do
        N=$(ls ./data_sdd_split/$split/*.npz 2>/dev/null | wc -l)
        echo "[$(date +%H:%M:%S)]   $split: $N files" >> $LOG
    done
fi

# ============================================================
# 4. 生成 OOD 数据 (bookstore, T=150)
# ============================================================
if [ -d "./data/ood_T150" ] && [ "$(ls -A ./data/ood_T150 2>/dev/null)" ]; then
    echo "[$(date +%H:%M:%S)] data/ood_T150 已存在, 跳过生成" >> $LOG
else
    echo "[$(date +%H:%M:%S)] === 4. Generate OOD data (T=150) ===" >> $LOG
    python data/gen_sdd_grid.py --sdd_root ./SDD --out_dir ./data/ood_T150 \
        --scenes bookstore --N 24 --K 8 --T 150 --fps_stride 6 >> $LOG 2>&1
    N=$(ls ./data/ood_T150/*.npz 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] Generated: $N OOD windows" >> $LOG
fi

# ============================================================
# 5. 训练 smoke test (10k 步)
# ============================================================
if [ -f "./results_stage2/sdd_30m_seed0_10k/best.pt" ]; then
    echo "[$(date +%H:%M:%S)] best.pt 已存在, 跳过训练" >> $LOG
else
    echo "[$(date +%H:%M:%S)] === 5. Train smoke test (10k steps) ===" >> $LOG
    python -u train.py --model three_chain_mamba2 \
        --config configs/sdd_mamba2_30m.yaml \
        --max_steps 10000 --batch_size 2 --seed 0 \
        --data_root ./data_sdd_split \
        --out_dir results_stage2/sdd_30m_seed0_10k >> $LOG 2>&1
    echo "[$(date +%H:%M:%S)] Train done" >> $LOG
    # 训练完清理 final.pt + optimizer 节省磁盘
    rm -f results_stage2/sdd_30m_seed0_10k/final.pt
fi

# ============================================================
# 6. OOD eval T150
# ============================================================
if [ -f "./results_stage2/sdd_30m_seed0_10k/ood_metrics.json" ]; then
    echo "[$(date +%H:%M:%S)] ood_metrics.json 已存在, 跳过 eval" >> $LOG
else
    echo "[$(date +%H:%M:%S)] === 6. OOD eval T150 ===" >> $LOG
    python -u eval_ood.py \
        --checkpoint results_stage2/sdd_30m_seed0_10k/best.pt \
        --config configs/sdd_mamba2_30m.yaml \
        --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
        --out results_stage2/sdd_30m_seed0_10k/ood_metrics.json >> $LOG 2>&1
    echo "[$(date +%H:%M:%S)] Eval done" >> $LOG
fi

# ============================================================
# 7. 数据完整性统计 (供本地验证)
# ============================================================
echo "[$(date +%H:%M:%S)] === 7. Data stats ===" >> $LOG
python -c "
import numpy as np
from pathlib import Path
# 抽查 1 个 train npz
trains = sorted(Path('./data_sdd_split/train').glob('scen_*.npz'))
if trains:
    d = np.load(trains[0], allow_pickle=True)
    S0 = d['S_0']; acts = d['actions']; St = d['S_t']
    nz = int((S0 > 0).sum())
    # 变化率
    changed = 0; total = 0
    for t in range(1, min(6, St.shape[0])):
        changed += int((St[t] != St[0]).sum())
        total += St[t].size
    chg_rate = changed / max(total, 1)
    print(f'  sample: {trains[0].name}')
    print(f'  S_0 shape={S0.shape} dtype={S0.dtype} nonzero={nz}')
    print(f'  actions shape={acts.shape} dtype={acts.dtype}')
    print(f'  S_t shape={St.shape} dtype={St.dtype}')
    print(f'  change_rate(first5)={chg_rate:.4f}')
    print(f'  scenario_type={d[\"scenario_type\"]}')
" >> $LOG 2>&1

touch /root/STAGE2_MVP_DONE
echo "[$(date +%H:%M:%S)] === STAGE2_MVP_DONE ===" >> $LOG
"""


def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    return c


def run(c, cmd, timeout=30):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors='replace'), err.read().decode(errors='replace')


def download_dir(sftp, remote_dir, local_dir, skip_ext=('.pt',)):
    """下载目录下所有文件 (跳过指定后缀的大文件)."""
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    try:
        for entry in sftp.listdir(remote_dir):
            if any(entry.endswith(ext) for ext in skip_ext):
                print(f"    - 跳过 {entry} (大文件)")
                continue
            remote_path = f"{remote_dir}/{entry}"
            local_path = os.path.join(local_dir, entry)
            try:
                sftp.get(remote_path, local_path)
                st = sftp.stat(remote_path)
                print(f"    ✓ {entry} ({st.st_size} bytes)")
                n += 1
            except Exception as e:
                print(f"    - 跳过 {entry}: {e}")
    except FileNotFoundError:
        print(f"    [警告] 远端目录不存在: {remote_dir}")
    return n


def download_file(sftp, remote_path, local_path):
    try:
        sftp.get(remote_path, local_path)
        st = sftp.stat(remote_path)
        print(f"    ✓ {os.path.basename(remote_path)} ({st.st_size} bytes)")
        return True
    except FileNotFoundError:
        print(f"    [警告] 远端文件不存在: {remote_path}")
        return False


def tail_log(c, n=5):
    out, _ = run(c, f"tail -n {n} /root/stage2_mvp.log 2>/dev/null || echo '(no log yet)'")
    return out.strip()


def gpu_status(c):
    out, _ = run(c, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo '0,0'")
    try:
        parts = out.strip().split(',')
        return f"GPU={parts[0].strip()}%, {parts[1].strip()} MiB"
    except Exception:
        return "GPU?"


def main():
    print("=" * 70)
    print("阶段 2 SDD 真实数据验证 — Phase A MVP 云端编排")
    print("=" * 70)

    c = ssh_connect()
    print("✓ SSH 已连接")
    sftp = c.open_sftp()

    # [0] 确认云端状态
    print(f"\n[0] 确认云端状态: {gpu_status(c)}")
    ps, _ = run(c, "pgrep -f 'train.py' | head -3")
    if ps.strip():
        print(f"  [警告] 有训练进程: {ps.strip()}, 可能冲突, 但仍继续 (断点续跑会跳过已完成的)")
    else:
        print("  ✓ 无训练进程, 空闲")

    # 检查断点 marker
    done, _ = run(c, "test -f /root/STAGE2_MVP_DONE && echo DONE || echo NONE")
    if done.strip() == "DONE":
        print("\n  [断点] 检测到 STAGE2_MVP_DONE, 跳过执行, 直接下载结果")
        skip_exec = True
    else:
        skip_exec = False

    if not skip_exec:
        # 清理旧 marker
        run(c, "rm -f /root/STAGE2_MVP_DONE /root/STAGE2_MVP_FAILED")
        print("  ✓ 清理旧 marker")

        # [1] 上传文件
        print("\n[1] 上传代码 ...")
        # 确保远端 scripts_sdd 目录存在
        run(c, f"mkdir -p {REMOTE_ROOT}/scripts_sdd")
        uploads = [
            (os.path.join(LOCAL_BASE, "data", "gen_sdd_grid.py"),
             f"{REMOTE_ROOT}/data/gen_sdd_grid.py"),
            (os.path.join(LOCAL_BASE, "configs", "sdd_mamba2_30m.yaml"),
             f"{REMOTE_ROOT}/configs/sdd_mamba2_30m.yaml"),
            (os.path.join(LOCAL_BASE, "scripts_sdd", "download_sdd_annotations.py"),
             f"{REMOTE_ROOT}/scripts_sdd/download_sdd_annotations.py"),
        ]
        for local, remote in uploads:
            if os.path.exists(local):
                sftp.put(local, remote)
                print(f"  ✓ {os.path.basename(local)} → {remote}")
            else:
                print(f"  [警告] 本地文件不存在: {local}")

        # [2] 上传 shell 脚本
        sh_path = "/root/run_stage2_mvp.sh"
        with sftp.open(sh_path, 'w') as f:
            f.write(RUN_SH)
        run(c, f"chmod +x {sh_path}")
        print(f"\n[2] ✓ 上传 {sh_path}")

        # [3] nohup 启动
        run(c, f"nohup bash {sh_path} > /root/stage2_mvp_nohup.out 2>&1 &", timeout=10)
        time.sleep(5)
        print("\n[3] ✓ 已后台启动 Stage 2 MVP")
        init_log, _ = run(c, "tail -n 5 /root/stage2_mvp.log 2>/dev/null || echo '(启动中...)'")
        for line in init_log.strip().split('\n')[-5:]:
            if line:
                print(f"    {line}")

        # [4] 轮询等待完成
        print(f"\n[4] 等待 Stage 2 MVP 完成 (预计 ~30-60 min) ...")
        waited = 0
        while True:
            time.sleep(60)
            waited += 1
            stat, _ = run(c, "test -f /root/STAGE2_MVP_DONE && echo DONE || (test -f /root/STAGE2_MVP_FAILED && echo FAILED || echo RUNNING)")
            s = stat.strip()
            if s == "DONE":
                print(f"\n  ✓✓✓ Stage 2 MVP 完成! (等了 {waited} min)")
                break
            if s == "FAILED":
                print(f"\n  ✗✗✗ Stage 2 MVP 失败! (等了 {waited} min)")
                print("  查看 /root/stage2_mvp.log 诊断")
                fl, _ = run(c, "tail -n 30 /root/stage2_mvp.log")
                print(fl)
                c.close()
                return
            if waited % 2 == 0 or waited == 1:
                print(f"  [已等 {waited}min] {gpu_status(c)}")
                lt = tail_log(c, 3)
                for line in lt.split('\n')[-3:]:
                    if line:
                        print(f"    日志: {line}")

    # [5] 下载结果
    print("\n[5] 下载结果 ...")
    remote_results = f"{REMOTE_ROOT}/results_stage2"
    local_results = os.path.join(LOCAL_BASE, "results_stage2")

    # 下载训练目录
    print("  训练目录:")
    download_dir(sftp, f"{remote_results}/sdd_30m_seed0_10k",
                os.path.join(local_results, "sdd_30m_seed0_10k"))

    # 下载日志
    print("  日志:")
    download_file(sftp, "/root/stage2_mvp.log",
                  os.path.join(local_results, "stage2_mvp.log"))

    # [6] 最终确认
    print("\n" + "=" * 70)
    print("Stage 2 MVP 完成, 最终日志:")
    print("=" * 70)
    final, _ = run(c, "tail -n 20 /root/stage2_mvp.log")
    print(final)

    # [7] 解析并展示 decay 结果
    print("=" * 70)
    print("结果解析:")
    print("=" * 70)
    metrics_json = os.path.join(local_results, "sdd_30m_seed0_10k", "ood_metrics.json")
    if os.path.exists(metrics_json):
        import json
        with open(metrics_json) as f:
            m = json.load(f)
        print(f"  train_T: {m.get('train_T', '?')}, ood_T: {m.get('ood_T', '?')}")
        ch = m.get("changed_acc_curve", {})
        a100 = ch.get("100", None)
        a150 = ch.get("150", None)
        decay = m.get("ood_decay_pct", None)
        print(f"  changed_acc@100: {a100}")
        print(f"  changed_acc@150: {a150}")
        print(f"  ood_decay_pct: {decay:.4%}" if decay is not None else "  ood_decay_pct: N/A")
        print(f"  acc_final: {m.get('acc_final', '?')}")
        print(f"  changed_acc: {m.get('changed_acc', '?')}")
        # 初步判据
        if decay is not None and a100 is not None:
            print("\n  初步迹象:")
            if abs(decay) <= 0.05:
                print(f"    ✓ decay ∈ ±5% → SSM 在真实数据有不退化迹象, 可推进 Phase B")
            elif abs(decay) <= 0.10:
                print(f"    ~ decay ∈ ±10% → 临界, 需调参 (fps_stride/网格)")
            else:
                print(f"    ✗ decay > ±10% → SSM 在真实数据退化, 需机制分析")
        if a100 is not None and a100 < 0.30:
            print(f"  [警告] changed_acc@100={a100} < 0.30 非退化门控未通过, 模型未学会任务")
    else:
        print(f"  [警告] ood_metrics.json 不存在: {metrics_json}")

    sftp.close()
    c.close()
    print("\n✓ Stage 2 MVP 编排完成, 结果已下载到 results_stage2/")
    print("  云端保持开机 (用户指示)")


if __name__ == "__main__":
    main()
