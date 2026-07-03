"""阶段 3.3 云端编排器: SSM vs Transformer 信号保留对比

流程:
  1. 上传 probe_token_retention.py + configs/matched_transformer_30m.yaml
  2. 云端训练 30M SSM baseline (seed=2, 500步) — 旧 .pt 已删, 需重训供 probe
  3. 云端训练 30M Transformer (seed=2, 500步)
  4. eval_ood 两个模型 (拿 decay 对照)
  5. probe_token_retention 两个模型 (拿信号保留曲线)
  6. 下载结果 (.npz + ood_metrics.json + 日志)

预计云端耗时: ~10 min (训练 2×2min + eval 2×1min + probe 2×2min)
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

# 云端 shell 脚本: 训练 + eval + probe
RUN_SH = r"""#!/bin/bash
LOG=/root/stage3_3.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date +%H:%M:%S)] Stage 3.3 Start ===" >> $LOG

# ============================================================
# 1. 训练 30M SSM baseline (seed=2, 500步) — 供 probe (旧 .pt 已删)
# ============================================================
echo "[$(date +%H:%M:%S)] === 1. Train 30M SSM baseline ===" >> $LOG
python -u train.py --model three_chain_mamba2 \
    --config configs/matched_mamba2_30m.yaml \
    --max_steps 500 --batch_size 2 --seed 2 \
    --data_root ./data_split_m3 \
    --out_dir results_stage3/ssm_30m_seed2_500 >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] SSM train done" >> $LOG

# eval OOD T150
python -u eval_ood.py \
    --checkpoint results_stage3/ssm_30m_seed2_500/best.pt \
    --config configs/matched_mamba2_30m.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 2 \
    --out results_stage3/ssm_30m_seed2_500/ood_metrics.json >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] SSM eval done" >> $LOG

# ============================================================
# 2. 训练 30M Transformer (seed=2, 500步)
# ============================================================
echo "[$(date +%H:%M:%S)] === 2. Train 30M Transformer ===" >> $LOG
python -u train.py --model transformer \
    --config configs/matched_transformer_30m.yaml \
    --max_steps 500 --batch_size 2 --seed 2 \
    --data_root ./data_split_m3 \
    --out_dir results_stage3/transformer_30m_seed2_500 >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Transformer train done" >> $LOG

# eval OOD T150
python -u eval_ood.py \
    --checkpoint results_stage3/transformer_30m_seed2_500/best.pt \
    --config configs/matched_transformer_30m.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 2 \
    --out results_stage3/transformer_30m_seed2_500/ood_metrics.json >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Transformer eval done" >> $LOG

# ============================================================
# 3. Probe token retention: SSM
# ============================================================
echo "[$(date +%H:%M:%S)] === 3. Probe SSM retention ===" >> $LOG
python -u probe_token_retention.py \
    --checkpoint results_stage3/ssm_30m_seed2_500/best.pt \
    --config configs/matched_mamba2_30m.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
    --max_samples 50 --n_perturb 3 --label ssm_30m \
    --out results_stage3/retention_ssm_30m.npz >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] SSM probe done" >> $LOG

# ============================================================
# 4. Probe token retention: Transformer
# ============================================================
echo "[$(date +%H:%M:%S)] === 4. Probe Transformer retention ===" >> $LOG
python -u probe_token_retention.py \
    --checkpoint results_stage3/transformer_30m_seed2_500/best.pt \
    --config configs/matched_transformer_30m.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
    --max_samples 50 --n_perturb 3 --label transformer_30m \
    --out results_stage3/retention_transformer_30m.npz >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Transformer probe done" >> $LOG

# 清理大文件 (保留 best.pt 供后续可能的复测, 删 final.pt + optimizer)
rm -f results_stage3/ssm_30m_seed2_500/final.pt
rm -f results_stage3/transformer_30m_seed2_500/final.pt

touch /root/STAGE3_3_DONE
echo "[$(date +%H:%M:%S)] === STAGE3_3_DONE ===" >> $LOG
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
    out, _ = run(c, f"tail -n {n} /root/stage3_3.log 2>/dev/null || echo '(no log yet)'")
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
    print("阶段 3.3 云端编排: SSM vs Transformer 信号保留对比")
    print("=" * 70)

    c = ssh_connect()
    print("✓ SSH 已连接")
    sftp = c.open_sftp()

    # [0] 确认云端空闲
    print(f"\n[0] 确认云端状态: {gpu_status(c)}")
    ps, _ = run(c, "pgrep -f 'train.py' | head -3")
    if ps.strip():
        print(f"  [警告] 有训练进程: {ps.strip()}")
    else:
        print("  ✓ 无训练进程, 空闲")

    # 清理旧 marker
    run(c, "rm -f /root/STAGE3_3_DONE")
    print("  ✓ 清理旧 marker")

    # [1] 上传文件
    print("\n[1] 上传代码 ...")
    uploads = [
        (os.path.join(LOCAL_BASE, "probe_token_retention.py"),
         f"{REMOTE_ROOT}/probe_token_retention.py"),
        (os.path.join(LOCAL_BASE, "configs", "matched_transformer_30m.yaml"),
         f"{REMOTE_ROOT}/configs/matched_transformer_30m.yaml"),
        (os.path.join(LOCAL_BASE, "models", "baselines.py"),
         f"{REMOTE_ROOT}/models/baselines.py"),
        (os.path.join(LOCAL_BASE, "models", "__init__.py"),
         f"{REMOTE_ROOT}/models/__init__.py"),
    ]
    for local, remote in uploads:
        if os.path.exists(local):
            sftp.put(local, remote)
            print(f"  ✓ {os.path.basename(local)} → {remote}")
        else:
            print(f"  [警告] 本地文件不存在: {local}")

    # [2] 上传 shell 脚本
    sh_path = "/root/run_stage3_3.sh"
    with sftp.open(sh_path, 'w') as f:
        f.write(RUN_SH)
    run(c, f"chmod +x {sh_path}")
    print(f"\n[2] ✓ 上传 {sh_path}")

    # [3] nohup 启动
    run(c, f"nohup bash {sh_path} > /root/stage3_3_nohup.out 2>&1 &", timeout=10)
    time.sleep(3)
    print("\n[3] ✓ 已后台启动 Stage 3.3")
    init_log, _ = run(c, "tail -n 3 /root/stage3_3.log 2>/dev/null || echo '(启动中...)'")
    for line in init_log.strip().split('\n')[-3:]:
        if line:
            print(f"    {line}")

    # [4] 轮询等待完成
    print(f"\n[4] 等待 Stage 3.3 完成 (预计 ~10 min) ...")
    waited = 0
    while True:
        time.sleep(60)
        waited += 1
        stat, _ = run(c, "test -f /root/STAGE3_3_DONE && echo DONE || echo RUNNING")
        if stat.strip() == "DONE":
            print(f"\n  ✓✓✓ Stage 3.3 完成! (等了 {waited} min)")
            break
        if waited % 2 == 0 or waited == 1:
            print(f"  [已等 {waited}min] {gpu_status(c)}")
            lt = tail_log(c, 2)
            for line in lt.split('\n')[-2:]:
                if line:
                    print(f"    日志: {line}")

    # [5] 下载结果
    print("\n[5] 下载结果 ...")
    remote_results = f"{REMOTE_ROOT}/results_stage3"
    local_results = os.path.join(LOCAL_BASE, "results_stage3")

    # 下载 SSM 目录
    print("  SSM 30M:")
    download_dir(sftp, f"{remote_results}/ssm_30m_seed2_500",
                os.path.join(local_results, "ssm_30m_seed2_500"))

    # 下载 Transformer 目录
    print("  Transformer 30M:")
    download_dir(sftp, f"{remote_results}/transformer_30m_seed2_500",
                os.path.join(local_results, "transformer_30m_seed2_500"))

    # 下载 probe .npz
    print("  Probe npz:")
    download_file(sftp, f"{remote_results}/retention_ssm_30m.npz",
                  os.path.join(local_results, "retention_ssm_30m.npz"))
    download_file(sftp, f"{remote_results}/retention_transformer_30m.npz",
                  os.path.join(local_results, "retention_transformer_30m.npz"))

    # 下载日志
    print("  日志:")
    download_file(sftp, "/root/stage3_3.log",
                  os.path.join(local_results, "stage3_3.log"))

    # [6] 最终确认
    print("\n" + "=" * 70)
    print("Stage 3.3 完成, 最终日志:")
    print("=" * 70)
    final, _ = run(c, "tail -n 15 /root/stage3_3.log")
    print(final)

    sftp.close()
    c.close()
    print("\n✓ Stage 3.3 编排完成, 结果已下载到 results_stage3/")
    print("  下一步: 运行 summarize_stage3_3.py 生成对比图 + 报告")


if __name__ == "__main__":
    main()
