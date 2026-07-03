"""控制实验编排器: 实验1 (B@1000) + 实验3 (shuffle_labels) + 实验2 (1B多seed)
按 CONTROL_EXPERIMENTS_CHECKLIST.md 执行。
实验3 应早跑 (一票否决, 8min), 但实验1 也快 (12min) 不需改代码, 先跑实验1 拿结果。
顺序: 实验1 → 实验3 → 实验2 (最长)
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
LOCAL_BASE = r"e:\three_chain_v3\results_cloud"

# 云端 shell 脚本: 三阶段, 每阶段末尾 touch marker
RUN_SH = r"""#!/bin/bash
LOG=/root/control_experiments.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date +%H:%M:%S)] Control Experiments Start ===" >> $LOG

# ============================================================
# 实验 1: B (n_layers=4) @ 1000 步对照 (消除 B vs B2 步数混淆)
# 配置: matched_mamba2_30m_deep.yaml (d_model=768, n_layers=4)
# 与 B2 相同: 1000 步, lr=3e-4, warmup=500
# ============================================================
echo "[$(date +%H:%M:%S)] === Exp1: B n_layers=4 @1000 steps ===" >> $LOG
python -u train.py --model three_chain_mamba2 \
    --config configs/matched_mamba2_30m_deep.yaml \
    --max_steps 1000 --batch_size 2 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_30m_deep_n4_seed0_1000 >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Exp1 train done" >> $LOG

# eval T150
python -u eval_ood.py \
    --checkpoint results/run_30m_deep_n4_seed0_1000/final.pt \
    --config configs/matched_mamba2_30m_deep.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
    --extend_max_T 1024 \
    --out results/run_30m_deep_n4_seed0_1000/ood_A_T150.json >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Exp1 T150 eval done" >> $LOG

# eval T300
python -u eval_ood.py \
    --checkpoint results/run_30m_deep_n4_seed0_1000/final.pt \
    --config configs/matched_mamba2_30m_deep.yaml \
    --data_root ./data/ood_T300 --batch_size 1 --seed 0 \
    --extend_max_T 1024 \
    --out results/run_30m_deep_n4_seed0_1000/ood_A_T300.json >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Exp1 T300 eval done" >> $LOG

rm -f results/run_30m_deep_n4_seed0_1000/final.pt
rm -f results/run_30m_deep_n4_seed0_1000/best.pt
touch /root/EXP1_DONE
echo "[$(date +%H:%M:%S)] === EXP1_DONE ===" >> $LOG

# ============================================================
# 实验 3: 随机标签 sanity check (一票否决!)
# 30M + shuffle_labels, 500 步, eval T150
# 若 ch_acc>0.4 且 decay≈0 → 指标坏了, 所有定量结论作废
# ============================================================
echo "[$(date +%H:%M:%S)] === Exp3: 30M shuffle_labels sanity ===" >> $LOG
python -u train.py --model three_chain_mamba2 \
    --config configs/matched_mamba2_30m.yaml \
    --max_steps 500 --batch_size 4 --seed 0 \
    --shuffle_labels \
    --data_root ./data_split_m3 \
    --out_dir results/run_30m_shufflelabel_seed0_500 >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Exp3 train done" >> $LOG

python -u eval_ood.py \
    --checkpoint results/run_30m_shufflelabel_seed0_500/final.pt \
    --config configs/matched_mamba2_30m.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
    --out results/run_30m_shufflelabel_seed0_500/ood_metrics.json >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Exp3 eval done" >> $LOG

rm -f results/run_30m_shufflelabel_seed0_500/final.pt
rm -f results/run_30m_shufflelabel_seed0_500/best.pt
touch /root/EXP3_DONE
echo "[$(date +%H:%M:%S)] === EXP3_DONE ===" >> $LOG

# ============================================================
# 实验 2: 1B 多 seed (seed=1, 2) 验证 +0.28% 不是单 seed 幸运
# 配置: matched_mamba2_1000m_ckpt_long.yaml, 2000 步
# ============================================================
for SEED in 1 2; do
    echo "[$(date +%H:%M:%S)] === Exp2: 1B seed=${SEED} ===" >> $LOG
    python -u train.py --model three_chain_mamba2 \
        --config configs/matched_mamba2_1000m_ckpt_long.yaml \
        --max_steps 2000 --batch_size 1 --seed ${SEED} \
        --data_root ./data_split_m3 \
        --out_dir results/run_1000m_ckpt_long_seed${SEED}_2000 >> $LOG 2>&1
    echo "[$(date +%H:%M:%S)] Exp2 seed=${SEED} train done" >> $LOG

    python -u eval_ood.py \
        --checkpoint results/run_1000m_ckpt_long_seed${SEED}_2000/final.pt \
        --config configs/matched_mamba2_1000m_ckpt_long.yaml \
        --data_root ./data/ood_T150 --batch_size 1 --seed ${SEED} \
        --out results/run_1000m_ckpt_long_seed${SEED}_2000/ood_metrics.json >> $LOG 2>&1
    echo "[$(date +%H:%M:%S)] Exp2 seed=${SEED} eval done" >> $LOG

    rm -f results/run_1000m_ckpt_long_seed${SEED}_2000/final.pt
    rm -f results/run_1000m_ckpt_long_seed${SEED}_2000/best.pt
done
touch /root/EXP2_DONE
echo "[$(date +%H:%M:%S)] === EXP2_DONE ===" >> $LOG

echo "[$(date +%H:%M:%S)] === ALL CONTROL EXPERIMENTS DONE ===" >> $LOG
"""

PHASES = [
    {
        'name': '实验1 B@1000步对照',
        'marker': '/root/EXP1_DONE',
        'remote_dir': f'{REMOTE_ROOT}/results/run_30m_deep_n4_seed0_1000',
        'local_dir': os.path.join(LOCAL_BASE, 'run_30m_deep_n4_seed0_1000'),
        'eta_min': 12,
    },
    {
        'name': '实验3 随机标签sanity (一票否决)',
        'marker': '/root/EXP3_DONE',
        'remote_dir': f'{REMOTE_ROOT}/results/run_30m_shufflelabel_seed0_500',
        'local_dir': os.path.join(LOCAL_BASE, 'run_30m_shufflelabel_seed0_500'),
        'eta_min': 8,
    },
    {
        'name': '实验2 1B多seed (seed1,2)',
        'marker': '/root/EXP2_DONE',
        'remote_dir': None,  # 两个目录
        'local_dir': None,
        'eta_min': 95,
        'sub_dirs': [
            ('run_1000m_ckpt_long_seed1_2000', 'run_1000m_ckpt_long_seed1_2000'),
            ('run_1000m_ckpt_long_seed2_2000', 'run_1000m_ckpt_long_seed2_2000'),
        ],
    },
]


def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    return c


def run(c, cmd, timeout=30):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors='replace'), err.read().decode(errors='replace')


def download_dir(sftp, remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    try:
        for entry in sftp.listdir(remote_dir):
            remote_path = f"{remote_dir}/{entry}"
            local_path = os.path.join(local_dir, entry)
            try:
                sftp.get(remote_path, local_path)
                st = sftp.stat(remote_path)
                print(f"    ✓ {entry} ({st.st_size} bytes)")
                n += 1
            except Exception as e:
                # 可能是子目录, 跳过
                print(f"    - 跳过 {entry}: {e}")
    except FileNotFoundError:
        print(f"    [警告] 远端目录不存在: {remote_dir}")
    return n


def tail_log(c, n=3):
    out, _ = run(c, f"tail -n {n} /root/control_experiments.log 2>/dev/null || echo '(no log yet)'")
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
    print("控制实验编排器: 实验1 + 实验3 + 实验2")
    print("=" * 70)

    c = ssh_connect()
    print("✓ SSH 已连接")
    sftp = c.open_sftp()

    # [0] 确认云端空闲
    print("\n[0] 确认云端状态 ...")
    print(f"  {gpu_status(c)}")
    ps, _ = run(c, "pgrep -f 'train.py' | head -3")
    if ps.strip():
        print(f"  [警告] 有训练进程: {ps.strip()}")
    else:
        print("  ✓ 无训练进程, 空闲")

    # [1] 上传修改后的 dataset.py, train.py
    print("\n[1] 上传修改后的代码 ...")
    sftp.put(r"e:\three_chain_v3\data\dataset.py", f"{REMOTE_ROOT}/data/dataset.py")
    print(f"  ✓ dataset.py → {REMOTE_ROOT}/data/dataset.py")
    sftp.put(r"e:\three_chain_v3\train.py", f"{REMOTE_ROOT}/train.py")
    print(f"  ✓ train.py → {REMOTE_ROOT}/train.py")

    # [2] 上传 shell 脚本
    sh_path = "/root/run_control.sh"
    with sftp.open(sh_path, 'w') as f:
        f.write(RUN_SH)
    run(c, f"chmod +x {sh_path}")
    print(f"\n[2] ✓ 上传 {sh_path}")

    # [3] 清理旧 marker
    run(c, "rm -f /root/EXP1_DONE /root/EXP3_DONE /root/EXP2_DONE")
    print("[3] ✓ 清理旧 marker")

    # [4] nohup 启动
    run(c, f"nohup bash {sh_path} > /root/control_nohup.out 2>&1 &", timeout=10)
    time.sleep(3)
    print("\n[4] ✓ 已后台启动控制实验")
    init_log, _ = run(c, "tail -n 5 /root/control_experiments.log 2>/dev/null || echo '(启动中...)'")
    print("  初始日志:")
    for line in init_log.strip().split('\n')[-5:]:
        print(f"    {line}")

    # [5] 轮询每个 phase
    for i, ph in enumerate(PHASES, 1):
        print(f"\n[{i+4}] 等待 {ph['name']} 完成 (marker: {ph['marker']}, 预计 {ph['eta_min']} min) ...")
        waited = 0
        while True:
            time.sleep(60)
            waited += 1
            stat, _ = run(c, f"test -f {ph['marker']} && echo DONE || echo RUNNING")
            if stat.strip() == "DONE":
                print(f"\n  ✓✓✓ {ph['name']} 完成! (等了 {waited} min)")
                break
            # 每 2 min 打印状态
            if waited % 2 == 0 or waited == 1:
                print(f"  [已等 {waited}min] {gpu_status(c)}")
                lt = tail_log(c, 2)
                for line in lt.split('\n')[-2:]:
                    if line:
                        print(f"    日志: {line}")

        # 立即下载结果
        print(f"\n  下载 {ph['name']} 结果 ...")
        if ph['remote_dir']:
            remote = ph['remote_dir']
            local = ph['local_dir']
            n = download_dir(sftp, remote, local)
            print(f"  共下载 {n} 个文件")
        elif ph.get('sub_dirs'):
            total = 0
            for rsub, lsub in ph['sub_dirs']:
                remote = f"{REMOTE_ROOT}/results/{rsub}"
                local = os.path.join(LOCAL_BASE, lsub)
                print(f"    子目录 {rsub}:")
                n = download_dir(sftp, remote, local)
                total += n
            print(f"  共下载 {total} 个文件")

        # 显示日志末尾
        lt = tail_log(c, 3)
        print("  日志末尾:")
        for line in lt.split('\n')[-3:]:
            if line:
                print(f"    {line}")

    # [6] 最终确认
    print("\n" + "=" * 70)
    print("全部控制实验完成, 最终确认")
    print("=" * 70)
    markers, _ = run(c, "ls -la /root/EXP1_DONE /root/EXP3_DONE /root/EXP2_DONE 2>&1")
    print(markers)
    final, _ = run(c, "tail -n 10 /root/control_experiments.log")
    print("最终日志:")
    print(final)

    sftp.close()
    c.close()
    print("\n✓ 控制实验编排完成, 所有结果已下载到 results_cloud/")


if __name__ == "__main__":
    main()
