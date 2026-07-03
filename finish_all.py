"""
finish_all.py - 一站式收尾
1. 下载所有已完成实验的 JSON 结果到本地 results_cloud/
2. 上传 tail_rerun.sh 到云端
3. nohup 启动 tail_rerun.sh（等 700M/1000M 完成后重跑 2 个失败实验）
4. 打印当前汇总
"""
import paramiko, os, json, time, sys

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

LOCAL_DIR = r"e:\three_chain_v3\results_cloud"

# 所有需要下载 JSON 的实验目录
ALL_EXPS = [
    "run_100m_hta_seed0_500",
    "run_100m_hta_seed2_500",
    "run_100m_seed0_500",
    "run_100m_seed1",
    "run_100m_seed2",
    "run_100m_seed3_500",
    "run_100m_seed4_500",
    "run_300m_seed0_2k",
    "run_300m_seed1_2k",
    "run_30m_seed1",
    "run_30m_seed2",
    "run_30m_seed3",
    "run_30m_seed4",
    "run_three_chain_mamba2_30m_seed0",
    "run_700m_seed0_2k",
    "run_1000m_seed0_2k",
]

# tail_rerun.sh: 等 700M/1000M 完成后重跑 2 个失败实验
TAIL_RERUN_SH = r"""#!/bin/bash
LOG=/root/tail_rerun.log
echo "[$(date '+%F %T')] tail_rerun 启动" > $LOG
echo "[1] 等待 run_700m_1000m.sh 完成..." >> $LOG
while pgrep -f run_700m_1000m.sh > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date '+%F %T')] run_700m_1000m.sh 已结束" >> $LOG

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# 检查 700M 和 1000M 的结果
echo "[2] 检查 700M/1000M 结果..." >> $LOG
for d in run_700m_seed0_2k run_1000m_seed0_2k; do
    if [ -f "results/$d/ood_metrics.json" ]; then
        echo "  $d: OOD 完成" >> $LOG
    elif [ -f "results/$d/summary.json" ]; then
        echo "  $d: 训练完成但无 OOD, 补跑 eval" >> $LOG
        cfg="configs/matched_mamba2_700m.yaml"
        [ "$d" = "run_1000m_seed0_2k" ] && cfg="configs/matched_mamba2_1000m.yaml"
        python -u eval_ood.py --checkpoint results/$d/final.pt --config $cfg --data_root ./data/ood_T150 --batch_size 4 --seed 0 --out results/$d/ood_metrics.json >> $LOG 2>&1
        rm -f results/$d/best.pt
    elif [ -f "results/$d/final.pt" ]; then
        echo "  $d: 有 final.pt 无 summary, 补跑 eval" >> $LOG
        cfg="configs/matched_mamba2_700m.yaml"
        [ "$d" = "run_1000m_seed0_2k" ] && cfg="configs/matched_mamba2_1000m.yaml"
        python -u eval_ood.py --checkpoint results/$d/final.pt --config $cfg --data_root ./data/ood_T150 --batch_size 4 --seed 0 --out results/$d/ood_metrics.json >> $LOG 2>&1
        rm -f results/$d/best.pt
    else
        echo "  $d: 无结果文件 (可能 OOM)" >> $LOG
    fi
done

# 重跑 300m_seed2_2k
echo "[3] 重跑 300m_seed2_2k..." >> $LOG
echo "[$(date '+%F %T')] 开始训练" >> $LOG
python -u train.py --model three_chain_mamba2 --config configs/matched_mamba2_300m.yaml \
    --max_steps 2000 --batch_size 1 --seed 2 --data_root ./data_split_m3 \
    --out_dir results/run_300m_seed2_2k >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] 训练结束 rc=$TRAIN_RC" >> $LOG
if [ $TRAIN_RC -eq 0 ] && [ -f results/run_300m_seed2_2k/final.pt ]; then
    rm -f results/run_300m_seed2_2k/best.pt
    python -u eval_ood.py --checkpoint results/run_300m_seed2_2k/final.pt \
        --config configs/matched_mamba2_300m.yaml --data_root ./data/ood_T150 \
        --batch_size 4 --seed 2 --out results/run_300m_seed2_2k/ood_metrics.json >> $LOG 2>&1
    echo "  300m_seed2_2k: OOD 完成" >> $LOG
else
    echo "  300m_seed2_2k: 训练失败 rc=$TRAIN_RC" >> $LOG
fi
rm -f results/run_300m_seed2_2k/best.pt

# 重跑 100m_hta_seed1_500
echo "[4] 重跑 100m_hta_seed1_500..." >> $LOG
echo "[$(date '+%F %T')] 开始训练" >> $LOG
python -u train.py --model three_chain_mamba2_hta --config configs/matched_mamba2_100m.yaml \
    --max_steps 500 --batch_size 2 --seed 1 --data_root ./data_split_m3 \
    --out_dir results/run_100m_hta_seed1_500 >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] 训练结束 rc=$TRAIN_RC" >> $LOG
if [ $TRAIN_RC -eq 0 ] && [ -f results/run_100m_hta_seed1_500/final.pt ]; then
    rm -f results/run_100m_hta_seed1_500/best.pt
    python -u eval_ood.py --checkpoint results/run_100m_hta_seed1_500/final.pt \
        --config configs/matched_mamba2_100m.yaml --data_root ./data/ood_T150 \
        --batch_size 4 --seed 1 --out results/run_100m_hta_seed1_500/ood_metrics.json >> $LOG 2>&1
    echo "  100m_hta_seed1_500: OOD 完成" >> $LOG
else
    echo "  100m_hta_seed1_500: 训练失败 rc=$TRAIN_RC" >> $LOG
fi
rm -f results/run_100m_hta_seed1_500/best.pt

echo "[5] 最终磁盘:" >> $LOG
df -h / >> $LOG 2>&1

echo "[$(date '+%F %T')] ALL_TAIL_DONE" >> $LOG
"""

def run(client, cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err

def download_file(sftp, remote_path, local_path):
    try:
        sftp.get(remote_path, local_path)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"    下载失败 {remote_path}: {e}")
        return False

def parse_ood(path):
    """解析本地 ood_metrics.json, 返回 (ch@100, ch@150, decay%)"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        curve = d.get("changed_acc_curve", {})
        c100 = curve.get("100")
        c150 = curve.get("150")
        if c100 is not None and c150 is not None:
            decay = (c150 - c100) / c100 * 100
            return c100, c150, decay
    except Exception:
        pass
    return None, None, None

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print("[1] 连接 ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)
    print("    ✓ 连上了")

    # ---- 2. 下载已完成结果 ----
    print("\n[2] 下载已完成实验的 JSON ...")
    sftp = client.open_sftp()
    downloaded = 0
    for exp in ALL_EXPS:
        remote_dir = f"/root/three_chain_v3/results/{exp}"
        local_dir = os.path.join(LOCAL_DIR, exp)
        os.makedirs(local_dir, exist_ok=True)
        for fname in ["ood_metrics.json", "summary.json", "log.jsonl"]:
            ok = download_file(sftp, f"{remote_dir}/{fname}", os.path.join(local_dir, fname))
            if ok:
                downloaded += 1
    sftp.close()
    print(f"    ✓ 下载 {downloaded} 个文件到 {LOCAL_DIR}")

    # ---- 3. 上传 tail_rerun.sh ----
    print("\n[3] 上传 tail_rerun.sh ...")
    sftp = client.open_sftp()
    with sftp.open("/root/tail_rerun.sh", "w") as f:
        f.write(TAIL_RERUN_SH)
    sftp.close()
    run(client, "chmod +x /root/tail_rerun.sh")
    print("    ✓ tail_rerun.sh 已上传")

    # ---- 4. 检查是否已有 tail_rerun 在跑 ----
    out, _ = run(client, "pgrep -f tail_rerun.sh || echo NONE")
    if "NONE" in out:
        print("\n[4] 启动 tail_rerun.sh (nohup) ...")
        # nohup 后台运行, 不阻塞 SSH
        run(client, "nohup bash /root/tail_rerun.sh > /root/tail_rerun_nohup.log 2>&1 &", timeout=10)
        time.sleep(2)
        out2, _ = run(client, "pgrep -f tail_rerun.sh && echo RUNNING || echo NOT_RUNNING")
        print(f"    ✓ tail_rerun.sh 状态: {out2.strip()}")
    else:
        print(f"\n[4] tail_rerun.sh 已在运行 (pid={out.strip()})")

    # ---- 5. 打印当前汇总 ----
    print("\n[5] 当前结果汇总 (本地已下载的 JSON):")
    print(f"{'实验':<35} {'ch@100':>8} {'ch@150':>8} {'decay':>8} {'状态':<10}")
    print("-" * 75)
    for exp in ALL_EXPS:
        local_path = os.path.join(LOCAL_DIR, exp, "ood_metrics.json")
        if os.path.exists(local_path):
            c100, c150, decay = parse_ood(local_path)
            if c100 is not None:
                print(f"{exp:<35} {c100:>8.4f} {c150:>8.4f} {decay:>+7.1f}% {'✓ 完成':<10}")
            else:
                print(f"{exp:<35} {'?':>8} {'?':>8} {'?':>8} {'解析失败':<10}")
        else:
            # 检查是否有 summary.json (训练完但没 OOD)
            sp = os.path.join(LOCAL_DIR, exp, "summary.json")
            if os.path.exists(sp):
                print(f"{exp:<35} {'-':>8} {'-':>8} {'-':>8} {'训练完无OOD':<10}")
            else:
                print(f"{exp:<35} {'-':>8} {'-':>8} {'-':>8} {'未完成':<10}")

    client.close()
    print(f"\n✓ 完成. tail_rerun.sh 在云端后台运行中, 预计 20-30 分钟后全部完成")
    print(f"  监控: python check_tail.py")
    print(f"  最终下载: python download_final.py")

if __name__ == "__main__":
    main()
