"""
全自动完成剩余实验:
1. 等 700M 训练+eval 完成
2. 重训 100m_hta_seed1 (500步, ~5分钟)
3. eval 100m_hta_seed1
4. 下载全部结果
5. 打印最终汇总
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

def check_700m_done(c):
    out, _ = run(c, "grep 'RERUN_700M_V2_DONE' /root/run_700m_v2.log 2>/dev/null || echo NOTFOUND")
    return "NOTFOUND" not in out

def get_700m_progress(c):
    out, _ = run(c, "tail -5 /root/run_700m_v2.log 2>/dev/null")
    return out.strip()

# 上传 100m_hta_seed1 训练+eval 脚本
RUN_HTA_SH = r"""#!/bin/bash
LOG=/root/run_hta_seed1.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

echo "[$(date '+%F %T')] === 100m_hta_seed1 重训 ===" > $LOG

# 等 700M 完成
echo "[0] 等待 700M 完成..." >> $LOG
while pgrep -f 'bash /root/run_700m_v2.sh' > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date '+%F %T')] 700M 已结束" >> $LOG

# 清理
pkill -f "train.py" 2>/dev/null || true
pkill -f "eval_ood.py" 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $LOG 2>&1
df -h / | tail -1 >> $LOG 2>&1

# 训练 100m_hta_seed1 (500步)
echo "[$(date '+%F %T')] 训练 100m_hta_seed1 (500步, batch=2)" >> $LOG
python -u train.py --model three_chain_mamba2_hta --config configs/matched_mamba2_100m.yaml \
    --max_steps 500 --batch_size 2 --seed 1 --data_root ./data_split_m3 \
    --out_dir results/run_100m_hta_seed1_500 >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] 训练结束 rc=$TRAIN_RC" >> $LOG

if [ $TRAIN_RC -eq 0 ] && [ -f results/run_100m_hta_seed1_500/final.pt ]; then
    rm -f results/run_100m_hta_seed1_500/best.pt
    echo "[$(date '+%F %T')] eval" >> $LOG
    python -u eval_ood.py --checkpoint results/run_100m_hta_seed1_500/final.pt \
        --config configs/matched_mamba2_100m.yaml --data_root ./data/ood_T150 \
        --batch_size 1 --seed 1 --out results/run_100m_hta_seed1_500/ood_metrics.json >> $LOG 2>&1
    EVAL_RC=$?
    echo "[$(date '+%F %T')] eval 结束 rc=$EVAL_RC" >> $LOG
    rm -f results/run_100m_hta_seed1_500/final.pt results/run_100m_hta_seed1_500/best.pt
else
    echo "[$(date '+%F %T')] 训练失败" >> $LOG
fi

df -h / | tail -1 >> $LOG 2>&1
echo "[$(date '+%F %T')] === HTA_SEED1_DONE ===" >> $LOG
"""

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 全自动完成剩余实验 ===")
    print(f"结果下载到: {LOCAL_DIR}\n")

    c = connect()
    print("✓ 已连接\n")

    # 1. 上传 100m_hta_seed1 脚本
    print("[1] 上传 100m_hta_seed1 训练脚本 ...")
    sftp = c.open_sftp()
    with sftp.open("/root/run_hta_seed1.sh", "w") as f:
        f.write(RUN_HTA_SH)
    sftp.close()
    run(c, "chmod +x /root/run_hta_seed1.sh")
    print("    ✓ run_hta_seed1.sh 已上传")

    # 2. 启动 (它会等 700M 完成后自动开始) — 但先检查是否已在运行
    print("\n[2] 检查 run_hta_seed1.sh 是否已运行 ...")
    out, _ = run(c, "pgrep -af 'bash /root/run_hta_seed1.sh' | grep -v grep || echo NONE")
    if "NONE" in out:
        print("    未运行, 启动 ...")
        run(c, "nohup setsid bash /root/run_hta_seed1.sh > /root/run_hta_seed1_nohup.log 2>&1 < /dev/null &", t=10)
        time.sleep(3)
        out, _ = run(c, "pgrep -af 'bash /root/run_hta_seed1.sh' | grep -v grep || echo NONE")
        print(f"    进程: {out.strip()}")
    else:
        print(f"    已在运行: {out.strip()}")

    # 3. 监控
    print("\n[3] 开始监控 (每 90s 检查一次) ...")
    start = time.time()
    MAX_WAIT = 5400  # 90 分钟

    while time.time() - start < MAX_WAIT:
        elapsed = int(time.time() - start)

        # 700M 状态
        done_700m = check_700m_done(c)
        prog_700m = get_700m_progress(c)

        # hta_seed1 状态
        out, _ = run(c, "grep 'HTA_SEED1_DONE' /root/run_hta_seed1.log 2>/dev/null || echo NOTFOUND")
        done_hta = "NOTFOUND" not in out
        out, _ = run(c, "tail -5 /root/run_hta_seed1.log 2>/dev/null")
        prog_hta = out.strip()

        # 结果检查
        out, _ = run(c, "cat /root/three_chain_v3/results/run_700m_seed0_2k/ood_metrics.json 2>/dev/null | head -1")
        r700m = bool(out.strip())
        out, _ = run(c, "cat /root/three_chain_v3/results/run_100m_hta_seed1_500/ood_metrics.json 2>/dev/null | head -1")
        rhta = bool(out.strip())

        print(f"\n[已等 {elapsed}s]")
        print(f"  700M: {'✓完成' if done_700m else '运行中'} | OOD结果: {'有' if r700m else '无'}")
        if prog_700m:
            for line in prog_700m.split("\n")[-3:]:
                print(f"    {line}")
        print(f"  hta_seed1: {'✓完成' if done_hta else '等待/运行中'} | OOD结果: {'有' if rhta else '无'}")
        if prog_hta:
            for line in prog_hta.split("\n")[-3:]:
                print(f"    {line}")

        if done_700m and done_hta:
            print("\n✓✓✓ 全部完成!")
            break

        time.sleep(90)
        c.close()
        c = connect()

    # 4. 下载所有结果
    print("\n[4] 下载所有结果 ...")
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
    print(f"    ✓ 下载 {count} 个文件")

    # 5. 打印最终汇总
    print("\n" + "="*70)
    print("=== 最终实验汇总 ===")
    print("="*70)
    print(f"{'实验':<40} {'ch@100':>8} {'ch@150':>8} {'decay':>8}")
    print("-"*70)

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

    # 按规模分组统计
    print("\n=== 按规模汇总 ===")
    groups = {
        "baseline (3.26M)": [],
        "30M (27M)": [],
        "100M (72M)": [],
        "100M HTA (72M)": [],
        "300M (182M)": [],
        "700M (~725M)": [],
    }
    for name, a100, a150, decay in all_results:
        if "700m" in name:
            groups["700M (~725M)"].append(decay)
        elif "300m" in name:
            groups["300M (182M)"].append(decay)
        elif "hta" in name:
            groups["100M HTA (72M)"].append(decay)
        elif "100m" in name:
            groups["100M (72M)"].append(decay)
        elif "30m" in name:
            groups["30M (27M)"].append(decay)
        else:
            groups["baseline (3.26M)"].append(decay)

    # 排除旧的 100 步结果 (非 _500 或 _2k 后缀的)
    for label, decays in groups.items():
        if decays:
            avg = sum(decays) / len(decays)
            print(f"  {label}: {len(decays)} seeds, avg decay={avg:+.1f}%")
        else:
            print(f"  {label}: 无结果")

    c.close()
    print("\n✓ 全部完成, 可以关机了!")

if __name__ == "__main__":
    main()
