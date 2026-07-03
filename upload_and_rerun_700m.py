"""
上传修复后的 eval_ood.py + 启动 700M 重训脚本
"""
import paramiko, time

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

RERUN_700M_SH = r"""#!/bin/bash
LOG=/root/rerun_700m.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

echo "[$(date '+%F %T')] === 700M 重训 ===" > $LOG

# 0. 等 final_rerun.sh 完成
echo "[0] 等待 final_rerun.sh 完成..." >> $LOG
while pgrep -f 'bash /root/final_rerun.sh' > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date '+%F %T')] final_rerun.sh 已结束" >> $LOG

# 1. kill 残留进程 (保留 jupyter/tensorboard)
echo "[1] 清理残留进程..." >> $LOG
pkill -f "train.py" 2>/dev/null || true
pkill -f "eval_ood.py" 2>/dev/null || true
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $LOG 2>&1
df -h / >> $LOG 2>&1

# 2. 重新训练 700M (2000 步)
echo "[2] 训练 700M (2000 步, batch=1)..." >> $LOG
echo "[$(date '+%F %T')] 开始训练" >> $LOG
python -u train.py --model three_chain_mamba2 --config configs/matched_mamba2_700m.yaml \
    --max_steps 2000 --batch_size 1 --seed 0 --data_root ./data_split_m3 \
    --out_dir results/run_700m_seed0_2k >> $LOG 2>&1
TRAIN_RC=$?
echo "[$(date '+%F %T')] 训练结束 rc=$TRAIN_RC" >> $LOG

if [ $TRAIN_RC -eq 0 ] && [ -f results/run_700m_seed0_2k/final.pt ]; then
    rm -f results/run_700m_seed0_2k/best.pt
    # 3. eval (用修复后的 eval_ood.py, 有 no_grad)
    echo "[3] eval (batch=1, no_grad)..." >> $LOG
    python -u eval_ood.py --checkpoint results/run_700m_seed0_2k/final.pt \
        --config configs/matched_mamba2_700m.yaml --data_root ./data/ood_T150 \
        --batch_size 1 --seed 0 --out results/run_700m_seed0_2k/ood_metrics.json >> $LOG 2>&1
    EVAL_RC=$?
    if [ $EVAL_RC -eq 0 ]; then
        echo "  700M eval 完成!" >> $LOG
    else
        echo "  700M eval 失败 rc=$EVAL_RC" >> $LOG
    fi
else
    echo "  700M 训练失败 rc=$TRAIN_RC, 无法 eval" >> $LOG
fi

# 4. 清理大文件
rm -f results/run_700m_seed0_2k/best.pt results/run_700m_seed0_2k/final.pt
df -h / >> $LOG 2>&1

echo "[$(date '+%F %T')] === RERUN_700M_DONE ===" >> $LOG
"""

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

def main():
    print("[1] 连接 ...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)
    print("    ✓ 连上了")

    # 检查云端 eval_ood.py 是否有 no_grad
    print("\n[2] 检查云端 eval_ood.py 是否有 @torch.no_grad() ...")
    out, _ = run(c, "grep -n 'no_grad' /root/three_chain_v3/eval_ood.py | head -5")
    if out.strip():
        print(f"    云端已有: {out.strip()}")
    else:
        print("    云端没有 no_grad! 需要上传修复版")

    # 上传修复后的 eval_ood.py
    print("\n[3] 上传修复后的 eval_ood.py ...")
    sftp = c.open_sftp()
    sftp.put(r"e:\three_chain_v3\eval_ood.py", "/root/three_chain_v3/eval_ood.py")
    print("    ✓ eval_ood.py 已上传")

    # 验证
    out, _ = run(c, "grep -n 'no_grad' /root/three_chain_v3/eval_ood.py | head -5")
    print(f"    验证: {out.strip()}")

    # 上传 rerun_700m.sh
    print("\n[4] 上传 rerun_700m.sh ...")
    with sftp.open("/root/rerun_700m.sh", "w") as f:
        f.write(RERUN_700M_SH)
    sftp.close()
    run(c, "chmod +x /root/rerun_700m.sh")
    print("    ✓ rerun_700m.sh 已上传")

    # 检查是否已有 rerun_700m 在跑
    print("\n[5] 检查是否已有 rerun_700m 在跑 ...")
    out, _ = run(c, "pgrep -af 'bash /root/rerun_700m.sh' | grep -v grep || echo NONE")
    print(f"    {out.strip()}")

    if "NONE" in out:
        print("\n[6] 启动 rerun_700m.sh (nohup) ...")
        run(c, "setsid bash /root/rerun_700m.sh > /root/rerun_700m_nohup.log 2>&1 < /dev/null &", t=10)
        time.sleep(3)
        out, _ = run(c, "pgrep -af 'bash /root/rerun_700m.sh' | grep -v grep || echo NONE")
        print(f"    进程: {out.strip()}")
    else:
        print("    已在运行, 不重启")

    # 当前状态
    print("\n[7] 当前 final_rerun 进度 ...")
    out, _ = run(c, "tail -5 /root/final_rerun.log 2>/dev/null")
    print(out.strip())

    c.close()
    print("\n✓ rerun_700m.sh 已启动, 会等 final_rerun 完成后自动重训 700M")
    print("  预计: final_rerun ~10 分钟 + 700M 训练 ~17 分钟 + eval ~3 分钟 = ~30 分钟")

if __name__ == "__main__":
    main()
