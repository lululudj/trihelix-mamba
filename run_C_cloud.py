"""
C 方案: 1B 参数突破 OOM
上传改后的模型代码 + 1000m_ckpt config, 训练 500 步测试 OOM 是否突破
"""
import paramiko
import sys, time, json
from pathlib import Path

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL = Path(r"e:\three_chain_v3")
REMOTE = "/root/three_chain_v3"

def ssh_exec(c, cmd, timeout=30):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace'), stderr.read().decode('utf-8', errors='replace')

def upload(c, local, remote):
    sftp = c.open_sftp()
    sftp.put(str(local), remote)
    sftp.close()

print("=" * 70)
print("C 方案: 1B 参数突破 OOM (gradient checkpointing)")
print("=" * 70)
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15)
print("✓ SSH 已连接")

# ============ 上传文件 ============
print("\n[1] 上传 C 方案文件 ...")
# 改后的模型代码 (含 gradient checkpointing)
upload(c, LOCAL / "models" / "three_chain_mamba2.py", f"{REMOTE}/models/three_chain_mamba2.py")
upload(c, LOCAL / "utils.py", f"{REMOTE}/utils.py")
print("  ✓ models/three_chain_mamba2.py, utils.py")

# 1000M + checkpoint config
upload(c, LOCAL / "configs" / "matched_mamba2_1000m_ckpt.yaml",
       f"{REMOTE}/configs/matched_mamba2_1000m_ckpt.yaml")
print("  ✓ configs/matched_mamba2_1000m_ckpt.yaml")

# ============ 等待 D+A+B 完成 ============
print("\n[2] 等待 D+A+B 完成 ...")
max_wait = 60 * 60  # 最多等 60 分钟
waited = 0
while waited < max_wait:
    out, _ = ssh_exec(c, "grep 'DAB_ALL_DONE' /root/extreme_test.log 2>&1 || echo NOTDONE")
    if "DAB_ALL_DONE" in out:
        print(f"  ✓ D+A+B 已完成 (等了 {waited//60}min)")
        break
    out, _ = ssh_exec(c, "pgrep -f 'bash /root/run_extreme_dab.sh' > /dev/null && echo RUNNING || echo STOPPED")
    if "STOPPED" in out:
        out, _ = ssh_exec(c, "grep 'DAB_ALL_DONE' /root/extreme_test.log 2>&1 || echo NOTDONE")
        if "DAB_ALL_DONE" in out:
            print(f"  ✓ D+A+B 已完成")
            break
        else:
            print(f"  ⚠️ D+A+B 进程停止但未完成, 可能崩溃")
    time.sleep(60)
    waited += 60
    out, _ = ssh_exec(c, "tail -3 /root/extreme_test.log 2>&1")
    print(f"  [等 {waited//60}min] {out.strip()[:100]}")

# ============ 启动 C 训练 ============
print("\n[3] 启动 C: 1000M + gradient checkpointing 训练 ...")

script = r'''#!/bin/bash
LOG=/root/run_c_1000m.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%F %T')] === C: 1000M + gradient checkpointing ===" > $LOG
echo "测试 OOM 是否突破 ..." >> $LOG

# 先快速测试: 只跑 1 步看显存
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_1000m_ckpt.yaml \
    --max_steps 100 --batch_size 1 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_1000m_ckpt_seed0_100 >> $LOG 2>&1

TRAIN_RC=$?
echo "[$(date '+%F %T')] 训练结束 rc=$TRAIN_RC" >> $LOG

if [ $TRAIN_RC -eq 0 ]; then
    echo "✓ 1000M + checkpoint 训练成功! 开始 eval" >> $LOG
    python -u eval_ood.py \
        --checkpoint results/run_1000m_ckpt_seed0_100/final.pt \
        --config configs/matched_mamba2_1000m_ckpt.yaml \
        --data_root ./data/ood_T150 \
        --batch_size 1 --seed 0 \
        --out results/run_1000m_ckpt_seed0_100/ood_metrics.json >> $LOG 2>&1
    echo "[$(date '+%F %T')] eval 完成 rc=$?" >> $LOG
    rm -f results/run_1000m_ckpt_seed0_100/final.pt results/run_1000m_ckpt_seed0_100/best.pt
else
    echo "✗ 1000M + checkpoint 仍然 OOM" >> $LOG
    echo "尝试降级: d_model=4608 (~900M params)" >> $LOG
    # 降级方案: d_model=4608
    sed 's/d_model: 5120/d_model: 4608/' configs/matched_mamba2_1000m_ckpt.yaml > /tmp/900m_ckpt.yaml
    python -u train.py \
        --model three_chain_mamba2 \
        --config /tmp/900m_ckpt.yaml \
        --max_steps 100 --batch_size 1 --seed 0 \
        --data_root ./data_split_m3 \
        --out_dir results/run_900m_ckpt_seed0_100 >> $LOG 2>&1
    TRAIN_RC2=$?
    echo "[$(date '+%F %T')] 900M 降级训练 rc=$TRAIN_RC2" >> $LOG
    if [ $TRAIN_RC2 -eq 0 ]; then
        python -u eval_ood.py \
            --checkpoint results/run_900m_ckpt_seed0_100/final.pt \
            --config /tmp/900m_ckpt.yaml \
            --data_root ./data/ood_T150 \
            --batch_size 1 --seed 0 \
            --out results/run_900m_ckpt_seed0_100/ood_metrics.json >> $LOG 2>&1
        rm -f results/run_900m_ckpt_seed0_100/final.pt results/run_900m_ckpt_seed0_100/best.pt
    fi
fi

echo "[$(date '+%F %T')] === C_ALL_DONE ===" >> $LOG
'''

sftp = c.open_sftp()
with sftp.file("/root/run_c_1000m.sh", "w") as f:
    f.write(script)
sftp.close()
print("  ✓ run_c_1000m.sh 已上传")

# 启动
stdin, stdout, stderr = c.exec_command(
    "nohup setsid bash /root/run_c_1000m.sh > /dev/null 2>&1 & echo PID=$!"
)
print(f"  启动: {stdout.read().decode().strip()}")

# ============ 监控 ============
print("\n[4] 监控 C 训练 (每 60s) ...")
MAX_WAIT_MIN = 40
waited = 0
while waited < MAX_WAIT_MIN * 60:
    time.sleep(60)
    waited += 60
    out, _ = ssh_exec(c, "grep 'C_ALL_DONE' /root/run_c_1000m.log 2>&1 || echo NOTDONE")
    if "C_ALL_DONE" in out:
        print(f"\n[已等 {waited//60}min] ✓✓✓ C 方案完成!")
        break
    out, _ = ssh_exec(c, "tail -5 /root/run_c_1000m.log 2>&1")
    print(f"\n[已等 {waited//60}min] {out.strip()}")
    out, _ = ssh_exec(c, "pgrep -f 'bash /root/run_c_1000m.sh' > /dev/null && echo RUNNING || echo STOPPED")
    if "STOPPED" in out:
        out, _ = ssh_exec(c, "grep 'C_ALL_DONE' /root/run_c_1000m.log 2>&1 || echo NOTDONE")
        if "C_ALL_DONE" not in out:
            print("  ⚠️ 进程停止但未完成")
            out, _ = ssh_exec(c, "tail -20 /root/run_c_1000m.log")
            print(out)
        break

# ============ 下载结果 ============
print("\n[5] 下载 C 方案结果 ...")
LOCAL_OUT = LOCAL / "results_cloud"
sftp = c.open_sftp()
downloaded = 0
for remote_dir in ["run_1000m_ckpt_seed0_100", "run_900m_ckpt_seed0_100"]:
    local_dir = LOCAL_OUT / remote_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        for f in sftp.listdir(f"{REMOTE}/results/{remote_dir}"):
            if f.endswith('.json'):
                sftp.get(f"{REMOTE}/results/{remote_dir}/{f}", str(local_dir / f))
                downloaded += 1
                print(f"  ✓ {remote_dir}/{f}")
    except FileNotFoundError:
        print(f"  ⚠️ {remote_dir} 不存在")
# 下载日志
try:
    sftp.get("/root/run_c_1000m.log", str(LOCAL / "run_c_1000m.log"))
    print("  ✓ run_c_1000m.log")
except:
    pass
sftp.close()

# ============ 打印结果 ============
print("\n" + "=" * 70)
print("=== C 方案结果 ===")
print("=" * 70)
log_path = LOCAL / "run_c_1000m.log"
if log_path.exists():
    print(log_path.read_text(encoding='utf-8', errors='replace')[-2000:])

for result_file in sorted(LOCAL_OUT.rglob("ood_metrics.json")):
    if "1000m" in str(result_file) or "900m" in str(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                d = json.load(f)
            ch_curve = d.get('changed_acc_curve', {})
            ch100 = ch_curve.get('100')
            ch150 = ch_curve.get('150')
            if ch100 and ch150 and ch100 > 0:
                decay = (ch150 - ch100) / ch100 * 100
                print(f"  {result_file.parent.name}: ch@100={ch100:.4f}, ch@150={ch150:.4f}, decay={decay:+.1f}%")
        except Exception as e:
            print(f"  {result_file}: {e}")

c.close()
print("\n✓ C 方案完成")
sys.stdout.flush()
