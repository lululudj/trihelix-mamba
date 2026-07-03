"""
云端极限测试 v2: D + A + B
策略: 训练一个 max_T=1024 的 100M checkpoint 给 D+A 共用, B 单独训
"""
import paramiko
import os, sys, time, json
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

def upload_dir(c, local_dir, remote_dir):
    sftp = c.open_sftp()
    try:
        sftp.mkdir(remote_dir)
    except IOError:
        pass
    count = 0
    for f in Path(local_dir).iterdir():
        if f.is_file() and f.suffix == '.npz':
            sftp.put(str(f), f"{remote_dir}/{f.name}")
            count += 1
    sftp.close()
    return count

# ============ 连接 ============
print("=" * 70)
print("云端极限测试 v2: D + A + B")
print("=" * 70)
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15)
print("✓ SSH 已连接")

# ============ 上传文件 ============
print("\n[1] 上传文件 ...")
upload(c, LOCAL / "utils.py", f"{REMOTE}/utils.py")
upload(c, LOCAL / "eval_ood.py", f"{REMOTE}/eval_ood.py")
print("  ✓ utils.py, eval_ood.py")

# 上传 100M_extreme config (max_T=1024)
upload(c, LOCAL / "configs" / "matched_mamba2_extreme.yaml",
       f"{REMOTE}/configs/matched_mamba2_extreme.yaml")
# 上传 30M_deep config (n_layers=4) - 需要基于 30M 改
# 先用 100M_extreme 的思路, 基于 30M 加 max_T=1024
upload(c, LOCAL / "configs" / "matched_mamba2_30m_deep.yaml",
       f"{REMOTE}/configs/matched_mamba2_30m_deep.yaml")
print("  ✓ configs: extreme + 30m_deep")

# 上传 T300/T500 数据
print("  上传 ood_T300/T500 数据 ...")
n1 = upload_dir(c, LOCAL / "data" / "ood_T300", f"{REMOTE}/data/ood_T300")
n2 = upload_dir(c, LOCAL / "data" / "ood_T500", f"{REMOTE}/data/ood_T500")
print(f"  ✓ T300: {n1} 文件, T500: {n2} 文件")

# ============ 创建执行脚本 ============
# 关键: 100M_extreme 用 extreme config (max_T=1024, d_model=256 太小)
# 实际应该用 100M config + max_T=1024. 让我创建一个 100m_maxT1024 config
script_100m_maxT = """# 100M + max_T=1024 (极限测试用)
task:
  cell_types: 16
  action_dim: 5
model:
  d_model: 1280
  n_layers: 2
  bind_heads: 4
  fusion_hidden: 1536
  enable_m3: false
  enable_jepa: false
  jepa_weight: 0.3
  max_T: 1024
eagle:
  tau_init: 1.0e9
  tau_target: 1.0
  tau_warmup_steps: 2000
  tau_anneal_steps: 10000
data:
  root: ./data_cache
  num_train: 8000
  num_val: 1000
  num_test: 1000
  N_choices: [6, 8, 12]
  K_choices: [4, 8, 12]
  T_choices: [30, 60, 100]
  p_transfer_choices: [0.0, 0.3, 0.6]
  scenario_mix: {random: 0.5, goal_directed: 0.3, adversarial: 0.2}
train:
  batch_size: 2
  max_steps: 50000
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 500
  loss_aux_weight: 0.3
  grad_clip: 1.0
  seed: 0
eval:
  acc_steps: [1, 5, 10, 30, 60, 100]
  ood_T: [30, 60, 100, 150, 300, 500]
  ood_N: [6, 8, 12, 16]
  tau_sweep: [0.5, 1.0, 2.0, 5.0, 1.0e9]
"""
sftp = c.open_sftp()
with sftp.file(f"{REMOTE}/configs/matched_mamba2_100m_maxT1024.yaml", "w") as f:
    f.write(script_100m_maxT)
# 30M_deep 也加 max_T=1024
script_30m_deep_maxT = """# 30M deep + max_T=1024
task:
  cell_types: 16
  action_dim: 5
model:
  d_model: 768
  n_layers: 4
  bind_heads: 4
  fusion_hidden: 1536
  enable_m3: false
  enable_jepa: false
  jepa_weight: 0.3
  max_T: 1024
eagle:
  tau_init: 1.0e9
  tau_target: 1.0
  tau_warmup_steps: 2000
  tau_anneal_steps: 10000
data:
  root: ./data_cache
  num_train: 8000
  num_val: 1000
  num_test: 1000
  N_choices: [6, 8, 12]
  K_choices: [4, 8, 12]
  T_choices: [30, 60, 100]
  p_transfer_choices: [0.0, 0.3, 0.6]
  scenario_mix: {random: 0.5, goal_directed: 0.3, adversarial: 0.2}
train:
  batch_size: 2
  max_steps: 50000
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 500
  loss_aux_weight: 0.3
  grad_clip: 1.0
  seed: 0
eval:
  acc_steps: [1, 5, 10, 30, 60, 100]
  ood_T: [30, 60, 100, 150, 300]
  ood_N: [6, 8, 12, 16]
  tau_sweep: [0.5, 1.0, 2.0, 5.0, 1.0e9]
"""
with sftp.file(f"{REMOTE}/configs/matched_mamba2_30m_deep_maxT.yaml", "w") as f:
    f.write(script_30m_deep_maxT)
sftp.close()
print("  ✓ configs: 100m_maxT1024 + 30m_deep_maxT")

# 主执行脚本
script = r'''#!/bin/bash
LOG=/root/extreme_test.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%F %T')] === 极限测试 D+A+B 开始 ===" > $LOG

# ====== Phase 1: 训练共用 checkpoint (100M max_T=1024) ======
echo "" >> $LOG
echo "[$(date '+%F %T')] === Phase 1: 训练 100M max_T=1024 ===" >> $LOG
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_100m_maxT1024.yaml \
    --max_steps 500 --batch_size 2 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_100m_maxT1024_seed0_500 >> $LOG 2>&1
echo "[$(date '+%F %T')] 100M 训练完成" >> $LOG

# ====== Phase 2: D + A 评估 (共用 100M checkpoint) ======
echo "" >> $LOG
echo "[$(date '+%F %T')] === Phase 2: D + A 评估 ===" >> $LOG

# D: 噪声/遮蔽鲁棒性
echo "--- D: 噪声/遮蔽鲁棒性 ---" >> $LOG
python -u eval_ood.py --checkpoint results/run_100m_maxT1024_seed0_500/final.pt \
    --config configs/matched_mamba2_100m_maxT1024.yaml \
    --data_root ./data/ood_T200_noise --batch_size 1 --seed 0 \
    --out results/run_100m_maxT1024_seed0_500/ood_D_T200noise.json >> $LOG 2>&1
echo "[$(date '+%F %T')] D1 T200noise 完成" >> $LOG

python -u eval_ood.py --checkpoint results/run_100m_maxT1024_seed0_500/final.pt \
    --config configs/matched_mamba2_100m_maxT1024.yaml \
    --data_root ./data/ood_T250_mask --batch_size 1 --seed 0 \
    --out results/run_100m_maxT1024_seed0_500/ood_D_T250mask.json >> $LOG 2>&1
echo "[$(date '+%F %T')] D2 T250mask 完成" >> $LOG

# A: 超长 OOD 外推
echo "--- A: 超长 OOD 外推 ---" >> $LOG
python -u eval_ood.py --checkpoint results/run_100m_maxT1024_seed0_500/final.pt \
    --config configs/matched_mamba2_100m_maxT1024.yaml \
    --data_root ./data/ood_T300 --batch_size 1 --seed 0 \
    --out results/run_100m_maxT1024_seed0_500/ood_A_T300.json >> $LOG 2>&1
echo "[$(date '+%F %T')] A1 T300 完成" >> $LOG

python -u eval_ood.py --checkpoint results/run_100m_maxT1024_seed0_500/final.pt \
    --config configs/matched_mamba2_100m_maxT1024.yaml \
    --data_root ./data/ood_T500 --batch_size 1 --seed 0 \
    --out results/run_100m_maxT1024_seed0_500/ood_A_T500.json >> $LOG 2>&1
echo "[$(date '+%F %T')] A2 T500 完成" >> $LOG

# 顺便也评估 T150 做对照
python -u eval_ood.py --checkpoint results/run_100m_maxT1024_seed0_500/final.pt \
    --config configs/matched_mamba2_100m_maxT1024.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
    --out results/run_100m_maxT1024_seed0_500/ood_T150.json >> $LOG 2>&1
echo "[$(date '+%F %T')] T150 对照完成" >> $LOG

# 删除 100M checkpoint (省空间, B 不需要)
rm -f results/run_100m_maxT1024_seed0_500/final.pt results/run_100m_maxT1024_seed0_500/best.pt
echo "[$(date '+%F %T')] 100M checkpoint 已删除" >> $LOG

# ====== Phase 3: B 超深模型训练 + 评估 ======
echo "" >> $LOG
echo "[$(date '+%F %T')] === Phase 3: B 超深模型 n_layers=4 ===" >> $LOG
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_30m_deep_maxT.yaml \
    --max_steps 500 --batch_size 2 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_30m_deep_n4_maxT1024_seed0_500 >> $LOG 2>&1
echo "[$(date '+%F %T')] B 训练完成, 开始 eval" >> $LOG

# B eval: T150, T300
python -u eval_ood.py --checkpoint results/run_30m_deep_n4_maxT1024_seed0_500/final.pt \
    --config configs/matched_mamba2_30m_deep_maxT.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
    --out results/run_30m_deep_n4_maxT1024_seed0_500/ood_T150.json >> $LOG 2>&1
echo "[$(date '+%F %T')] B T150 完成" >> $LOG

python -u eval_ood.py --checkpoint results/run_30m_deep_n4_maxT1024_seed0_500/final.pt \
    --config configs/matched_mamba2_30m_deep_maxT.yaml \
    --data_root ./data/ood_T300 --batch_size 1 --seed 0 \
    --out results/run_30m_deep_n4_maxT1024_seed0_500/ood_A_T300.json >> $LOG 2>&1
echo "[$(date '+%F %T')] B T300 完成" >> $LOG

# 删除 B checkpoint
rm -f results/run_30m_deep_n4_maxT1024_seed0_500/final.pt results/run_30m_deep_n4_maxT1024_seed0_500/best.pt
echo "[$(date '+%F %T')] B checkpoint 已删除" >> $LOG

echo "" >> $LOG
echo "[$(date '+%F %T')] === DAB_ALL_DONE ===" >> $LOG
'''

sftp = c.open_sftp()
with sftp.file("/root/run_extreme_dab.sh", "w") as f:
    f.write(script)
sftp.close()
print("  ✓ run_extreme_dab.sh 已上传")

# ============ 启动执行 ============
print("\n[2] 启动 D+A+B 极限测试 ...")
stdin, stdout, stderr = c.exec_command(
    "nohup setsid bash /root/run_extreme_dab.sh > /dev/null 2>&1 & echo PID=$!"
)
print(f"  启动: {stdout.read().decode().strip()}")

time.sleep(8)
out, _ = ssh_exec(c, "cat /root/extreme_test.log 2>&1 | head -5")
print(f"  初始日志:\n{out.strip()}")

# ============ 监控 ============
print("\n[3] 监控执行进度 (每 60s) ...")
MAX_WAIT_MIN = 80
waited = 0
while waited < MAX_WAIT_MIN * 60:
    time.sleep(60)
    waited += 60
    out, _ = ssh_exec(c, "grep 'DAB_ALL_DONE' /root/extreme_test.log 2>&1 || echo NOTDONE")
    if "DAB_ALL_DONE" in out:
        print(f"\n[已等 {waited//60}min] ✓✓✓ D+A+B 全部完成!")
        break
    out, _ = ssh_exec(c, "tail -6 /root/extreme_test.log 2>&1")
    print(f"\n[已等 {waited//60}min] {out.strip()}")
    out, _ = ssh_exec(c, "pgrep -f 'bash /root/run_extreme_dab.sh' > /dev/null && echo RUNNING || echo STOPPED")
    if "STOPPED" in out:
        print("  ⚠️ 进程停止")
        out, _ = ssh_exec(c, "tail -20 /root/extreme_test.log")
        print(out)
        if "DAB_ALL_DONE" not in out:
            print("  可能崩溃, 检查错误")
        break

# ============ 下载结果 ============
print("\n[4] 下载所有结果 ...")
LOCAL_OUT = LOCAL / "results_cloud"
sftp = c.open_sftp()
downloaded = 0
for remote_dir in ["run_100m_maxT1024_seed0_500", "run_30m_deep_n4_maxT1024_seed0_500"]:
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
sftp.close()
print(f"\n  共下载 {downloaded} 个结果文件")

# ============ 打印最终结果 ============
print("\n" + "=" * 70)
print("=== 极限测试结果汇总 ===")
print("=" * 70)
for result_file in sorted(LOCAL_OUT.rglob("ood_*.json")):
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        name = result_file.stem
        parent = result_file.parent.name
        ch_curve = d.get('changed_acc_curve', {})
        max_t = d.get('max_T', '?')
        n = d.get('n_samples', '?')
        ch100 = ch_curve.get('100')
        ch_final_key = str(max_t) if str(max_t) in ch_curve else str(sorted([int(k) for k in ch_curve.keys()])[-1] if ch_curve else '?')
        ch_final = ch_curve.get(ch_final_key)
        if ch100 and ch_final and ch100 > 0:
            decay = (ch_final - ch100) / ch100 * 100
            print(f"  {parent}/{name}: T={max_t}, ch@100={ch100:.4f}, ch@{max_t}={ch_final:.4f}, decay={decay:+.1f}%")
        else:
            print(f"  {parent}/{name}: T={max_t}, n={n}")
    except Exception as e:
        print(f"  {result_file}: {e}")

c.close()
print("\n✓ D+A+B 完成, 可继续 C 方案 (1B 突破 OOM)")
sys.stdout.flush()
