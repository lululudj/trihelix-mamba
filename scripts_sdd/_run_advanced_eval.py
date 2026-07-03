"""上传 eval_ood_advanced.py 到云端, 在 SSM 10k + shuffle checkpoint 上跑, 下载对比.
用 IP 直连.
"""
import os
import time
import json
import paramiko

IP = '123.127.15.155'
PORT = 50472
USER = 'root'
PWD = 'i9D1S9IoRMLR'
LOCAL_SCRIPT = r'e:\three_chain_v3\scripts_sdd\eval_ood_advanced.py'
REMOTE_DIR = '/root/three_chain_v3'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(IP, port=PORT, username=USER, password=PWD, timeout=30)

# 0. 确认 GPU 空闲
print("=== [0] 确认 GPU 空闲 ===")
_, o, _ = c.exec_command(
    'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; '
    'pgrep -af "train.py|eval" | grep -v grep', timeout=30)
print(o.read().decode(errors='replace'))

# 1. 上传脚本
print("=== [1] 上传 eval_ood_advanced.py ===")
sftp = c.open_sftp()
sftp.put(LOCAL_SCRIPT, f'{REMOTE_DIR}/eval_ood_advanced.py')
print("  已上传")
sftp.close()

# 2. 确认 OOD 数据 + checkpoint 存在
print("\n=== [2] 确认数据 + checkpoint ===")
_, o, _ = c.exec_command(
    f'ls {REMOTE_DIR}/data/ood_T150_sdd/scen_*.npz 2>/dev/null | wc -l; '
    f'echo ---; '
    f'ls -la {REMOTE_DIR}/results_stage2/sdd_30m_seed0_10k/best.pt '
    f'{REMOTE_DIR}/results_stage2/sdd_shuffle_3k/best.pt 2>&1', timeout=30)
print(o.read().decode(errors='replace'))

PY = '/root/miniconda3/bin/python'

# 3. 跑 SSM 10k 高级评估
print("=== [3] 跑 SSM 10k 高级评估 (~2min) ===")
cmd_ssm = (f'cd {REMOTE_DIR} && '
           f'{PY} -u eval_ood_advanced.py '
           f'--checkpoint results_stage2/sdd_30m_seed0_10k/best.pt '
           f'--config configs/sdd_mamba2_30m.yaml '
           f'--data_root ./data/ood_T150_sdd '
           f'--batch_size 1 --label SSM_10k 2>&1')
_, o, e = c.exec_command(cmd_ssm, timeout=300)
out = o.read().decode(errors='replace')
err = e.read().decode(errors='replace')
print(out)
if err.strip():
    print(f"[stderr] {err}")

# 4. 跑 shuffle 高级评估
print("\n=== [4] 跑 shuffle 高级评估 (~2min) ===")
cmd_sh = (f'cd {REMOTE_DIR} && '
          f'{PY} -u eval_ood_advanced.py '
          f'--checkpoint results_stage2/sdd_shuffle_3k/best.pt '
          f'--config configs/sdd_mamba2_30m.yaml '
          f'--data_root ./data/ood_T150_sdd '
          f'--batch_size 1 --label shuffle 2>&1')
_, o, e = c.exec_command(cmd_sh, timeout=300)
out = o.read().decode(errors='replace')
err = e.read().decode(errors='replace')
print(out)
if err.strip():
    print(f"[stderr] {err}")

# 5. 下载两个 advanced metrics
print("\n=== [5] 下载 advanced metrics ===")
LOCAL_RES = r'e:\three_chain_v3\results_stage2'
for sub in ['sdd_30m_seed0_10k', 'sdd_shuffle_3k']:
    sftp = c.open_sftp()
    remote = f'{REMOTE_DIR}/results_stage2/{sub}/ood_metrics_advanced.json'
    local = os.path.join(LOCAL_RES, sub, 'ood_metrics_advanced.json')
    try:
        sftp.get(remote, local)
        print(f"  下载 {sub}/ood_metrics_advanced.json ✓")
    except Exception as ex:
        print(f"  [失败] {sub}: {ex}")
    sftp.close()

# 6. 打印对比
print("\n=== [6] SSM 10k vs shuffle 高级指标对比 ===")
for sub, label in [('sdd_30m_seed0_10k', 'SSM_10k'),
                    ('sdd_shuffle_3k', 'shuffle')]:
    path = os.path.join(LOCAL_RES, sub, 'ood_metrics_advanced.json')
    if os.path.exists(path):
        try:
            m = json.load(open(path))
            c = m['curves']
            print(f"\n[{label}] (random agent baseline={m['random_baseline_agent']:.4f})")
            print(f"{'t':>5} {'ch_acc':>8} {'enter':>8} {'leave':>8} "
                  f"{'agent_id':>9} {'pos_iou':>8}")
            for t in [50, 100, 120, 150]:
                if str(t) in c:
                    x = c[str(t)]
                    print(f"{t:>5} {x['changed_acc']:>8.4f} {x['enter_acc']:>8.4f} "
                          f"{x['leave_acc']:>8.4f} {x['agent_id_acc']:>9.4f} "
                          f"{x['position_iou']:>8.4f}")
        except Exception as ex:
            print(f"  [{label}] 解析失败: {ex}")
    else:
        print(f"  [{label}] 文件不存在")

c.close()
print("\n=== 全部完成 ===")
