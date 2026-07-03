"""检查 stage2_full 完整状态: marker + log tail + 两个 ood_metrics.json + 进程.
用 IP 直连绕过 paramiko DNS 问题.
"""
import time
import json
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('123.127.15.155', port=50472, username='root',
          password='i9D1S9IoRMLR', timeout=30)


def run(cmd, label=None):
    if label:
        print(f"\n=== {label} ===")
    _, o, e = c.exec_command(cmd, timeout=60)
    out = o.read().decode(errors='replace')
    err = e.read().decode(errors='replace')
    if out:
        print(out, end='' if out.endswith('\n') else '\n')
    if err.strip():
        print(f"[stderr] {err}", end='' if err.endswith('\n') else '\n')
    return out


# 1. 完成 marker
run('ls -la /root/STAGE2_FULL_DONE 2>&1; '
    'echo ---; '
    'cat /root/STAGE2_FULL_DONE 2>/dev/null',
    label='[1] 完成标记')

# 2. 进程状态
run('pgrep -af "train.py|eval_ood.py|stage2_full" 2>&1 | grep -v grep',
    label='[2] 运行中的相关进程')

# 3. log 尾部 30 行
run('tail -30 /root/stage2_full.log 2>/dev/null',
    label='[3] stage2_full.log 尾部 30 行')

# 4. 阶段分隔标记
run('grep -nE "=== |^\\[" /root/stage2_full.log 2>/dev/null | tail -25',
    label='[4] 阶段标记')

# 5. 10k OOD metrics
out = run('cat /root/three_chain_v3/results_stage2/sdd_30m_seed0_10k/'
          'ood_metrics.json 2>/dev/null',
          label='[5] SSM 10k OOD metrics')
if out.strip():
    try:
        m = json.loads(out)
        ch100 = m.get('changed_acc_curve', {}).get('100', None)
        ch150 = m.get('changed_acc_curve', {}).get('150', None)
        print(f"\n  >>> ch@100={ch100}, ch@150={ch150}, "
              f"decay_pct={m.get('ood_decay_pct')}, "
              f"decay_abs={m.get('ood_decay_100_to_150')}")
    except Exception as ex:
        print(f"  [parse fail] {ex}")

# 6. shuffle OOD metrics
out = run('cat /root/three_chain_v3/results_stage2/sdd_shuffle_3k/'
          'ood_metrics.json 2>/dev/null',
          label='[6] shuffle 3k OOD metrics')
if out.strip():
    try:
        m = json.loads(out)
        ch100 = m.get('changed_acc_curve', {}).get('100', None)
        ch150 = m.get('changed_acc_curve', {}).get('150', None)
        print(f"\n  >>> ch@100={ch100}, ch@150={ch150}, "
              f"decay_pct={m.get('ood_decay_pct')}, "
              f"decay_abs={m.get('ood_decay_100_to_150')}")
    except Exception as ex:
        print(f"  [parse fail] {ex}")

# 7. 结果目录
run('ls -la /root/three_chain_v3/results_stage2/ 2>&1',
    label='[7] results_stage2 目录')

# 8. GPU 状态
run('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total '
    '--format=csv,noheader 2>&1',
    label='[8] GPU 状态')

c.close()
print("\n=== 检查完成 ===")
