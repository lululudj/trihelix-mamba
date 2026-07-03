"""重新跑 nexus 的 3 个 advanced eval (sys.path 修复后).

上传修复后的 eval_ood_advanced.py + eval_negative_shadow.py,
重跑: SSM advanced + shuffle advanced + SSM negative_shadow,
下载结果到本地.
"""
import os
import time
import paramiko
from pathlib import Path

HOST = "123.127.15.155"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_ROOT = Path(r"e:\three_chain_v3")
REMOTE_DIR = "/root/three_chain_v3"
PY = "/root/miniconda3/bin/python"


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    sftp = c.open_sftp()

    # 1. 上传修复后的 eval 脚本
    for fn in ["eval_ood_advanced.py", "eval_negative_shadow.py"]:
        local = LOCAL_ROOT / "scripts_sdd" / fn
        remote = f"{REMOTE_DIR}/scripts_sdd/{fn}"
        sftp.put(str(local), remote)
        print(f"[1] 上传修复版 {fn}")

    # 2. 确认 GPU 空闲
    _, o, _ = c.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null')
    print(f"[2] GPU: {o.read().decode(errors='replace').strip()}")
    _, o, _ = c.exec_command('pgrep -af "train.py|eval_" 2>/dev/null | grep -v grep')
    print(f"[3] 进程: {o.read().decode(errors='replace').strip() or '(无)'}")

    # 3. 串行重跑 3 个 advanced eval
    evals = [
        ("6a. eval_ood_advanced SSM",
         f"{PY} -u scripts_sdd/eval_ood_advanced.py "
         f"--checkpoint results_stage2/nexus_30m_seed0_10k/best.pt "
         f"--config configs/sdd_mamba2_30m_nexus.yaml "
         f"--data_root ./data/ood_T150_sdd_nexus "
         f"--out results_stage2/nexus_30m_seed0_10k/ood_metrics_advanced.json"),
        ("6b. eval_ood_advanced shuffle",
         f"{PY} -u scripts_sdd/eval_ood_advanced.py "
         f"--checkpoint results_stage2/nexus_shuffle_3k/best.pt "
         f"--config configs/sdd_mamba2_30m_nexus.yaml "
         f"--data_root ./data/ood_T150_sdd_nexus "
         f"--out results_stage2/nexus_shuffle_3k/ood_metrics_advanced.json"),
        ("6c. eval_negative_shadow SSM",
         f"{PY} -u scripts_sdd/eval_negative_shadow.py "
         f"--checkpoint results_stage2/nexus_30m_seed0_10k/best.pt "
         f"--config configs/sdd_mamba2_30m_nexus.yaml "
         f"--data_root ./data/ood_T150_sdd_nexus "
         f"--out results_stage2/nexus_30m_seed0_10k/negative_shadow.json"),
    ]

    for name, cmd in evals:
        print(f"\n[4] 跑 {name}...")
        full = f"cd {REMOTE_DIR} && export CUDA_HOME=/usr/local/cuda && {cmd} 2>&1"
        _, o, e = c.exec_command(full, timeout=600)
        out = o.read().decode('utf-8', errors='replace')
        err = e.read().decode('utf-8', errors='replace')
        # 打印尾部结果
        tail = '\n'.join(out.strip().split('\n')[-15:])
        print(tail)
        if err.strip():
            print(f"  STDERR: {err.strip()[-500:]}")
        time.sleep(2)

    # 4. 下载结果到本地
    print("\n[5] 下载结果到本地...")
    downloads = [
        (f"{REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/ood_metrics_advanced.json",
         LOCAL_ROOT / "results_stage2" / "nexus_30m_seed0_10k" / "ood_metrics_advanced.json"),
        (f"{REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/negative_shadow.json",
         LOCAL_ROOT / "results_stage2" / "nexus_30m_seed0_10k" / "negative_shadow.json"),
        (f"{REMOTE_DIR}/results_stage2/nexus_shuffle_3k/ood_metrics_advanced.json",
         LOCAL_ROOT / "results_stage2" / "nexus_shuffle_3k" / "ood_metrics_advanced.json"),
        # 基础 metrics 也补下
        (f"{REMOTE_DIR}/results_stage2/nexus_30m_seed0_10k/ood_metrics.json",
         LOCAL_ROOT / "results_stage2" / "nexus_30m_seed0_10k" / "ood_metrics.json"),
        (f"{REMOTE_DIR}/results_stage2/nexus_shuffle_3k/ood_metrics.json",
         LOCAL_ROOT / "results_stage2" / "nexus_shuffle_3k" / "ood_metrics.json"),
    ]
    for r, l in downloads:
        try:
            l.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(r, str(l))
            print(f"  OK {l.name}")
        except IOError as e:
            print(f"  FAIL {l.name}: {e}")

    sftp.close()
    c.close()
    print("\n✓ 3 个 advanced eval 重跑 + 下载完成")


if __name__ == "__main__":
    main()
