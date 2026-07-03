"""阶段 3.2 消融实验云端编排器

流程:
1. SSH 连接 AutoDL
2. 上传修改的代码 (three_chain_mamba2.py, utils.py, 4 config, run_stage3_ablation.py)
3. nohup 后台运行 run_stage3_ablation.py
4. 轮询等待 4 个 ablation 完成 (检查 ood_metrics.json)
5. 下载结果到本地 results_stage3/ablation/

预估: 4 模型 × (500步 train ~12min + eval ~3min) ≈ 1 小时
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

# 需要上传的文件 (local_path, remote_path)
UPLOAD_FILES = [
    (r"models\three_chain_mamba2.py", f"{REMOTE_ROOT}/models/three_chain_mamba2.py"),
    (r"utils.py", f"{REMOTE_ROOT}/utils.py"),
    (r"configs\ablation_no_spatial.yaml", f"{REMOTE_ROOT}/configs/ablation_no_spatial.yaml"),
    (r"configs\ablation_no_temporal.yaml", f"{REMOTE_ROOT}/configs/ablation_no_temporal.yaml"),
    (r"configs\ablation_no_causal.yaml", f"{REMOTE_ROOT}/configs/ablation_no_causal.yaml"),
    (r"configs\ablation_no_all.yaml", f"{REMOTE_ROOT}/configs/ablation_no_all.yaml"),
    (r"run_stage3_ablation.py", f"{REMOTE_ROOT}/run_stage3_ablation.py"),
]

# 云端 shell 脚本
RUN_SH = r"""#!/bin/bash
LOG=/root/stage3_ablation.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date +%H:%M:%S)] Stage 3.2 Ablation Start ===" >> $LOG
python -u run_stage3_ablation.py >> $LOG 2>&1
echo "=== [$(date +%H:%M:%S)] Stage 3.2 Ablation ALL DONE ===" >> $LOG
touch /root/STAGE3_DONE
"""

# 4 个 ablation (name, remote_dir, eta_min)
ABLATIONS = [
    ("no_spatial",  f"{REMOTE_ROOT}/results_stage3/ablation/no_spatial",  15),
    ("no_temporal", f"{REMOTE_ROOT}/results_stage3/ablation/no_temporal", 15),
    ("no_causal",   f"{REMOTE_ROOT}/results_stage3/ablation/no_causal",   15),
    ("no_all",      f"{REMOTE_ROOT}/results_stage3/ablation/no_all",      15),
]


def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    return c


def run(c, cmd, timeout=30):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors='replace').strip(), err.read().decode(errors='replace').strip()


def gpu_status(c):
    out, _ = run(c, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo '0,0'")
    try:
        parts = out.strip().split(',')
        return f"GPU={parts[0].strip()}%, {parts[1].strip()} MiB"
    except Exception:
        return "GPU?"


def tail_log(c, n=5):
    out, _ = run(c, f"tail -n {n} /root/stage3_ablation.log 2>/dev/null || echo '(no log yet)'")
    return out.strip()


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
                print(f"    - 跳过 {entry}: {e}")
    except FileNotFoundError:
        print(f"    [警告] 远端目录不存在: {remote_dir}")
    return n


def main():
    print("=" * 70)
    print("阶段 3.2 消融实验云端编排器")
    print("=" * 70)

    # [0] 连接 + 检查状态
    print("\n[0] 连接 AutoDL ...")
    c = ssh_connect()
    print("  ✓ SSH 已连接")
    sftp = c.open_sftp()

    print(f"  {gpu_status(c)}")
    ps, _ = run(c, "pgrep -f 'train.py' | head -3")
    if ps.strip():
        print(f"  [警告] 有训练进程: {ps.strip()}")
        # 不退出, 可能是之前的残留, 先 kill
        run(c, "pkill -f 'train.py' 2>/dev/null; sleep 2")
        print("  [INFO] 已清理旧训练进程")
    else:
        print("  ✓ 无训练进程, 空闲")

    # [1] 上传代码
    print("\n[1] 上传修改的代码 ...")
    for local_rel, remote_path in UPLOAD_FILES:
        local_path = os.path.join(LOCAL_BASE, local_rel)
        if not os.path.exists(local_path):
            print(f"  [警告] 本地文件不存在: {local_path}")
            continue
        sftp.put(local_path, remote_path)
        print(f"  ✓ {local_rel} → {remote_path}")

    # [2] 上传 shell 脚本
    print("\n[2] 上传运行脚本 ...")
    sh_path = "/root/run_stage3_ablation.sh"
    with sftp.open(sh_path, 'w') as f:
        f.write(RUN_SH)
    run(c, f"chmod +x {sh_path}")
    print(f"  ✓ {sh_path}")

    # [3] 清理旧 marker + 旧结果
    print("\n[3] 清理旧状态 ...")
    run(c, "rm -f /root/STAGE3_DONE")
    run(c, f"rm -rf {REMOTE_ROOT}/results_stage3/ablation")
    run(c, f"mkdir -p {REMOTE_ROOT}/results_stage3/ablation")
    print("  ✓ 已清理")

    # [4] nohup 后台启动
    print("\n[4] 后台启动消融实验 ...")
    run(c, f"nohup bash {sh_path} > /root/stage3_nohup.out 2>&1 &", timeout=10)
    time.sleep(5)
    print("  ✓ 已后台启动")
    init_log = tail_log(c, 3)
    print("  初始日志:")
    for line in init_log.split('\n')[-3:]:
        if line:
            print(f"    {line}")

    # [5] 轮询每个 ablation
    print("\n[5] 等待 4 个 ablation 完成 ...")
    for name, remote_dir, eta_min in ABLATIONS:
        print(f"\n  --- 等待 {name} (预计 {eta_min} min) ---")
        waited = 0
        while True:
            time.sleep(60)
            waited += 1
            # 检查 ood_metrics.json 是否存在
            ood_file = f"{remote_dir}/ood_metrics.json"
            stat, _ = run(c, f"test -f {ood_file} && echo DONE || echo RUNNING")
            if stat.strip() == "DONE":
                print(f"\n  ✓✓✓ {name} 完成! (等了 {waited} min)")
                # 读 decay
                try:
                    decay_raw, _ = run(c, f"python -c \"import json; m=json.load(open('{ood_file}')); print('{{:.4f}}'.format(m.get('ood_decay_pct',0)*100))\"")
                    ch_raw, _ = run(c, f"python -c \"import json; m=json.load(open('{ood_file}')); print('{{:.4f}}'.format(m.get('changed_acc',0)))\"")
                    print(f"      decay={float(decay_raw):+.2f}%, changed_acc@150={float(ch_raw):.4f}")
                except Exception as e:
                    print(f"      [读取 decay 失败: {e}]")
                break
            # 每 2 min 打印状态
            if waited % 2 == 0 or waited == 1:
                print(f"  [已等 {waited}min] {gpu_status(c)}")
                lt = tail_log(c, 3)
                for line in lt.split('\n')[-3:]:
                    if line:
                        print(f"    日志: {line}")

        # 检查是否整体完成 (提前退出)
        done_check, _ = run(c, "test -f /root/STAGE3_DONE && echo ALL_DONE || echo CONTINUE")
        if done_check.strip() == "ALL_DONE":
            print("\n  ✓ 全部 ablation 已完成 (检测到 STAGE3_DONE marker)")
            # 下载剩余的
            for n2, rd2, _ in ABLATIONS:
                ood2 = f"{rd2}/ood_metrics.json"
                st2, _ = run(c, f"test -f {ood2} && echo YES || echo NO")
                if st2.strip() == "YES" and n2 != name:
                    print(f"  {n2}: 已完成")
            break

    # [6] 下载所有结果
    print("\n[6] 下载所有 ablation 结果 ...")
    total = 0
    for name, remote_dir, _ in ABLATIONS:
        local_dir = os.path.join(LOCAL_BASE, "results_stage3", "ablation", name)
        print(f"\n  下载 {name}:")
        n = download_dir(sftp, remote_dir, local_dir)
        total += n
    print(f"\n  共下载 {total} 个文件")

    # 下载 progress.log + nohup.out
    for log_file in ["/root/stage3_ablation.log", "/root/stage3_nohup.out"]:
        try:
            local_log = os.path.join(LOCAL_BASE, "results_stage3", "ablation",
                                     os.path.basename(log_file))
            sftp.get(log_file, local_log)
            print(f"  ✓ {log_file} → {local_log}")
        except Exception as e:
            print(f"  - 跳过 {log_file}: {e}")

    # [7] 最终确认
    print("\n" + "=" * 70)
    print("全部消融实验完成, 最终确认")
    print("=" * 70)
    final_log = tail_log(c, 15)
    print("最终日志:")
    for line in final_log.split('\n')[-15:]:
        if line:
            print(f"  {line}")

    sftp.close()
    c.close()
    print(f"\n✓ 消融实验完成, 所有结果已下载到 {LOCAL_BASE}\\results_stage3\\ablation\\")
    print("\n下一步: 运行 python summarize_stage3_ablation.py 生成图表报告")


if __name__ == "__main__":
    main()
