"""补跑 30M 500 步多 seed (task #51)
原因: 30M@100 步 4 seed decay 在 [-1.0%, +6.8%] 间剧烈波动, 不能作为 converged 数据点。
       需补跑 500 步把 30M 也变成 converged, 恢复 "30M→1B 不退化" 主张。

策略:
- 上传 shell 脚本到云端 /root/run_30m_500.sh
- 5 个 seed (0,1,2,3,4) 串行训练 + eval, 共约 55-65 分钟
- 每 seed 完成 touch /root/30M_SEED{N}_DONE marker
- 本地脚本每 60s 轮询, 一完成立即 SFTP 下载
- 全部完成后, 重跑 summarize_all_experiments.py 更新表格

设计参考: control_check_cloud.py (已验证可用的 SSH 编排模板)
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
SEEDS = [0, 1, 2, 3, 4]
STEPS = 500

# 云端 shell: 5 个 seed 串行训练 + eval, 每 seed 末尾 touch marker
RUN_SH = r"""#!/bin/bash
LOG=/root/run_30m_500.log
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/three_chain_v3
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date +%H:%M:%S)] 30M 500 steps multi-seed START ===" >> $LOG

for SEED in 0 1 2 3 4; do
    echo "[$(date +%H:%M:%S)] === 30M seed=${SEED} @500 steps ===" >> $LOG
    python -u train.py --model three_chain_mamba2 \
        --config configs/matched_mamba2_30m.yaml \
        --max_steps 500 --batch_size 2 --seed ${SEED} \
        --data_root ./data_split_m3 \
        --out_dir results/run_30m_seed${SEED}_500 >> $LOG 2>&1
    echo "[$(date +%H:%M:%S)] seed=${SEED} train done" >> $LOG

    python -u eval_ood.py \
        --checkpoint results/run_30m_seed${SEED}_500/final.pt \
        --config configs/matched_mamba2_30m.yaml \
        --data_root ./data/ood_T150 --batch_size 1 --seed ${SEED} \
        --out results/run_30m_seed${SEED}_500/ood_metrics.json >> $LOG 2>&1
    echo "[$(date +%H:%M:%S)] seed=${SEED} eval done" >> $LOG

    rm -f results/run_30m_seed${SEED}_500/final.pt
    rm -f results/run_30m_seed${SEED}_500/best.pt
    touch /root/30M_SEED${SEED}_DONE
    echo "[$(date +%H:%M:%S)] === 30M_SEED${SEED}_DONE ===" >> $LOG
done

touch /root/30M_ALL_DONE
echo "[$(date +%H:%M:%S)] === 30M_ALL_DONE ===" >> $LOG
"""


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
                print(f"    ✓ {entry} ({st.st_size} bytes)", flush=True)
                n += 1
            except Exception as e:
                print(f"    - 跳过 {entry}: {e}", flush=True)
    except FileNotFoundError:
        print(f"    [警告] 远端目录不存在: {remote_dir}", flush=True)
    return n


def tail_log(c, n=3):
    out, _ = run(c, f"tail -n {n} /root/run_30m_500.log 2>/dev/null || echo '(no log yet)'")
    return out.strip()


def gpu_status(c):
    out, _ = run(c, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo '0,0'")
    try:
        parts = out.strip().split(',')
        return f"GPU={parts[0].strip()}%, {parts[1].strip()} MiB"
    except Exception:
        return "GPU?"


def main():
    print("=" * 70, flush=True)
    print(f"补跑 30M 500 步多 seed (seed={SEEDS})", flush=True)
    print(f"预计每 seed ~12 min, 5 seed 共 ~60 min", flush=True)
    print("=" * 70, flush=True)

    c = ssh_connect()
    print("✓ SSH 已连接", flush=True)
    sftp = c.open_sftp()

    # [0] 确认云端空闲
    print("\n[0] 确认云端状态 ...", flush=True)
    print(f"  {gpu_status(c)}", flush=True)
    ps, _ = run(c, "pgrep -f 'train.py' | head -3")
    if ps.strip():
        print(f"  [警告] 有训练进程: {ps.strip()}", flush=True)
    else:
        print("  ✓ 无训练进程, 空闲", flush=True)

    # [1] 上传 shell 脚本
    sh_path = "/root/run_30m_500.sh"
    with sftp.open(sh_path, 'w') as f:
        f.write(RUN_SH)
    run(c, f"chmod +x {sh_path}")
    print(f"\n[1] ✓ 上传 {sh_path}", flush=True)

    # [2] 清理旧 marker
    run(c, "rm -f /root/30M_SEED*_DONE /root/30M_ALL_DONE")
    print("[2] ✓ 清理旧 marker", flush=True)

    # [3] nohup 启动
    run(c, f"nohup bash {sh_path} > /root/30m_nohup.out 2>&1 &", timeout=10)
    time.sleep(3)
    print("\n[3] ✓ 已后台启动 30M 500 步多 seed 实验", flush=True)
    init_log = tail_log(c, 5)
    print("  初始日志:", flush=True)
    for line in init_log.split('\n')[-5:]:
        if line:
            print(f"    {line}", flush=True)

    # [4] 轮询每 seed marker
    for seed in SEEDS:
        marker = f"/root/30M_SEED{seed}_DONE"
        remote_dir = f"{REMOTE_ROOT}/results/run_30m_seed{seed}_500"
        local_dir = os.path.join(LOCAL_BASE, f"run_30m_seed{seed}_500")
        print(f"\n[4.{seed}] 等待 seed={seed} 完成 (marker: {marker}) ...", flush=True)
        waited = 0
        while True:
            time.sleep(60)
            waited += 1
            stat, _ = run(c, f"test -f {marker} && echo DONE || echo RUNNING")
            if stat.strip() == "DONE":
                print(f"\n  ✓✓✓ seed={seed} 完成! (等了 {waited} min)", flush=True)
                break
            # 每 2 min 打印状态
            if waited % 2 == 0 or waited == 1:
                print(f"  [已等 {waited}min] {gpu_status(c)}", flush=True)
                lt = tail_log(c, 2)
                for line in lt.split('\n')[-2:]:
                    if line:
                        print(f"    日志: {line}", flush=True)

        # 立即下载该 seed 结果
        print(f"\n  下载 seed={seed} 结果 ...", flush=True)
        n = download_dir(sftp, remote_dir, local_dir)
        print(f"  共下载 {n} 个文件", flush=True)

    # [5] 最终确认
    print("\n" + "=" * 70, flush=True)
    print("全部 30M 500 步多 seed 完成", flush=True)
    print("=" * 70, flush=True)
    markers, _ = run(c, "ls -la /root/30M_SEED*_DONE /root/30M_ALL_DONE 2>&1")
    print(markers, flush=True)
    final, _ = run(c, "tail -n 10 /root/run_30m_500.log")
    print("最终日志:", flush=True)
    print(final, flush=True)

    sftp.close()
    c.close()
    print("\n✓ 30M 多 seed 补跑完成, 所有结果已下载到 results_cloud/run_30m_seed{N}_500/", flush=True)
    print("下一步: 重跑 python summarize_all_experiments.py 更新表格", flush=True)


if __name__ == "__main__":
    main()
