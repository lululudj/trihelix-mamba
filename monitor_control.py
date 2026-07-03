"""纯监控下载: 云端 nohup 已在跑, 本脚本只轮询 marker + 下载结果"""
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

PHASES = [
    {
        'name': '实验1 B@1000步对照',
        'marker': '/root/EXP1_DONE',
        'dirs': [('run_30m_deep_n4_seed0_1000', 'run_30m_deep_n4_seed0_1000')],
        'eta_min': 12,
    },
    {
        'name': '实验3 随机标签sanity (一票否决)',
        'marker': '/root/EXP3_DONE',
        'dirs': [('run_30m_shufflelabel_seed0_500', 'run_30m_shufflelabel_seed0_500')],
        'eta_min': 8,
    },
    {
        'name': '实验2 1B多seed (seed1,2)',
        'marker': '/root/EXP2_DONE',
        'dirs': [
            ('run_1000m_ckpt_long_seed1_2000', 'run_1000m_ckpt_long_seed1_2000'),
            ('run_1000m_ckpt_long_seed2_2000', 'run_1000m_ckpt_long_seed2_2000'),
        ],
        'eta_min': 95,
    },
]


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
    out, _ = run(c, f"tail -n {n} /root/control_experiments.log 2>/dev/null || echo '(no log yet)'")
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
    print("监控模式: 云端 nohup 已在跑, 本脚本只轮询 + 下载", flush=True)
    print("=" * 70, flush=True)

    c = ssh_connect()
    print("✓ SSH 已连接", flush=True)
    sftp = c.open_sftp()

    # 确认 run_control.sh 父进程在
    ps, _ = run(c, "pgrep -af 'run_control.sh' | head -3")
    if ps.strip():
        print(f"✓ run_control.sh 在运行: {ps.strip()}", flush=True)
    else:
        print("⚠ 未发现 run_control.sh, 检查是否已完成或崩溃 ...", flush=True)
        # 可能已经全部完成, 或某阶段崩了
        for ph in PHASES:
            st, _ = run(c, f"test -f {ph['marker']} && echo DONE || echo PENDING")
            print(f"  {ph['name']}: {st.strip()}", flush=True)

    # 轮询每个 phase
    for i, ph in enumerate(PHASES, 1):
        # 先检查是否已完成
        st, _ = run(c, f"test -f {ph['marker']} && echo DONE || echo PENDING")
        if st.strip() == "DONE":
            print(f"\n[{i}] {ph['name']} 已完成, 直接下载 ...", flush=True)
        else:
            print(f"\n[{i}] 等待 {ph['name']} (marker: {ph['marker']}, 预计 {ph['eta_min']} min) ...", flush=True)
            waited = 0
            while True:
                time.sleep(60)
                waited += 1
                st, _ = run(c, f"test -f {ph['marker']} && echo DONE || echo RUNNING")
                if st.strip() == "DONE":
                    print(f"\n  ✓✓✓ {ph['name']} 完成! (等了 {waited} min)", flush=True)
                    break
                if waited % 2 == 0 or waited == 1:
                    print(f"  [已等 {waited}min] {gpu_status(c)}", flush=True)
                    lt = tail_log(c, 2)
                    for line in lt.split('\n')[-2:]:
                        if line:
                            print(f"    日志: {line}", flush=True)

        # 下载
        print(f"  下载 {ph['name']} 结果 ...", flush=True)
        total = 0
        for rsub, lsub in ph['dirs']:
            remote = f"{REMOTE_ROOT}/results/{rsub}"
            local = os.path.join(LOCAL_BASE, lsub)
            print(f"    {rsub}:", flush=True)
            total += download_dir(sftp, remote, local)
        print(f"  共下载 {total} 个文件", flush=True)

        lt = tail_log(c, 3)
        print("  日志末尾:", flush=True)
        for line in lt.split('\n')[-3:]:
            if line:
                print(f"    {line}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("全部控制实验完成!", flush=True)
    print("=" * 70, flush=True)
    markers, _ = run(c, "ls -la /root/EXP1_DONE /root/EXP3_DONE /root/EXP2_DONE 2>&1")
    print(markers, flush=True)
    final, _ = run(c, "tail -n 15 /root/control_experiments.log")
    print("最终日志:", flush=True)
    print(final, flush=True)

    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
