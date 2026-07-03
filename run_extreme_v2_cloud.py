"""V2 极限测试编排器: C2 长训 + B2 超深 + D2 retry
每阶段完成立即下载结果, 避免连接断导致数据丢失。
"""
import os
import time
import posixpath
import paramiko

HOST = 'connect.bjb1.seetacloud.com'
PORT = 50472
USER = 'root'
PWD = 'i9D1S9IoRMLR'

LOCAL_BASE = r'e:\three_chain_v3\results_cloud'
LOCAL_FILES_TO_UPLOAD = [
    (r'e:\three_chain_v3\configs\matched_mamba2_30m_deep8.yaml',
     '/root/three_chain_v3/configs/matched_mamba2_30m_deep8.yaml'),
    (r'e:\three_chain_v3\configs\matched_mamba2_1000m_ckpt_long.yaml',
     '/root/three_chain_v3/configs/matched_mamba2_1000m_ckpt_long.yaml'),
    (r'e:\three_chain_v3\run_extreme_v2.sh',
     '/root/run_extreme_v2.sh'),
]

# 每个 phase 的云端结果目录 + 本地目录 + marker 文件
PHASES = [
    {
        'name': 'C2 1B 长训 2000步',
        'marker': '/root/C2_ALL_DONE',
        'remote_dir': '/root/three_chain_v3/results/run_1000m_ckpt_long_seed0_2000',
        'local_dir': os.path.join(LOCAL_BASE, 'run_1000m_ckpt_long_seed0_2000'),
    },
    {
        'name': 'B2 n_layers=8 超深',
        'marker': '/root/B2_ALL_DONE',
        'remote_dir': '/root/three_chain_v3/results/run_30m_deep_n8_seed0_1000',
        'local_dir': os.path.join(LOCAL_BASE, 'run_30m_deep_n8_seed0_1000'),
    },
    {
        'name': 'D2 T250mask retry',
        'marker': '/root/D2_ALL_DONE',
        'remote_dir': '/root/three_chain_v3/results/run_100m_maxT1024_v2_seed0_500',
        'local_dir': os.path.join(LOCAL_BASE, 'run_100m_maxT1024_v2_seed0_500'),
    },
]


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15)
    return c


def run(c, cmd, timeout=30):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip(), \
        e.read().decode(errors='replace').strip()


def download_dir(c, remote_dir, local_dir):
    """下载云端目录所有 <500MB 文件到本地 (跳过 .pt)。"""
    os.makedirs(local_dir, exist_ok=True)
    sftp = c.open_sftp()
    downloaded = []
    try:
        files = sftp.listdir(remote_dir)
    except Exception as e:
        sftp.close()
        return [], f"列目录失败: {e}"

    for fname in files:
        remote_path = posixpath.join(remote_dir, fname)
        local_path = os.path.join(local_dir, fname)
        try:
            st = sftp.stat(remote_path)
            if st.st_size > 500_000_000:
                print(f"    ⏭ {fname} ({st.st_size/1e6:.0f}MB) 太大跳过")
                continue
            sftp.get(remote_path, local_path)
            sz = os.path.getsize(local_path)
            downloaded.append((fname, sz))
            print(f"    ✓ {fname} ({sz} bytes)")
        except Exception as e:
            print(f"    ✗ {fname}: {e}")
    sftp.close()
    return downloaded, None


def main():
    print("=" * 70)
    print("V2 极限测试编排器: C2 + B2 + D2")
    print("=" * 70)

    c = connect()
    print("✓ SSH 已连接")

    # 1. 上传配置和脚本
    print("\n[1] 上传文件 ...")
    sftp = c.open_sftp()
    for local, remote in LOCAL_FILES_TO_UPLOAD:
        sftp.put(local, remote)
        print(f"  ✓ {os.path.basename(local)} → {remote}")
    sftp.close()

    # 给脚本可执行权限
    run(c, 'chmod +x /root/run_extreme_v2.sh')
    print("  ✓ chmod +x run_extreme_v2.sh")

    # 清理旧 marker (避免误判)
    run(c, 'rm -f /root/C2_ALL_DONE /root/B2_ALL_DONE /root/D2_ALL_DONE /root/V2_ALL_DONE')
    print("  ✓ 清理旧 marker")

    # 2. 启动 nohup 后台运行
    print("\n[2] 启动 V2 测试 (nohup 后台) ...")
    out, _ = run(c, 'nohup bash /root/run_extreme_v2.sh > /root/extreme_v2_nohup.log 2>&1 & echo "PID=$!"')
    print(f"  {out}")

    # 等待 5 秒确认启动
    time.sleep(5)
    out, _ = run(c, 'tail -3 /root/extreme_v2.log 2>&1')
    print(f"  初始日志:\n{out}")

    # 3. 顺序监控每个 phase
    for idx, phase in enumerate(PHASES, 1):
        print(f"\n[{idx+1}] 等待 {phase['name']} 完成 (marker: {phase['marker']}) ...")
        wait_min = 0
        while True:
            out, _ = run(c, f'test -f {phase["marker"]} && echo DONE || echo PENDING')
            if out == 'DONE':
                print(f"\n  ✓✓✓ {phase['name']} 完成! (等了 {wait_min} min)")
                break
            time.sleep(60)
            wait_min += 1
            # 每 2 min 打印一次进度
            if wait_min % 2 == 0:
                log_tail, _ = run(c, f'tail -2 /root/extreme_v2.log 2>&1')
                gpu, _ = run(c, 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1')
                print(f"  [已等 {wait_min}min] GPU={gpu}")
                print(f"    日志: {log_tail[:200]}")

        # 立即下载该 phase 的结果
        print(f"\n  下载 {phase['name']} 结果 ...")
        print(f"    云端: {phase['remote_dir']}")
        print(f"    本地: {phase['local_dir']}")
        downloaded, err = download_dir(c, phase['remote_dir'], phase['local_dir'])
        if err:
            print(f"  ⚠ {err}")
        else:
            print(f"  共下载 {len(downloaded)} 个文件")

        # 打印该 phase 的日志末尾
        log_tail, _ = run(c, f'tail -15 /root/extreme_v2.log 2>&1')
        print(f"\n  日志末尾:")
        for line in log_tail.split('\n'):
            print(f"    {line}")

    # 4. 最终确认
    print("\n" + "=" * 70)
    print("全部 phase 完成, 最终确认")
    print("=" * 70)
    out, _ = run(c, 'ls -la /root/C2_ALL_DONE /root/B2_ALL_DONE /root/D2_ALL_DONE /root/V2_ALL_DONE 2>&1')
    print(out)
    print()
    out, _ = run(c, 'tail -20 /root/extreme_v2.log 2>&1')
    print("V2 日志末尾:")
    print(out)

    c.close()
    print("\n✓ V2 编排完成, 所有结果已下载")


if __name__ == '__main__':
    main()
