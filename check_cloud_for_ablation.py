"""快速检查 AutoDL 状态 + 云端数据路径，为 3.2 消融做准备"""
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
REMOTE_ROOT = "/root/three_chain_v3"

try:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15)
    print("✓ SSH 连接成功 - AutoDL 实例仍在运行")

    def run(cmd):
        _, out, err = c.exec_command(cmd, timeout=30)
        return out.read().decode(errors='replace').strip(), err.read().decode(errors='replace').strip()

    # GPU 状态
    gpu, _ = run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1")
    print(f"GPU 状态: {gpu}")

    # 训练进程
    procs, _ = run("pgrep -af 'python.*train.py' 2>&1 || echo '无训练进程'")
    print(f"训练进程: {procs}")

    # 云端代码版本检查
    print("\n--- 云端代码状态 ---")
    # 检查 three_chain_mamba2.py 是否已有 ablate (本地刚改的还没上传)
    abl, _ = run(f"grep -c 'ablate_s' {REMOTE_ROOT}/models/three_chain_mamba2.py 2>/dev/null || echo 0")
    print(f"云端 three_chain_mamba2.py ablate_s 出现次数: {abl} (0=未更新)")

    # 数据路径检查
    print("\n--- 数据路径检查 ---")
    for d in ["data_split", "data_split_m3", "data/ood_T150"]:
        st, _ = run(f"ls -d {REMOTE_ROOT}/{d} 2>/dev/null && echo EXISTS || echo MISSING")
        print(f"  {d}: {st}")

    # ood_T150 样本数
    n_ood, _ = run(f"ls {REMOTE_ROOT}/data/ood_T150/*.npz 2>/dev/null | wc -l")
    print(f"  ood_T150 样本数: {n_ood}")

    # 磁盘空间
    disk, _ = run("df -h /root | tail -1")
    print(f"\n磁盘: {disk}")

    # conda 环境
    py, _ = run("source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && which python && python -c 'import mamba_ssm; print(\"mamba_ssm OK\")' 2>&1")
    print(f"\nPython 环境:\n{py}")

    c.close()
    print("\n✓ 检查完成")
except Exception as e:
    print(f"✗ SSH 连接失败: {e}")
    print("→ AutoDL 实例可能已关机")
