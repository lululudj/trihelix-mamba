"""关闭 AutoDL 云端机器 (省成本)

阶段 3 全部完成, 结果已下载本地, 云端空跑浪费钱, 执行关机。
关机前先检查: (1) GPU 占用 (2) 是否有 python 训练进程在跑
"""
import paramiko
import sys

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"


def main():
    print(f"=== 连接 AutoDL {HOST}:{PORT} ===")
    try:
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15)
    except Exception as e:
        print(f"[ERROR] SSH 连接失败: {e}")
        print("可能机器已经关机, 或网络问题。如果是已关机则无需操作。")
        sys.exit(0)

    # 1. 检查 GPU 占用
    print("\n--- 1. GPU 占用检查 ---")
    stdin, stdout, stderr = cli.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader")
    gpu_info = stdout.read().decode().strip()
    print(f"GPU: {gpu_info}")

    # 2. 检查 python 训练进程
    print("\n--- 2. Python 进程检查 ---")
    stdin, stdout, stderr = cli.exec_command("pgrep -af 'python.*train' || echo 'NO_TRAIN_PROCESS'")
    proc_info = stdout.read().decode().strip()
    print(f"训练进程: {proc_info}")

    # 3. 检查 marker (3.3 是否完成)
    print("\n--- 3. Stage 3.3 完成标记检查 ---")
    stdin, stdout, stderr = cli.exec_command("ls -la /root/three_chain_v3/results_stage3/*.npz 2>/dev/null; grep STAGE3_3_DONE /root/stage3_3.log 2>/dev/null || echo 'MARKER_NOT_FOUND'")
    marker_info = stdout.read().decode().strip()
    print(f"标记: {marker_info}")

    # 4. 决策: 是否安全关机 (无训练进程 → 自动关机; 有训练 → 警告并退出)
    has_train = "NO_TRAIN_PROCESS" not in proc_info
    if has_train:
        print("\n[警告] 检测到训练进程仍在运行, 不关机。请先确认实验状态。")
        cli.close()
        return

    # 5. 执行关机
    print("\n--- 4. 执行关机 ---")
    print("AutoDL 实例将通过 shutdown 进入'已关机'状态, 停止 GPU 计费。")
    print("(系统盘存储费可能继续按少量计费, 这是 AutoDL 规则)")
    try:
        # 用 nohup + sleep 确保 SSH 断开后命令仍执行
        cli.exec_command("nohup bash -c 'sleep 2 && shutdown -h now' >/dev/null 2>&1 &")
        print("[OK] 关机命令已发送, 2 秒后实例关闭。")
    except Exception as e:
        print(f"[ERROR] 关机命令发送失败: {e}")
    finally:
        cli.close()


if __name__ == "__main__":
    main()
