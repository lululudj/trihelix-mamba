"""下载云端消融实验结果到本地"""
import os
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
REMOTE_BASE = "/root/three_chain_v3/results_stage3/ablation"
LOCAL_BASE = r"e:\three_chain_v3\results_stage3\ablation"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=15)
sftp = c.open_sftp()
print("✓ SSH 已连接")


def download(remote, local):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    try:
        sftp.get(remote, local)
        st = sftp.stat(remote)
        print(f"  ✓ {os.path.basename(remote)} ({st.st_size} bytes)")
        return True
    except FileNotFoundError:
        print(f"  - [跳过] 远端不存在: {remote}")
        return False
    except Exception as e:
        print(f"  - [错误] {remote}: {e}")
        return False


# 下载 4 个 ablation 目录
for name in ["no_spatial", "no_temporal", "no_causal", "no_all"]:
    print(f"\n--- {name} ---")
    remote_dir = f"{REMOTE_BASE}/{name}"
    local_dir = os.path.join(LOCAL_BASE, name)
    os.makedirs(local_dir, exist_ok=True)
    try:
        entries = sftp.listdir(remote_dir)
    except FileNotFoundError:
        print(f"  [警告] 远端目录不存在: {remote_dir}")
        continue
    for entry in entries:
        # 跳过大的 .pt 文件 (best.pt/final.pt 各 ~100MB, 不需要本地)
        if entry.endswith(".pt"):
            print(f"  - [跳过 .pt] {entry}")
            continue
        download(f"{remote_dir}/{entry}", os.path.join(local_dir, entry))

# 下载日志文件
print("\n--- 日志文件 ---")
for log in ["/root/stage3_ablation.log", "/root/stage3_nohup.out"]:
    name = os.path.basename(log)
    download(log, os.path.join(LOCAL_BASE, name))

# 下载 progress.log
download(f"{REMOTE_BASE}/progress.log", os.path.join(LOCAL_BASE, "progress.log"))

sftp.close()
c.close()
print("\n✓ 下载完成")
