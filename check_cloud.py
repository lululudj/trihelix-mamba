"""检查 run_cloud_30m.sh + 启动训练"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
WORKDIR = "/root/three_chain_v3"


def run_cmd(cli, cmd, timeout=60):
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.exec_command(cmd + " 2>&1")
    out = b""
    start = time.time()
    while True:
        if chan.recv_ready():
            out += chan.recv(4096)
        elif chan.exit_status_ready():
            while chan.recv_ready():
                out += chan.recv(4096)
            break
        else:
            time.sleep(0.2)
            if time.time() - start > timeout:
                break
    return out.decode("utf-8", errors="replace")


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    # [2] 检查 run_cloud_30m.sh 内容 (是否需要 conda activate)
    print("\n[2] 检查 run_cloud_30m.sh ...")
    out = run_cmd(cli, f"cat {WORKDIR}/run_cloud_30m.sh")
    print(out)

    # [3] 检查数据/代码目录
    print("\n[3] 检查关键文件 ...")
    out = run_cmd(cli, f"ls {WORKDIR}/train.py {WORKDIR}/eval_ood.py {WORKDIR}/utils.py 2>&1; "
                       f"echo '---'; ls {WORKDIR}/configs/matched_mamba2_30m.yaml 2>&1; "
                       f"echo '---'; ls {WORKDIR}/data_split_m3/ 2>&1 | head -5; "
                       f"echo '---'; ls {WORKDIR}/data/ood_T150/ 2>&1 | head -5")
    print(out)

    cli.close()


if __name__ == "__main__":
    main()
