"""读 log.jsonl 看训练进度"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
LOG_PATH = "/root/three_chain_v3/results/run_three_chain_mamba2_30m_seed0/log.jsonl"


def run_cmd(cli, cmd, timeout=30):
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

    print("\n[2] 训练日志 log.jsonl ...")
    out = run_cmd(cli, f"cat {LOG_PATH} 2>&1; echo '---'; "
                       f"ps aux | grep train.py | grep -v grep | head -1; echo '---'; "
                       f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    print(out)

    cli.close()


if __name__ == "__main__":
    main()
