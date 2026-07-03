"""查 GPU 使用率 + 训练进程状态"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"


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

    print("\n[2] GPU 使用率 ...")
    out = run_cmd(cli, "nvidia-smi")
    print(out)

    print("\n[3] 训练进程 ...")
    out = run_cmd(cli, "ps aux | grep -E 'train.py|python' | grep -v grep")
    print(out)

    print("\n[4] 训练日志末尾 ...")
    out = run_cmd(cli, "ls -la /root/three_chain_v3/results/run_three_chain_mamba2_30m_seed0/ 2>&1; echo '---'; cat /root/three_chain_v3/results/run_three_chain_mamba2_30m_seed0/log.jsonl 2>&1 | tail -10")
    print(out)

    cli.close()


if __name__ == "__main__":
    main()
