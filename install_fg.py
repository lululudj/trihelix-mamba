"""调试 - 直接前台跑 install_mamba.sh (不后台)
让 paramiko 阻塞等命令完成, mamba-ssm 编译大约 2-5 分钟
"""
import sys
import time
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PASSWORD = "i9D1S9IoRMLR"
INSTALL_SH = "/root/install_mamba.sh"


def main():
    print("[1] 连接 ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("    ✓ 连上了")

    # 不用 get_pty, 不用 timeout, 直接前台跑
    print("\n[2] 直接前台跑 install_mamba.sh (paramiko 通道 timeout=None 永久等待) ...")
    print("    mamba-ssm 2.2.4 编译约 2-5 分钟, 请耐心等 ...")

    # 关键: timeout=None 让 channel 不超时
    transport = cli.get_transport()
    channel = transport.open_session()
    # 不设 timeout (默认 None = 永久等待)
    channel.exec_command(f"bash {INSTALL_SH} 2>&1")

    out_buf = []
    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode("utf-8", errors="replace")
            out_buf.append(data)
            # 实时打印
            for line in data.splitlines():
                print("    " + line.rstrip())
        elif channel.exit_status_ready():
            # 读完剩余
            while channel.recv_ready():
                data = channel.recv(4096).decode("utf-8", errors="replace")
                out_buf.append(data)
                for line in data.splitlines():
                    print("    " + line.rstrip())
            break
        else:
            time.sleep(0.5)

    rc = channel.recv_exit_status()
    print(f"\n[3] 完成, rc={rc}")

    out = "".join(out_buf)
    if "INSTALL_DONE" in out:
        print("\n✓✓✓ mamba_ssm 装好了!")
    elif "OK" in out and "mamba_ssm" in out:
        print("\n✓✓✓ mamba_ssm 验证 OK!")
    else:
        print("\n✗ 安装可能失败, 查看上面的输出")

    cli.close()


if __name__ == "__main__":
    main()
