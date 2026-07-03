"""直接启动 tail_rerun.sh 的 nohup, 并验证"""
import paramiko, time

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)

# 先确认脚本文件存在且内容正确
print("[1] 确认 tail_rerun.sh 内容 ...")
out,_ = run(c, "head -5 /root/tail_rerun.sh")
print(out.strip())

# 确认没有 tail_rerun 在跑 (用更精确的匹配, 避免误匹配)
print("\n[2] 检查是否已有 tail_rerun bash 进程 ...")
out,_ = run(c, "pgrep -af 'bash /root/tail_rerun.sh' || echo NONE")
print(out.strip())

if "NONE" in out:
    print("\n[3] 启动 nohup ...")
    # 用 setsid 确保完全脱离 SSH session
    run(c, "setsid bash /root/tail_rerun.sh > /root/tail_rerun_nohup.log 2>&1 < /dev/null &", t=10)
    time.sleep(3)

    print("[4] 验证进程 ...")
    out,_ = run(c, "pgrep -af 'bash /root/tail_rerun.sh' || echo NONE")
    print(out.strip())

    print("\n[5] 检查日志 ...")
    out,_ = run(c, "cat /root/tail_rerun.log 2>/dev/null || echo '(日志还没生成)'")
    print(out.strip())
else:
    print("\n[3] tail_rerun.sh 已在运行, 不重启")
    print(out.strip())

c.close()
print("\n✓ 完成")
