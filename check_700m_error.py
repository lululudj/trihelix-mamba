"""查看 700M eval 的完整错误信息"""
import paramiko

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

print("=== final_rerun.log 完整内容 ===")
out,_ = run(c, "cat /root/final_rerun.log")
print(out)

print("\n=== final_rerun_nohup.log (可能有更多 stderr) ===")
out,_ = run(c, "cat /root/final_rerun_nohup.log 2>/dev/null | head -80")
print(out)

c.close()
