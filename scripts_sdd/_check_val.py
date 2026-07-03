"""等待并检查 val 趋势."""
import time
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.bjb1.seetacloud.com', port=50472, username='root', password='i9D1S9IoRMLR', timeout=30)

time.sleep(540)

# grep val 行
_, o, _ = c.exec_command('grep -E "val" /root/stage2_full.log 2>/dev/null')
print("=== Val 评估点 ===")
print(o.read().decode(errors='replace'))

# 最新进度
_, o, _ = c.exec_command('tail -5 /root/stage2_full.log 2>/dev/null')
print("=== 最新 ===")
print(o.read().decode(errors='replace'))

c.close()
