"""找出云端 python 路径."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('123.127.15.155', port=50472, username='root',
          password='i9D1S9IoRMLR', timeout=30)
_, o, _ = c.exec_command(
    'source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null; '
    'which python python3 2>&1; '
    'echo ---; '
    'ls /root/miniconda3/bin/python* 2>&1; '
    'echo ---; '
    'cat /root/stage2_full.sh 2>/dev/null | head -5', timeout=30)
print(o.read().decode(errors='replace'))
c.close()
