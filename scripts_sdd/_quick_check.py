"""快速检查 shuffle 当前进度."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('123.127.15.155', port=50472, username='root',
          password='i9D1S9IoRMLR', timeout=30)
_, o, _ = c.exec_command('tail -4 /root/stage2_full.log 2>/dev/null; '
                        'echo ---; '
                        'nvidia-smi --query-gpu=utilization.gpu,memory.used '
                        '--format=csv,noheader', timeout=30)
print(o.read().decode(errors='replace'))
c.close()
