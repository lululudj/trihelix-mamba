"""列出云端 SDD annotations 可用场景 + 各场景视频数 (不影响训练)."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('123.127.15.155', port=50472, username='root',
          password='i9D1S9IoRMLR', timeout=30)
_, o, _ = c.exec_command(
    'ls /root/three_chain_v3/SDD/annotations/ 2>/dev/null; '
    'echo ===视频数===; '
    'for d in /root/three_chain_v3/SDD/annotations/*/; do '
    'echo "$(basename $d): $(ls -d $d*/ 2>/dev/null | wc -l) videos"; '
    'done', timeout=30)
print(o.read().decode(errors='replace'))
c.close()
