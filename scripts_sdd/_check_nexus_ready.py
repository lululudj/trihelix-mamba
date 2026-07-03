"""检查云端实验 2 (nexus) 就绪状况: SDD annotations / gen 脚本 / ood 数据格式."""
import paramiko

HOST = "123.127.15.155"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)

def run(cmd):
    _, stdout, stderr = c.exec_command(cmd, timeout=30)
    return stdout.read().decode('utf-8', errors='replace'), stderr.read().decode('utf-8', errors='replace')

print("=== 1. SDD annotations 根目录 ===")
o, e = run("ls -d /root/three_chain_v3/data/sdd* /root/three_chain_v3/sdd* /root/sdd* 2>/dev/null")
print(o.strip() or "(未找到 sdd 目录)")

print("\n=== 2. 各场景子目录 ===")
o, e = run("ls /root/three_chain_v3/data/sdd_annotations/annotations/ 2>/dev/null")
print("annotations 目录:", o.strip() or "(无)")
o, e = run("find /root/three_chain_v3 -maxdepth 4 -type d -name 'nexus' 2>/dev/null")
print("nexus 目录:", o.strip() or "(无)")
o, e = run("find /root/three_chain_v3 -maxdepth 4 -type d -name 'bookstore' 2>/dev/null")
print("bookstore 目录:", o.strip() or "(无)")

print("\n=== 3. nexus 视频数 ===")
o, e = run("find /root/three_chain_v3 -type d -name 'nexus' -exec ls -d {}/*/ \\; 2>/dev/null | wc -l")
print("nexus 子目录数:", o.strip())

print("\n=== 4. gen_sdd_grid.py 是否在云端 ===")
o, e = run("ls -la /root/three_chain_v3/data/gen_sdd_grid.py 2>/dev/null")
print(o.strip() or "(无)")

print("\n=== 5. bookstore 数据格式参考 ===")
o, e = run("ls /root/three_chain_v3/data_sdd_split/train/ 2>/dev/null | head -3")
print("train split:", o.strip() or "(无)")
o, e = run("ls /root/three_chain_v3/data/ood_T150_sdd/ 2>/dev/null | head -3")
print("ood T150:", o.strip() or "(无)")

print("\n=== 6. split_dataset 脚本 ===")
o, e = run("ls /root/three_chain_v3/data/split_dataset.py 2>/dev/null")
print(o.strip() or "(无)")
o, e = run("grep -rn 'def split_dataset\\|def main' /root/three_chain_v3/data/dataset.py 2>/dev/null | head -5")
print("dataset.py:", o.strip())

c.close()
