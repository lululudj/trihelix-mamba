"""安装 Python 依赖到云服务器 (用 conda 绝对路径)。"""
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 16141
USER = "root"
PASS = "BaJ+0sL75oMj"

# 用绝对路径 + source conda
INIT = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && export HF_ENDPOINT=https://hf-mirror.com && "

COMMANDS = [
    # 1. 装 transformers + accelerate + requests
    INIT + 'pip install -q transformers accelerate requests -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -15',
    # 2. 装 mamba-ssm + causal-conv1d (可能要编译, 给长超时)
    INIT + 'pip install mamba-ssm causal-conv1d 2>&1 | tail -40',
    # 3. 验证 transformers
    INIT + 'python -c "import transformers, accelerate, requests; print(\'transformers=\', transformers.__version__)"',
    # 4. 验证 mamba_ssm
    INIT + 'python -c "from mamba_ssm import Mamba2; import torch; m=Mamba2(d_model=256).cuda(); x=torch.randn(1,100,256).cuda(); print(\'mamba ok:\', m(x).shape)"',
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
print(f"[SSH] 连接成功\n{'='*60}")

for i, cmd in enumerate(COMMANDS, 1):
    print(f"\n--- Step {i}/{len(COMMANDS)} ---")
    # 不用 get_pty，避免 conda 初始化问题
    stdin, stdout, stderr = client.exec_command(cmd, timeout=900)
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    exit_code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out)
    if err.strip():
        print(f"[stderr]\n{err[-2000:]}")
    print(f"[exit={exit_code}]")

client.close()
print(f"\n{'='*60}\n[SSH] 完成")
