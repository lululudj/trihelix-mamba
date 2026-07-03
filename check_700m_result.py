"""检查 700M eval 结果 + 全面盘点"""
import paramiko, os, json

HOST = "connect.bjb1.seetacloud.com"
PORT = 50472
USER = "root"
PWD = "i9D1S9IoRMLR"
LOCAL_DIR = r"e:\three_chain_v3\results_cloud"

def run(c, cmd, t=30):
    i,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8","replace"), e.read().decode("utf-8","replace")

def parse_ood_local(exp):
    p = os.path.join(LOCAL_DIR, exp, "ood_metrics.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        curve = d.get("changed_acc_curve", {})
        c100 = curve.get("100"); c150 = curve.get("150")
        if c100 and c150:
            return (c100, c150, (c150-c100)/c100*100)
    except:
        pass
    return None

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=30)

print("=== 700M eval 结果 ===")
out,_ = run(c, "cat /root/three_chain_v3/results/run_700m_seed0_2k/ood_metrics.json 2>/dev/null")
if out.strip():
    try:
        d = json.loads(out)
        curve = d.get("changed_acc_curve", {})
        c100 = curve.get("100"); c150 = curve.get("150")
        if c100 and c150:
            decay = (c150-c100)/c100*100
            print(f"  ch@100={c100:.4f}  ch@150={c150:.4f}  decay={decay:+.1f}%")
            print(f"  完整曲线: {curve}")
        else:
            print(f"  JSON 但无曲线: {list(d.keys())}")
    except:
        print(f"  解析失败: {out[:200]}")
else:
    print("  (尚未生成)")

print("\n=== final_rerun.log 关键行 ===")
out,_ = run(c, "cat /root/final_rerun.log 2>/dev/null")
for line in out.strip().split("\n"):
    if any(k in line for k in ["eval", "完成", "失败", "删除", "训练", "DONE", "700M", "300m", "100m"]):
        print(f"  {line.strip()}")

print("\n=== 300m_seed2 训练进度 ===")
out,_ = run(c, "tail -3 /root/three_chain_v3/results/run_300m_seed2_2k/log.jsonl 2>/dev/null")
print(out.strip() if out.strip() else "(无日志)")

print("\n=== 磁盘 ===")
out,_ = run(c, "df -h / | tail -1")
print(out.strip())

c.close()

print("\n\n" + "=" * 70)
print("全面盘点: 所有实验完成情况")
print("=" * 70)

EXPS = {
    "3.26M baseline": [("baseline", "run_three_chain_mamba2_30m_seed0")],  # 之前的实验
    "27M (30m) seed0-4": [(f"seed{s}", f"run_30m_seed{s}" if s>0 else "run_three_chain_mamba2_30m_seed0") for s in range(5)],
    "72M (100m) seed0-4": [(f"seed{s}", f"run_100m_seed{s}{'_500' if s in [0,3,4] else ''}") for s in range(5)],
    "72M HTA seed0,2": [(f"seed{s}", f"run_100m_hta_seed{s}_500") for s in [0,2]],
    "182M (300m) seed0,1": [(f"seed{s}", f"run_300m_seed{s}_2k") for s in [0,1]],
    "728M (700m) seed0": [("seed0", "run_700m_seed0_2k")],
    "1.1B (1000m) seed0": [("seed0", "run_1000m_seed0_2k")],
}

for category, exps in EXPS.items():
    print(f"\n{category}:")
    for label, dirname in exps:
        result = parse_ood_local(dirname)
        if result:
            c100, c150, decay = result
            print(f"  {label}: ch@100={c100:.4f} ch@150={c150:.4f} decay={decay:+.1f}%")
        else:
            print(f"  {label}: 待完成/无结果")

# 正在跑的
print("\n正在云端跑:")
print("  300M seed2: 训练中")
print("  100M HTA seed1: 排队中")
print("\n放弃:")
print("  1000M (1.1B): OOM (d_model=5120 需要 ~25GB 显存 > 24.5GB)")
