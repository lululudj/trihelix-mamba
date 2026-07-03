"""检查云端消融实验进度"""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.bjb1.seetacloud.com", port=50472, username="root",
          password="i9D1S9IoRMLR", timeout=15)


def run(cmd):
    _, o, _ = c.exec_command(cmd, timeout=30)
    return o.read().decode(errors='replace').strip()


print("=== LOG TAIL (30 lines) ===")
print(run("tail -30 /root/stage3_ablation.log 2>/dev/null"))
print()
print("=== PROGRESS.LOG ===")
print(run("cat /root/three_chain_v3/results_stage3/ablation/progress.log 2>/dev/null"))
print()
print("=== ABLATION STATUS ===")
for d in ["no_spatial", "no_temporal", "no_causal", "no_all"]:
    f = f"/root/three_chain_v3/results_stage3/ablation/{d}/ood_metrics.json"
    st = run(f"test -f {f} && echo DONE || echo PENDING")
    if st == "DONE":
        # 读 decay
        decay = run(f"""python -c "import json; m=json.load(open('{f}')); print('{{:.2f}}'.format(m.get('ood_decay_pct',0)*100))" """)
        ch = run(f"""python -c "import json; m=json.load(open('{f}')); print('{{:.4f}}'.format(m.get('changed_acc',0)))" """)
        print(f"  {d}: DONE (decay={decay}%, changed_acc={ch})")
    else:
        print(f"  {d}: PENDING")
print()
print("=== GPU ===")
print(run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"))
print()
print("=== RUNNING PROCESS ===")
print(run("pgrep -af 'python.*run_stage3_ablation' | head -3"))
print()
print("=== MARKER ===")
print(run("ls -la /root/STAGE3_DONE 2>/dev/null || echo NOT_DONE"))
c.close()
