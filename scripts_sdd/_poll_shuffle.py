"""轮询 shuffle_labels sanity 完成, 完成后下载结果并打印对比.
带 flush 防止 stdout 缓冲. 用 IP 直连.
"""
import sys
import time
import json
import paramiko

IP = '123.127.15.155'
PORT = 50472
USER = 'root'
PWD = 'i9D1S9IoRMLR'

DEADLINE = time.time() + 45 * 60  # 最多等 45 分钟

print(f"[{time.strftime('%H:%M:%S')}] 开始轮询 shuffle_labels sanity 完成...",
      flush=True)

last_step = -1
while time.time() < DEADLINE:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(IP, port=PORT, username=USER, password=PWD, timeout=30)

        # 1. 完成 marker
        _, o, _ = c.exec_command('cat /root/STAGE2_FULL_DONE 2>/dev/null',
                                 timeout=15)
        done = o.read().decode(errors='replace').strip()

        # 2. shuffle 进度 (log 尾部)
        _, o, _ = c.exec_command(
            'tail -6 /root/stage2_full.log 2>/dev/null', timeout=15)
        tail = o.read().decode(errors='replace')

        # 提取当前 step
        cur_step = last_step
        for line in tail.splitlines()[::-1]:
            if 'step' in line and '/3000' in line:
                try:
                    cur_step = int(line.split('step')[1].split('/')[0]
                                    .replace('[', '').strip())
                except Exception:
                    pass
                break

        # 3. shuffle metrics 是否已生成
        _, o, _ = c.exec_command(
            'ls -la /root/three_chain_v3/results_stage2/sdd_shuffle_3k/'
            'ood_metrics.json 2>&1', timeout=15)
        metrics_exists = 'No such file' not in o.read().decode(errors='replace')

        c.close()

        if cur_step != last_step:
            print(f"[{time.strftime('%H:%M:%S')}] shuffle step={cur_step}/3000 "
                  f"| metrics_done={metrics_exists} | DONE={bool(done)}",
                  flush=True)
            last_step = cur_step
        else:
            print(f"[{time.strftime('%H:%M:%S')}] (no step change) "
                  f"metrics_done={metrics_exists} DONE={bool(done)}",
                  flush=True)

        if done or metrics_exists:
            print(f"\n[{time.strftime('%H:%M:%S')}] "
                  f"!!! shuffle 完成 (DONE={bool(done)}, "
                  f"metrics_exists={metrics_exists})", flush=True)

            # 下载 metrics 对比
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(IP, port=PORT, username=USER, password=PWD, timeout=30)
            _, o, _ = c.exec_command(
                'cat /root/three_chain_v3/results_stage2/sdd_shuffle_3k/'
                'ood_metrics.json 2>/dev/null', timeout=30)
            content = o.read().decode(errors='replace')
            c.close()

            if content.strip():
                m = json.loads(content)
                ch100 = m.get('changed_acc_curve', {}).get('100')
                ch150 = m.get('changed_acc_curve', {}).get('150')
                print(f"\n=== shuffle_labels sanity 结果 ===", flush=True)
                print(f"  ch_acc@100 = {ch100}", flush=True)
                print(f"  ch_acc@150 = {ch150}", flush=True)
                print(f"  decay_pct  = {m.get('ood_decay_pct')}", flush=True)
                print(f"  decay_abs  = {m.get('ood_decay_100_to_150')}",
                      flush=True)
                print(f"\n=== 对比 ===", flush=True)
                print(f"  SSM 10k:  ch@100=0.1777 ch@150=0.2723 "
                      f"decay=+53.2%", flush=True)
                print(f"  shuffle:  ch@100={ch100} ch@150={ch150} "
                      f"decay={m.get('ood_decay_pct')}", flush=True)
                print(f"\n  预期: shuffle 应 ch_acc≈随机(接近 1/16=0.0625), "
                      f"证明指标有效", flush=True)
            else:
                print("  [警告] shuffle metrics 文件不存在或为空", flush=True)

            print(f"\n=== 全部完成 ===", flush=True)
            sys.exit(0)

    except Exception as ex:
        print(f"[{time.strftime('%H:%M:%S')}] [error] {ex}", flush=True)

    time.sleep(120)  # 每 2 分钟检查一次

print(f"\n[{time.strftime('%H:%M:%S')}] 超时 45 分钟,仍未完成", flush=True)
