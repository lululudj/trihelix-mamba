"""从云服务器 (RTX 5090D) 下载 CodeBERT 训练 + 度量 + 自进化结果到本地.

下载关键文件 (用 _cloud 后缀避免覆盖本地 RTX 4060 训练结果):
  - transcripts/text_b_codebert_measure_report.json  (度量报告)
  - checkpoints/text_encoder_b_codebert_loss.json     (训练曲线)
  - transcripts/real_evolve_report_text_hf_bert_small.md (自进化报告)
  - checkpoints/text_encoder_b_codebert.pt            (499MB checkpoint, 可选)
  - transcripts/real_evolve_text_hf_bert_small.log    (自进化日志)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 16141
USER = "root"
PASS = "BaJ+0sL75oMj"

REMOTE_BASE = "/root/autodl-tmp/mambacoding"
LOCAL_BASE = Path("e:/mambacoding")

# (remote_path, local_path, 是否必须)
FILES = [
    ("transcripts/text_b_codebert_measure_report.json",
     "transcripts/text_b_codebert_measure_report_cloud.json", True),
    ("checkpoints/text_encoder_b_codebert_loss.json",
     "checkpoints/text_encoder_b_codebert_loss_cloud.json", True),
    ("transcripts/real_evolve_report_text_hf_bert_small.md",
     "transcripts/real_evolve_report_text_hf_bert_small_cloud.md", True),
    ("transcripts/real_evolve_text_hf_bert_small.log",
     "transcripts/real_evolve_text_hf_bert_small_cloud.log", False),
    ("transcripts/real_evolve_report_text_hf_codebert.md",
     "transcripts/real_evolve_report_text_hf_codebert_cloud.md", False),
    ("transcripts/train_b_codebert_log.txt",
     "transcripts/train_b_codebert_log_cloud.txt", False),
    # 大文件 checkpoint (499MB, 最后下载)
    ("checkpoints/text_encoder_b_codebert.pt",
     "checkpoints/text_encoder_b_codebert_cloud.pt", False),
]


def download(sftp, remote: str, local: Path, required: bool) -> bool:
    """下载单个文件, 返回是否成功."""
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 检查远程文件是否存在 + 大小
        try:
            st = sftp.stat(remote)
            size_mb = st.st_size / 1024 / 1024
            print(f"[下载] {remote} ({size_mb:.1f} MB) → {local}")
        except FileNotFoundError:
            if required:
                print(f"[缺失] 必须文件不存在: {remote}")
                return False
            else:
                print(f"[跳过] 远程不存在: {remote}")
                return False

        t0 = time.time()
        # 进度回调
        last_print = [0]

        def cb(transferred, total):
            now = time.time()
            if now - last_print[0] > 2.0 or transferred == total:
                pct = transferred / total * 100 if total else 0
                speed = transferred / 1024 / 1024 / max(now - t0, 0.1)
                print(f"  {pct:5.1f}%  {transferred/1024/1024:6.1f}/{total/1024/1024:.1f} MB  {speed:.1f} MB/s",
                      end="\r", flush=True)
                last_print[0] = now

        sftp.get(remote, str(local), callback=cb)
        elapsed = time.time() - t0
        print(f"  ✅ 完成 ({elapsed:.1f}s)                    ")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        if required:
            return False
        return False


def main():
    print("=" * 60)
    print("下载云服务器 (RTX 5090D) 训练结果到本地")
    print("=" * 60)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[SSH] 连接 {USER}@{HOST}:{PORT} ...")
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    except Exception as e:
        print(f"[SSH] 连接失败: {e}")
        print("[提示] 云实例可能已关机或网络不通, 请检查 AutoDL 控制台实例状态")
        sys.exit(1)
    print("[SSH] 连接成功\n")

    sftp = client.open_sftp()

    # 先列出远程 transcripts 和 checkpoints 目录, 确认有哪些文件
    print("[远程目录] transcripts/")
    try:
        for f in sorted(sftp.listdir(f"{REMOTE_BASE}/transcripts")):
            st = sftp.stat(f"{REMOTE_BASE}/transcripts/{f}")
            print(f"  {f}  ({st.st_size/1024:.1f} KB)")
    except Exception as e:
        print(f"  列目录失败: {e}")

    print("\n[远程目录] checkpoints/ (只列 .pt 和 .json)")
    try:
        for f in sorted(sftp.listdir(f"{REMOTE_BASE}/checkpoints")):
            if f.endswith(".pt") or f.endswith(".json"):
                st = sftp.stat(f"{REMOTE_BASE}/checkpoints/{f}")
                print(f"  {f}  ({st.st_size/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"  列目录失败: {e}")

    # 下载文件
    print("\n" + "=" * 60)
    print("开始下载")
    print("=" * 60)
    results = []
    for rel_remote, rel_local, required in FILES:
        remote = f"{REMOTE_BASE}/{rel_remote}"
        local = LOCAL_BASE / rel_local
        ok = download(sftp, remote, local, required)
        results.append((rel_remote, ok, required))

    sftp.close()
    client.close()
    print(f"\n{'='*60}\n[SSH] 连接关闭")

    # 汇总
    print("\n" + "=" * 60)
    print("下载汇总")
    print("=" * 60)
    success = sum(1 for _, ok, _ in results if ok)
    failed_required = [(r,) for r, ok, req in results if not ok and req]
    for rel_remote, ok, required in results:
        mark = "✅" if ok else ("⚠️" if not required else "❌")
        print(f"  {mark} {rel_remote}")

    print(f"\n成功: {success}/{len(results)}")
    if failed_required:
        print(f"⚠️ {len(failed_required)} 个必须文件下载失败")
        sys.exit(1)
    print("✅ 所有关键文件下载完成")


if __name__ == "__main__":
    main()
