"""2 步内存测试: 确认 mamba3 OOM 是累积问题还是单步就超。

每步后强制 gc.collect() + torch.cuda.empty_cache()。
若 step 2 能过 → 是累积问题, 加 gc 即可解决。
若 step 2 仍 OOM → 单步 backward 就超 16GB, 需要梯度检查点。
"""
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    from utils import load_config, build_model

    cfg = load_config("configs/matched_mamba3.yaml")
    model = build_model("three_chain_mamba3", cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 大输入 (和 train.py 一致, N=8 T=100), 测 checkpoint 后能否扛住
    B, N, K, T = 1, 8, 4, 100
    torch.manual_seed(0)
    S_0 = torch.randint(0, 16, (B, N, N), device=device, dtype=torch.long)
    actions = torch.randint(0, 5, (B, K, T), device=device, dtype=torch.long)
    S_t = torch.randint(0, 16, (B, T + 1, N, N), device=device, dtype=torch.long)

    for step in range(1, 3):
        print(f"\n--- step {step} ---")
        t0 = time.time()
        try:
            logits, info = model(S_0, actions)
            total, loss_info = model.loss(logits, S_t, info)
            print(f"  forward+loss done ({time.time()-t0:.1f}s), loss={total.item():.4f}")

            t1 = time.time()
            total.backward()
            print(f"  backward done ({time.time()-t1:.1f}s)")

            opt.step()
            opt.zero_grad(set_to_none=True)

            if device.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU mem allocated: {mem:.2f} GB")
        except RuntimeError as e:
            print(f"[FAIL] step {step} 失败: {e}")
            if "out of memory" in str(e).lower():
                print(">>> 单步就 OOM, 需要梯度检查点 <<<")
            sys.exit(1)

        # 强制清理内存
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated() / 1024**3
            print(f"  清理后 GPU mem: {mem_after:.2f} GB")

        # 报告 RSS (进程实际内存)
        try:
            import resource
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(f"  进程峰值 RSS: {rss_kb/1024:.1f} MB")
        except Exception:
            pass

        print(f"  step {step} 总耗时: {time.time()-t0:.1f}s")

    print("\n>>> 2 步全部通过, OOM 是累积问题, 加 gc 可解决 <<<")


if __name__ == "__main__":
    main()
