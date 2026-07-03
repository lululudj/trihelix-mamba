"""ThreeChainMamba3 端到端 smoke test（小输入，定位 GPU 稳定性问题）。

用比 train.py 小的输入测 forward + loss + backward:
    B=1, N=6 (N²=36), K=4, T=30
    空间链 L=36, 时间链 L=30, 因果链 L=4

如果小输入也报 "CUDA driver error: device not ready"，
说明 mamba3 的 Python for-loop 在 WSL2 GPU 上不稳定，需考虑放弃或改 CPU。
"""
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ThreeChainMamba3 参数量: {n_params:,}")

    # 小输入: B=1, N=6, K=4, T=30
    B, N, K, T = 1, 6, 4, 30
    S_0 = torch.randint(0, 16, (B, N, N), device=device, dtype=torch.long)
    actions = torch.randint(0, 5, (B, K, T), device=device, dtype=torch.long)
    S_t = torch.randint(0, 16, (B, T + 1, N, N), device=device, dtype=torch.long)

    print(f"输入: S_0={tuple(S_0.shape)}, actions={tuple(actions.shape)}, S_t={tuple(S_t.shape)}")

    # forward
    print("\n--- forward ---")
    t0 = time.time()
    try:
        logits, info = model(S_0, actions)
        torch.cuda.synchronize() if device.type == "cuda" else None
        print(f"[OK] forward 跑通 ({time.time()-t0:.2f}s)")
        print(f"     logits shape={tuple(logits.shape)}")
        print(f"     均值={logits.mean().item():.4f}")
    except RuntimeError as e:
        print(f"[FAIL] forward 失败: {e}")
        if "device not ready" in str(e):
            print("\n>>> 结论: mamba3 的 Python for-loop 在 WSL2 GPU 上不稳定 <<<")
            print(">>> 即使小输入也报 'device not ready'，建议放弃 GPU，考虑 CPU 或放弃 mamba3 <<<")
        sys.exit(1)

    # loss
    print("\n--- loss ---")
    try:
        total, loss_info = model.loss(logits, S_t, info)
        print(f"[OK] loss 跑通, total={total.item():.4f}, info={loss_info}")
    except Exception as e:
        print(f"[FAIL] loss 失败: {e}")
        sys.exit(1)

    # backward
    print("\n--- backward ---")
    try:
        model.zero_grad(set_to_none=True)
        t0 = time.time()
        total.backward()
        torch.cuda.synchronize() if device.type == "cuda" else None
        print(f"[OK] backward 跑通 ({time.time()-t0:.2f}s)")
        # 抽查梯度
        g = model.mamba.mamba_t[0].in_proj.weight.grad
        print(f"     时间链 in_proj.grad 范数={g.float().norm().item():.4f}")
    except RuntimeError as e:
        print(f"[FAIL] backward 失败: {e}")
        sys.exit(1)

    print("\n>>> smoke test 全部通过 <<<")


if __name__ == "__main__":
    main()
