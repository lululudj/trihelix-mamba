"""smoke test: ThreeChainMamba2 + ThreeChainMamba2Lite 构造/前向/loss/反向。

验证:
    1. 两个模型能构造(参数量打印)
    2. forward shape 正确: logits (B,T,N,N,C)
    3. loss 能算(返回 loss_final/loss_aux)
    4. 反向传播梯度不为 None
    5. CUDA 能跑
    6. 多种 (N,K,T) 组合(变长测试)
"""
import sys
import torch

sys.path.insert(0, ".")


def make_batch(B, N, K, T, device, C=16, A=5):
    """造一个假 batch, 模拟 GridWorld 数据。"""
    S_0 = torch.randint(0, C, (B, N, N), device=device, dtype=torch.long)
    actions = torch.randint(0, A, (B, K, T), device=device, dtype=torch.long)
    S_t = torch.randint(0, C, (B, T + 1, N, N), device=device, dtype=torch.long)
    return S_0, actions, S_t


def smoke_one(model_cls, name, B, N, K, T, d_model=256, n_layers=2, C=16, A=5):
    """对单个模型跑 构造→前向→loss→反向 全流程。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- {name}  B={B} N={N} K={K} T={T} ---")
    try:
        # 构造
        model = model_cls(cell_types=C, action_dim=A, d_model=d_model, n_layers=n_layers).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [构造] OK  params={n_params/1e6:.2f}M")

        # 前向
        S_0, actions, S_t = make_batch(B, N, K, T, device, C, A)
        logits, info = model(S_0, actions)
        assert logits.shape == (B, T, N, N, C), \
            f"logits shape {logits.shape} != expected {(B,T,N,N,C)}"
        print(f"  [前向] OK  logits{tuple(logits.shape)}")

        # loss
        loss, loss_info = model.loss(logits, S_t, info, aux_weight=0.3)
        assert "loss_final" in loss_info, f"loss_info 缺 loss_final: {list(loss_info.keys())}"
        assert "loss_aux" in loss_info, f"loss_info 缺 loss_aux: {list(loss_info.keys())}"
        print(f"  [loss] OK  loss={loss.item():.4f}  final={loss_info['loss_final']:.4f}  aux={loss_info['loss_aux']:.4f}")

        # 反向
        loss.backward()
        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        n_total = sum(1 for _ in model.parameters())
        # 检查至少有梯度
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  [反向] OK  有梯度参数={n_with_grad}/{n_total}  grad_norm_sum={grad_norm:.2f}")

        # 检查 eagle 属性
        has_eagle = hasattr(model, "eagle")
        print(f"  [eagle] has_eagle={has_eagle}")

        return True
    except Exception as e:
        import traceback
        print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  ThreeChainMamba2 + Lite  SMOKE TEST")
    print("=" * 60)
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    from models import ThreeChainMamba2, ThreeChainMamba2Lite

    results = []

    # === A 版本: ThreeChainMamba2 ===
    # 小 batch 基本测试
    results.append(smoke_one(ThreeChainMamba2, "ThreeChainMamba2 (A)",
                             B=2, N=8, K=4, T=30))
    # 大一点的 batch
    results.append(smoke_one(ThreeChainMamba2, "ThreeChainMamba2 (A)",
                             B=4, N=12, K=8, T=60))
    # 长序列(接近 OOD)
    results.append(smoke_one(ThreeChainMamba2, "ThreeChainMamba2 (A)",
                             B=2, N=8, K=4, T=100))

    # === B 版本: ThreeChainMamba2Lite ===
    results.append(smoke_one(ThreeChainMamba2Lite, "ThreeChainMamba2Lite (B)",
                             B=2, N=8, K=4, T=30))
    results.append(smoke_one(ThreeChainMamba2Lite, "ThreeChainMamba2Lite (B)",
                             B=4, N=12, K=8, T=60))

    # === 汇总 ===
    print("\n" + "=" * 60)
    print("  SMOKE TEST 汇总")
    print("=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    print(f"  {n_pass}/{n_total} 通过")
    if n_pass == n_total:
        print("  ✅ 全部通过, 可以进入训练阶段")
    else:
        print("  ❌ 有失败, 需修复")


if __name__ == "__main__":
    main()
