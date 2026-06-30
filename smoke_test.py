"""烟雾测试：所有模型能跑通 forward + backward + loss"""
import torch
from models import MODEL_REGISTRY


def main():
    torch.manual_seed(0)
    S_0 = torch.randint(0, 12, (4, 8, 8))
    actions = torch.randint(0, 5, (4, 2, 10))
    S_t = torch.randint(0, 12, (4, 11, 8, 8))

    print(f"{'model':<25} {'params':>10} {'logits_shape':<25} {'loss':>10}")
    print("-" * 75)
    for name, ModelCls in MODEL_REGISTRY.items():
        m = ModelCls()
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        logits, info = m(S_0, actions)
        loss, _ = m.loss(logits, S_t)
        loss.backward()
        print(f"{name:<25} {n_params:>8.2f}M {str(tuple(logits.shape)):<25} {loss.item():>10.4f}")


if __name__ == "__main__":
    main()
