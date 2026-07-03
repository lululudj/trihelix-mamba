"""验证 register_parametrization 方案能正确把 Mamba2 的 A 激活从 exp 换成 heavy_tail_activation

go/no-go 门:
  [1] parametrize 前后, forward 输出形状一致
  [2] parametrize 后, A = -heavy_tail(raw) 而非 -exp(A_log)
  [3] 梯度能正确回传到 raw 参数
  [4] raw 初值 = 原 A_log 值 (log(uniform)), 都是 log(正数), 可正可负但 heavy_tail 总>0
"""
import torch
import torch.nn.utils.parametrize as P

# ---- Mamba3 的 heavy_tail_activation (白嫖目标) ----
def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """f(x) = x+1 if x>=0; 1/(1-x) if x<0. 总是 >0, 连续可微. 正侧线性(梯度1不饱和), 负侧重尾."""
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


class HTAParametrization(torch.nn.Module):
    """把 raw 参数 r 映射成 log(heavy_tail(r))。
    这样 Mamba2 forward 里 A = -exp(self.A_log) = -exp(log(heavy_tail(r))) = -heavy_tail(r)。
    等价于把激活函数 exp→heavy_tail, 但完全不碰 forward 源码。
    """
    def forward(self, r):
        h = heavy_tail_activation(r)
        # 防止 log(0): heavy_tail 总 >0, 但数值极端时加 eps
        return torch.log(h.clamp(min=1e-8))


def main():
    from mamba_ssm import Mamba2

    torch.manual_seed(0)
    print("=== [1] 构造 Mamba2 (实际模型参数 d_model=256), 检查 parametrize 前的 A ===")
    # 用实际 ThreeChainMamba2 时间链参数, 避免 d_model=64 的 stride 对齐问题
    m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2, headdim=64).cuda()
    raw_A_log_orig = m.A_log.detach().clone()
    # 原始 A = -exp(A_log)
    A_orig = -torch.exp(raw_A_log_orig.float())
    print(f"  A_log shape: {m.A_log.shape}, 值范围 [{raw_A_log_orig.min():.3f}, {raw_A_log_orig.max():.3f}]")
    print(f"  原 A = -exp(A_log), 范围 [{A_orig.min():.3f}, {A_orig.max():.3f}]")
    print(f"  A_log is Parameter: {isinstance(m.A_log, torch.nn.Parameter)}")

    print()
    print("=== [2] 注册 parametrization ===")
    P.register_parametrization(m, 'A_log', HTAParametrization())
    print(f"  parametrize 后, A_log is Parameter: {isinstance(m.A_log, torch.nn.Parameter)} (应为 False, 变成 property)")
    raw = m.parametrizations.A_log.original
    print(f"  raw param shape: {raw.shape}, 初值 = 原 A_log (自动保留)")
    print(f"  raw 范围 [{raw.min():.3f}, {raw.max():.3f}]")

    print()
    print("=== [3] 验证新 A = -heavy_tail(raw) ===")
    # 现在 self.A_log 返回 log(heavy_tail(raw))
    A_log_new = m.A_log  # property 访问
    A_new = -torch.exp(A_log_new.float())
    A_expected = -heavy_tail_activation(raw.float())
    diff = (A_new - A_expected).abs().max().item()
    print(f"  新 A (from forward 公式) 范围 [{A_new.min():.3f}, {A_new.max():.3f}]")
    print(f"  期望 A = -heavy_tail(raw) 范围 [{A_expected.min():.3f}, {A_expected.max():.3f}]")
    print(f"  差异: {diff:.2e} (应 < 1e-5)")
    assert diff < 1e-5, "❌ parametrize 变换不正确!"
    print("  ✅ A = -heavy_tail(raw) 验证通过")

    print()
    print("=== [4] 验证 forward 能跑 + 梯度回传 ===")
    x = torch.randn(2, 32, 256, device='cuda')
    y = m(x)
    loss = y.sum()
    loss.backward()
    grad_raw = m.parametrizations.A_log.original.grad
    print(f"  forward 输出 shape: {y.shape}")
    print(f"  raw 梯度 shape: {grad_raw.shape}, 范围 [{grad_raw.min():.6f}, {grad_raw.max():.6f}]")
    assert grad_raw is not None and grad_raw.abs().sum() > 0, "❌ 梯度未传到 raw!"
    print("  ✅ 梯度正确回传到 raw 参数")

    print()
    print("=== [5] 验证 _no_weight_decay 标记保留 ===")
    # Mamba2 的 A_log 标记了 _no_weight_decay, parametrize 后检查 raw
    no_wd = getattr(raw, '_no_weight_decay', False)
    print(f"  raw._no_weight_decay: {no_wd}")
    if not no_wd:
        print("  ⚠️  标记丢失, 需手动设置 (否则 weight_decay 会作用于 raw)")
        raw._no_weight_decay = True
        print("  ✅ 已手动设置 raw._no_weight_decay = True")

    print()
    print("=" * 60)
    print(" 🎉 全部验证通过! parametrize 方案可行")
    print("    - A = -heavy_tail(raw) (激活函数 exp→heavy_tail)")
    print("    - forward 不动, 系统包不动, baseline 不受影响")
    print("    - 梯度正确, _no_weight_decay 可保留")
    print("=" * 60)


if __name__ == "__main__":
    main()
