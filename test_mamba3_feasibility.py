"""Mamba3 可行性测试：go/no-go 门。

试验出真知 —— 不试不知道 Mamba3 的 Python 顺序扫描到底有多慢。
五件事：
    1. einops 依赖检查（mamba3_ref.py import 需要）
    2. Mamba3 实例化 + forward 是否能跑通
    3. Mamba3 vs Mamba2 速度对比（同形状 B=2, L=150, d=256）
    4. Mamba3 backward 是否能跑（梯度有值）
    5. go/no-go 判断：
         forward 能跑 + backward 能跑 + 速度比 ≤ 50x  → GO
         否则                                          → NO-GO（放弃，理由会写明）
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def section(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------- 1. einops 依赖检查 ----------
    section("1. einops 依赖检查")
    try:
        import einops
        ver = getattr(einops, "__version__", "unknown")
        print(f"[OK] einops 已装, 版本={ver}")
    except ImportError as e:
        print(f"[FAIL] einops 未装: {e}")
        print("装一下: pip install einops")
        sys.exit(1)

    # ---------- 2. Mamba3 实例化 + forward ----------
    section("2. Mamba3 实例化 + forward")
    from models.mamba3_ref import Mamba3

    # 同 Mamba2 时间链形状: d_model=256, d_state=64, expand=2, headdim=32
    # 注意: mamba3 没有 d_conv 参数（用梯形离散化代替）
    m3 = Mamba3(
        d_model=256, d_state=64, expand=2, headdim=32,
        ngroups=1, is_mimo=False,
    ).to(device)
    n_params_m3 = sum(p.numel() for p in m3.parameters())
    print(f"Mamba3 参数量: {n_params_m3:,}")

    x = torch.randn(2, 150, 256, device=device)
    try:
        y = m3(x)
        print(f"[OK] forward 跑通, 输出 shape={tuple(y.shape)}")
        print(f"     均值={y.mean().item():.4f}, 标准差={y.std().item():.4f}")
    except Exception as e:
        print(f"[FAIL] forward 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---------- 3. 速度对比 Mamba3 vs Mamba2 ----------
    section("3. 速度对比 Mamba3 vs Mamba2")
    m2 = None
    ratio = float("inf")
    try:
        from models.three_chain_mamba2 import make_mamba2
        m2 = make_mamba2(d_model=256, d_state=64, d_conv=4, expand=2, headdim=32).to(device)
        n_params_m2 = sum(p.numel() for p in m2.parameters())
        print(f"Mamba2 参数量: {n_params_m2:,}")
    except Exception as e:
        print(f"[WARN] Mamba2 构造失败，跳过速度对比: {e}")

    if m2 is not None:
        # warmup（首轮含编译/cache 开销，扔掉）
        for _ in range(3):
            _ = m3(x)
            _ = m2(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Mamba3 测速
        t0 = time.time()
        for _ in range(5):
            _ = m3(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_m3 = (time.time() - t0) / 5

        # Mamba2 测速
        t0 = time.time()
        for _ in range(5):
            _ = m2(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_m2 = (time.time() - t0) / 5

        ratio = t_m3 / t_m2 if t_m2 > 0 else float("inf")
        print(f"Mamba3 forward: {t_m3 * 1000:.1f} ms/次")
        print(f"Mamba2 forward: {t_m2 * 1000:.1f} ms/次")
        print(f"速度比 Mamba3/Mamba2 = {ratio:.1f}x")

    # ---------- 4. backward 检查 ----------
    section("4. Mamba3 backward 检查")
    try:
        # 清掉 forward 残留的图
        m3.zero_grad(set_to_none=True)
        y = m3(x)
        loss = y.float().sum()
        loss.backward()
        g_in = m3.in_proj.weight.grad
        g_out = m3.out_proj.weight.grad
        print(f"[OK] backward 跑通")
        print(f"     in_proj.grad  shape={tuple(g_in.shape)},  范数={g_in.float().norm().item():.4f}")
        print(f"     out_proj.grad shape={tuple(g_out.shape)}, 范数={g_out.float().norm().item():.4f}")
        if g_in.float().norm().item() == 0 or g_out.float().norm().item() == 0:
            print("[WARN] 梯度为 0，可能有断梯度问题")
    except Exception as e:
        print(f"[FAIL] backward 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---------- 5. go/no-go 判断 ----------
    section("5. go/no-go 判断")
    ok = True
    reasons = []
    if ratio > 50:
        ok = False
        reasons.append(f"速度太慢: Mamba3/Mamba2 = {ratio:.1f}x > 50x")

    print(f"参数量: Mamba3={n_params_m3:,}")
    if m2 is not None:
        print(f"参数量: Mamba2={n_params_m2:,}  (比值 {n_params_m3 / n_params_m2:.2f}x)")
    print()
    if ok:
        print(">>> 结论: [GO] 继续搭 three_chain_mamba3 变体 <<<")
    else:
        print(">>> 结论: [NO-GO] 放弃 mamba3，原因如下 <<<")
        for r in reasons:
            print(f"  - {r}")
        print("建议: 不用 mamba3 reference 实现，保持 Mamba2 baseline。")


if __name__ == "__main__":
    main()
