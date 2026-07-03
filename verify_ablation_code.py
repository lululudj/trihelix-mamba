"""阶段 3.2 代码修改验证脚本

验证:
1. import 无语法错误
2. build_model 用 ablation config 正确构建 (ablate 开关生效)
3. build_model 用 default config (ablate=False) 向后兼容
4. 加载旧 checkpoint 不报 missing/unexpected keys (无新参数, 只加控制开关)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, build_model
import torch


def test_import():
    """测试 1: import 无语法错误"""
    print("[测试 1] import 模型...")
    from models.three_chain_mamba2 import HeteroMamba2, ThreeChainMamba2
    # 检查 HeteroMamba2 有 ablate 开关
    import inspect
    sig = inspect.signature(HeteroMamba2.__init__)
    params = list(sig.parameters.keys())
    assert "ablate_s" in params, f"HeteroMamba2.__init__ 缺 ablate_s 参数: {params}"
    assert "ablate_t" in params, f"HeteroMamba2.__init__ 缺 ablate_t 参数: {params}"
    assert "ablate_c" in params, f"HeteroMamba2.__init__ 缺 ablate_c 参数: {params}"
    print(f"  [OK] HeteroMamba2.__init__ 参数: {params}")

    sig2 = inspect.signature(ThreeChainMamba2.__init__)
    params2 = list(sig2.parameters.keys())
    assert "ablate_s" in params2, f"ThreeChainMamba2.__init__ 缺 ablate_s 参数: {params2}"
    print(f"  [OK] ThreeChainMamba2.__init__ 参数: {params2}")
    print()


def test_build_ablation():
    """测试 2: build_model 用 ablation config 正确构建"""
    print("[测试 2] build_model 读 ablation config...")
    configs = [
        ("configs/ablation_no_spatial.yaml",  True,  False, False),
        ("configs/ablation_no_temporal.yaml", False, True,  False),
        ("configs/ablation_no_causal.yaml",   False, False, True),
        ("configs/ablation_no_all.yaml",      True,  True,  True),
    ]
    base = Path(__file__).parent
    for cfg_path, exp_s, exp_t, exp_c in configs:
        cfg = load_config(str(base / cfg_path))
        model = build_model("three_chain_mamba2", cfg)
        assert model.mamba.ablate_s == exp_s, f"{cfg_path}: ablate_s={model.mamba.ablate_s}, 期望 {exp_s}"
        assert model.mamba.ablate_t == exp_t, f"{cfg_path}: ablate_t={model.mamba.ablate_t}, 期望 {exp_t}"
        assert model.mamba.ablate_c == exp_c, f"{cfg_path}: ablate_c={model.mamba.ablate_c}, 期望 {exp_c}"
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [OK] {cfg_path}: ablate_s={model.mamba.ablate_s}, "
              f"ablate_t={model.mamba.ablate_t}, ablate_c={model.mamba.ablate_c}, "
              f"params={n_params/1e6:.2f}M")
    print()


def test_backward_compat():
    """测试 3: default config (无 ablate 字段) 向后兼容"""
    print("[测试 3] 向后兼容 (default config 无 ablate 字段)...")
    base = Path(__file__).parent
    cfg = load_config(str(base / "configs/matched_mamba2.yaml"))
    # 确认 default config 没有 ablate 字段
    assert "ablate_s" not in cfg["model"], "matched_mamba2.yaml 不应有 ablate_s"
    model = build_model("three_chain_mamba2", cfg)
    assert model.mamba.ablate_s == False, "默认应为 False"
    assert model.mamba.ablate_t == False
    assert model.mamba.ablate_c == False
    print(f"  [OK] default config: ablate_s/t/c 全 False (向后兼容)")
    print()


def test_param_count_unchanged():
    """测试 4: ablate 不改变参数量 (置零输出, 不删模块)"""
    print("[测试 4] 参数量不变 (置零输出不删模块)...")
    base = Path(__file__).parent
    cfg_base = load_config(str(base / "configs/matched_mamba2_30m.yaml"))
    cfg_abl = load_config(str(base / "configs/ablation_no_all.yaml"))
    model_base = build_model("three_chain_mamba2", cfg_base)
    model_abl = build_model("three_chain_mamba2", cfg_abl)
    n_base = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    n_abl = sum(p.numel() for p in model_abl.parameters() if p.requires_grad)
    assert n_base == n_abl, f"参数量不一致: base={n_base}, ablation={n_abl}"
    print(f"  [OK] 参数量相同: {n_base/1e6:.2f}M == {n_abl/1e6:.2f}M")
    print()


def test_forward_ablate_all():
    """测试 5: ablate_all 时 forward 不报错 (只剩残差)"""
    print("[测试 5] forward (ablate_all, 只剩残差)...")
    base = Path(__file__).parent
    cfg = load_config(str(base / "configs/ablation_no_all.yaml"))
    model = build_model("three_chain_mamba2", cfg)
    # 小输入测试 forward
    B, N, K, T = 1, 6, 4, 10
    S_0 = torch.zeros(B, N, N, dtype=torch.long)
    actions = torch.zeros(B, K, T, dtype=torch.long)
    logits, info = model(S_0, actions)
    assert logits.shape == (B, T, N, N, 16), f"logits shape 错: {logits.shape}"
    print(f"  [OK] forward 成功, logits shape={logits.shape}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("阶段 3.2 代码修改验证")
    print("=" * 60)
    print()
    test_import()
    test_build_ablation()
    test_backward_compat()
    test_param_count_unchanged()
    test_forward_ablate_all()
    print("=" * 60)
    print("[全部通过] 代码修改验证成功, 可以上云端跑训练")
    print("=" * 60)
