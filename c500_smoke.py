import os, sys
# ★ 关键: MXMACA Triton后端需要MACA_PATH环境变量
# source /opt/maca/env.sh 没设这个变量, 必须手动设置
os.environ['MACA_PATH'] = '/opt/maca'
os.environ['MACA_CLANG_PATH'] = '/opt/maca/mxgpu_llvm/bin'
os.environ['LD_LIBRARY_PATH'] = '/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib'
sys.path.insert(0, '/data/TriHelix-Mamba')

print("=== Python ===")
print(f"Python: {sys.executable}")

print("\n=== Torch ===")
try:
    import torch
    print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
except Exception as e:
    print(f"torch FAIL: {e}")

print("\n=== mamba_ssm ===")
try:
    import mamba_ssm
    print(f"mamba_ssm: {mamba_ssm.__version__}")
except Exception as e:
    print(f"mamba_ssm FAIL: {e}")

print("\n=== Mamba2 ===")
try:
    from mamba_ssm import Mamba2
    print("Mamba2 OK")
except Exception as e:
    print(f"Mamba2 FAIL: {e}")

print("\n=== Mamba3 ===")
try:
    from mamba_ssm import Mamba3
    print("Mamba3 OK")
except Exception as e:
    print(f"Mamba3 FAIL: {e}")

print("\n=== Mamba3 forward test ===")
try:
    from mamba_ssm import Mamba3
    m = Mamba3(d_model=64, d_state=64, expand=2, headdim=64,
               ngroups=1, is_mimo=False, chunk_size=64, is_outproj_norm=False).cuda()
    x = torch.randn(2, 100, 64, device='cuda', requires_grad=True)
    y = m(x)
    y.sum().backward()
    print(f"Mamba3 GPU forward+backward OK: y={y.shape}, grad={x.grad.mean().item():.4f}")
except Exception as e:
    print(f"Mamba3 test FAIL: {type(e).__name__}: {str(e)[:300]}")

print("\n=== PairwiseBasePairMamba3 ===")
try:
    from models.three_chain_mamba3 import PairwiseBasePairMamba3, ThreeChainMamba3
    bp = PairwiseBasePairMamba3(64).cuda()
    x_s = torch.randn(2, 100, 36, 64, device='cuda', requires_grad=True)
    x_t = torch.randn(2, 100, 36, 64, device='cuda', requires_grad=True)
    c_inject = torch.randn(2, 100, 1, 64, device='cuda', requires_grad=True)
    x_ref = torch.randn(2, 100, 36, 64, device='cuda')
    x_s_new, x_t_new, c_new = bp(x_ref, x_s, x_t, c_inject)
    loss = x_s_new.sum() + x_t_new.sum() + c_new.sum()
    loss.backward()
    print(f"PairwiseBasePairMamba3 OK: alpha={bp.alpha.item():.6f}, grad={bp.alpha.grad.item():.6f}")

    # Full model
    model = ThreeChainMamba3(cell_types=16, action_dim=5, d_model=64, n_layers=1,
                             enable_bp=True).cuda()
    S_0 = torch.zeros(2, 6, 6, dtype=torch.long, device='cuda')
    S_0[0, 0, 0] = 1; S_0[1, 1, 1] = 2
    actions = torch.randint(0, 5, (2, 4, 100), device='cuda')
    logits, info = model(S_0, actions)
    print(f"ThreeChainMamba3+BP forward OK: logits={logits.shape}")
    S_t = torch.randint(0, 16, (2, 101, 6, 6), device='cuda')
    loss, li = model.loss(logits, S_t, info, aux_weight=0.3)
    loss.backward()
    alphas = [bp.alpha.item() for bp in model.mamba.base_pair]
    print(f"ThreeChainMamba3+BP backward OK: loss={loss.item():.4f}, alpha={alphas}")
except Exception as e:
    import traceback
    print(f"BP test FAIL: {type(e).__name__}: {str(e)[:300]}")
    traceback.print_exc()

print("\n=== scipy ===")
try:
    import scipy
    print(f"scipy: {scipy.__version__}")
except ImportError:
    print("scipy: not installed")

print("\n=== DONE ===")
