"""阶段 A forward + loss + .m3 round-trip 端到端测试。

测试矩阵：
1. .m3 文件 round-trip（CPU 即可）—— 存了再读回，数据一致
2. baseline forward+loss（GPU）—— 开关关闭，info 无 jepa_pred，loss_info 无 jepa_loss
3. m3+jepa forward+loss（GPU）—— 开关开启，info 有 jepa_pred/target，loss_info 有 jepa_loss
"""
import sys, os
sys.path.insert(0, '.')
import torch
from models.three_chain_mamba2 import ThreeChainMamba2
from models.common import M3SnapshotManager
from utils import load_config, build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device = {device}")

# ============ Test 1: .m3 round-trip（CPU 即可）============
print("\n=== Test 1: .m3 round-trip ===")
N, K, T, d, n_layers = 8, 4, 30, 256, 2
chain_states_fake = [
    {
        "h_s": torch.randn(N * N, d),
        "h_t": torch.randn(T, d),
        "h_c": torch.randn(K, d),
    } for _ in range(n_layers)
]
meta = {"N": N, "K": K, "T": T, "d_model": d, "n_layers": n_layers,
        "step": 0, "sample_id": 0, "scenario_type": "random"}
test_path = "/tmp/test_roundtrip.m3"
M3SnapshotManager.save_to_file(test_path, meta, chain_states_fake)
print(f"  saved: {os.path.getsize(test_path)} bytes")
meta_loaded, chains_loaded = M3SnapshotManager.load_from_file(test_path)
assert meta_loaded["N"] == N
assert meta_loaded["n_layers"] == n_layers
for i, c in enumerate(chains_loaded):
    assert c["h_s"].shape == (N * N, d), f"layer {i} h_s shape: {c['h_s'].shape}"
    assert c["h_t"].shape == (T, d), f"layer {i} h_t shape: {c['h_t'].shape}"
    assert c["h_c"].shape == (K, d), f"layer {i} h_c shape: {c['h_c'].shape}"
    assert torch.allclose(c["h_s"], chain_states_fake[i]["h_s"], atol=1e-5), f"layer {i} h_s 数据不一致"
    assert torch.allclose(c["h_t"], chain_states_fake[i]["h_t"], atol=1e-5), f"layer {i} h_t 数据不一致"
    assert torch.allclose(c["h_c"], chain_states_fake[i]["h_c"], atol=1e-5), f"layer {i} h_c 数据不一致"
print("  round-trip 数据一致 OK")

# ============ Test 2 & 3: forward + loss（GPU）============
if device.type == 'cuda':
    print("\n=== Test 2: baseline forward+loss（开关关闭）===")
    B = 2
    S_0 = torch.randint(0, 16, (B, N, N), device=device)
    actions = torch.randint(0, 5, (B, K, T), device=device)
    S_t = torch.randint(0, 16, (B, T + 1, N, N), device=device)

    cfg = load_config("configs/matched_mamba2.yaml")
    model = build_model("three_chain_mamba2", cfg).to(device)
    model.eval()
    with torch.no_grad():
        logits, info = model(S_0, actions)
    print(f"  baseline logits: {logits.shape}, info keys: {list(info.keys())}")
    assert "jepa_pred" not in info, "baseline 不应有 jepa_pred"
    assert "jepa_target" not in info, "baseline 不应有 jepa_target"
    loss, loss_info = model.loss(logits, S_t, info, aux_weight=0.3)
    print(f"  baseline loss: {loss.item():.4f}, loss_info keys: {list(loss_info.keys())}")
    assert "jepa_loss" not in loss_info, "baseline 不应有 jepa_loss"
    print("  baseline forward+loss OK")

    print("\n=== Test 3: m3+jepa forward+loss（开关开启）===")
    cfg2 = load_config("configs/matched_mamba2_m3.yaml")
    model2 = build_model("three_chain_mamba2", cfg2).to(device)
    model2.eval()
    with torch.no_grad():
        logits2, info2 = model2(S_0, actions)
    print(f"  m3+jepa logits: {logits2.shape}, info keys: {list(info2.keys())}")
    assert "jepa_pred" in info2, "m3+jepa 应有 jepa_pred"
    assert "jepa_target" in info2, "m3+jepa 应有 jepa_target"
    print(f"  jepa_pred: {info2['jepa_pred'].shape}, jepa_target: {info2['jepa_target'].shape}")
    loss2, loss_info2 = model2.loss(logits2, S_t, info2, aux_weight=0.3)
    print(f"  m3+jepa loss: {loss2.item():.4f}, jepa_loss: {loss_info2['jepa_loss']:.4f}")
    assert "jepa_loss" in loss_info2, "m3+jepa 应有 jepa_loss"
    print("  m3+jepa forward+loss OK")

    # chain_states 缓存检查
    assert model2._last_chain_states is not None, "enable_m3=True 应缓存 chain_states"
    print(f"  chain_states 缓存: {len(model2._last_chain_states)} 层")
    h_s_last, h_t_last, h_c_last = model2._last_chain_states[-1]
    print(f"  最后层 h_s: {h_s_last.shape}, h_t: {h_t_last.shape}, h_c: {h_c_last.shape}")

    # ============ Test 4: save_m3 真实 round-trip ============
    print("\n=== Test 4: save_m3 真实 round-trip ===")
    import tempfile, os
    # model2 已 forward 过（上面 Test 3），_last_chain_states 已缓存，可直接 save_m3
    rt_path = os.path.join(tempfile.gettempdir(), "real_save.m3")
    model2.save_m3(rt_path, sample_id=0, N=N, K=K, T=T,
                   scenario_type="random", batch_idx=0)
    fsize = os.path.getsize(rt_path)
    print(f"  saved: {fsize} bytes ({fsize/1024:.0f}KB)")
    meta_r, chains_r = M3SnapshotManager.load_from_file(rt_path)
    assert meta_r["N"] == N, f"meta N={meta_r['N']} != {N}"
    assert meta_r["K"] == K, f"meta K={meta_r['K']} != {K}"
    assert meta_r["T"] == T, f"meta T={meta_r['T']} != {T}"
    assert meta_r["n_layers"] == 2, f"meta n_layers={meta_r['n_layers']} != 2"
    assert meta_r["sample_id"] == 0
    for i, c in enumerate(chains_r):
        assert c["h_s"].shape == (N * N, d), f"layer {i} h_s {c['h_s'].shape} != {(N*N, d)}"
        assert c["h_t"].shape == (T, d), f"layer {i} h_t {c['h_t'].shape} != {(T, d)}"
        assert c["h_c"].shape == (K, d), f"layer {i} h_c {c['h_c'].shape} != {(K, d)}"
    # 数据一致性：save_m3 存的是 detach 后的 chain_states，应与 _last_chain_states 对得上
    #   h_s 存最后时间步：_last_chain_states[i][0][0, -1] 对应 chains_r[i]["h_s"]
    for i in range(len(chains_r)):
        s_orig = model2._last_chain_states[i][0][0, -1].cpu().float()
        assert torch.allclose(chains_r[i]["h_s"], s_orig, atol=1e-5), \
            f"layer {i} h_s 数据不一致"
        c_orig = model2._last_chain_states[i][2][0, :, -1, :].cpu().float()
        assert torch.allclose(chains_r[i]["h_c"], c_orig, atol=1e-5), \
            f"layer {i} h_c 数据不一致"
    print(f"  save_m3 真实 round-trip OK, 形状+数据一致, 文件 {fsize/1024:.0f}KB")
else:
    print("\n[跳过 GPU forward 测试，device=cpu]")

print("\n" + "=" * 50)
print("ALL FORWARD+LOSS+M3 TESTS PASSED")
print("=" * 50)
