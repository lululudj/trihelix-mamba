"""阶段 A 改造后的 import 和实例化测试（不依赖 GPU，纯 CPU 验证）。"""
import sys
sys.path.insert(0, '.')

print("=== Test 1: baseline 开关关闭（应等价于原版）===")
from models.three_chain_mamba2 import ThreeChainMamba2
m1 = ThreeChainMamba2(enable_m3=False, enable_jepa=False)
p1 = sum(p.numel() for p in m1.parameters())
print(f"  baseline params = {p1}")
assert not hasattr(m1, 'm3'), "enable_m3=False 不应实例化 self.m3"
assert not hasattr(m1, 'jepa'), "enable_jepa=False 不应实例化 self.jepa"
print("  OK: 无 m3/jepa 模块")

print("\n=== Test 2: 开关全开（应多出 JEPA 参数）===")
m2 = ThreeChainMamba2(enable_m3=True, enable_jepa=True)
p2 = sum(p.numel() for p in m2.parameters())
print(f"  m3+jepa params = {p2}")
assert hasattr(m2, 'm3'), "enable_m3=True 应实例化 self.m3"
assert hasattr(m2, 'jepa'), "enable_jepa=True 应实例化 self.jepa"
diff = p2 - p1
print(f"  JEPA 新增参数 = {diff}")
# JEPA_Predictor: GRUCell(d=256, hidden=64) + Linear(64, 256)
# GRUCell: 3*(input*hidden + hidden*hidden + 2*hidden) = 3*(256*64 + 64*64 + 2*64) = 3*20544 = 61632
# Linear: 64*256 + 256 = 16640
# 合计 ≈ 78272
assert 70000 < diff < 90000, f"JEPA 应新增约 78K 参数，实际 {diff}"
print("  OK: JEPA 参数量正确")

print("\n=== Test 3: M3SnapshotManager 文件方法存在 ===")
from models.common import M3SnapshotManager
assert hasattr(M3SnapshotManager, 'save_to_file'), "应有 save_to_file 静态方法"
assert hasattr(M3SnapshotManager, 'load_from_file'), "应有 load_from_file 静态方法"
assert M3SnapshotManager.MAGIC == b"M3SNP01\n", f"magic 错: {M3SnapshotManager.MAGIC}"
print(f"  MAGIC = {M3SnapshotManager.MAGIC!r}")
print("  OK: 文件方法就绪")

print("\n=== Test 4: build_model 从 config 取开关 ===")
from utils import build_model, load_config
cfg = load_config("configs/matched_mamba2.yaml")  # 默认全 false
m3 = build_model("three_chain_mamba2", cfg)
assert not m3.enable_m3 and not m3.enable_jepa, "matched_mamba2.yaml 应全 false"
print(f"  matched_mamba2.yaml: enable_m3={m3.enable_m3}, enable_jepa={m3.enable_jepa}")

cfg2 = load_config("configs/matched_mamba2_m3.yaml")  # 全 true
m4 = build_model("three_chain_mamba2", cfg2)
assert m4.enable_m3 and m4.enable_jepa, "matched_mamba2_m3.yaml 应全 true"
print(f"  matched_mamba2_m3.yaml: enable_m3={m4.enable_m3}, enable_jepa={m4.enable_jepa}")
print("  OK: config 开关正确传递")

print("\n=== Test 5: 其他模型不受影响（应正常构建，不传开关）===")
for name in ["three_chain_mamba2_lite", "single_chain"]:
    m = build_model(name, load_config("configs/matched_mamba2.yaml"))
    print(f"  {name}: OK, params={sum(p.numel() for p in m.parameters())}")

print("\n" + "=" * 50)
print("ALL IMPORT TESTS PASSED ✓")
print("=" * 50)
