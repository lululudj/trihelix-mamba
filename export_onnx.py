"""导出 ThreeChainMamba2 到 ONNX + CPU 推理验证

阶段 2 边缘设备专精: 把模型导出 ONNX, 验证 CPU 推理可行。
卖点: 3.4M 参数长程状态预测, OOD 不崩, 边缘设备可跑。

注意: Mamba2 的 triton 内核不支持 ONNX tracing, 需要 CPU fallback。
本脚本用 CPU 模式导出, 验证推理可行性。
"""
import sys
import time
import torch
import numpy as np

sys.path.insert(0, "/mnt/e/three_chain_v3")

from models.three_chain_mamba2 import ThreeChainMamba2
from utils import count_params

print("=" * 60)
print("  ONNX 导出 + CPU 推理验证")
print("=" * 60)

# 1. 构建模型 (CPU)
device = torch.device("cpu")
model = ThreeChainMamba2(cell_types=16, action_dim=5, d_model=256, n_layers=2).to(device)
model.eval()
n_params = count_params(model)
print(f"\n模型: ThreeChainMamba2")
print(f"参数: {n_params/1e6:.2f}M")
print(f"设备: {device}")

# 2. 构造 dummy input
B, N, K, T = 1, 8, 4, 20  # 边缘设备典型输入
S_0 = torch.randint(0, 16, (B, N, N), device=device)
actions = torch.randint(0, 5, (B, K, T), device=device)
print(f"\n输入: B={B}, N={N}, K={K}, T={T}")

# 3. PyTorch CPU 推理基准
print("\n--- PyTorch CPU 推理基准 ---")
with torch.no_grad():
    # warmup
    for _ in range(3):
        _ = model(S_0, actions)
    # 计时
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        logits, info = model(S_0, actions)
        times.append(time.perf_counter() - t0)

pt_mean = np.mean(times) * 1000
pt_std = np.std(times) * 1000
print(f"PyTorch CPU: {pt_mean:.1f} ± {pt_std:.1f} ms / step")
print(f"输出: logits {tuple(logits.shape)}")

# 4. 尝试 ONNX 导出
print("\n--- ONNX 导出 ---")
try:
    import onnx
    import onnxruntime as ort

    onnx_path = "/mnt/e/three_chain_v3/three_chain_mamba2.onnx"

    # Mamba2 用 triton 内核, ONNX tracing 可能失败
    # 尝试导出, 失败则记录原因
    torch.onnx.export(
        model,
        (S_0, actions),
        onnx_path,
        opset_version=17,
        input_names=["S_0", "actions"],
        output_names=["logits"],
        dynamic_axes={
            "S_0": {0: "batch", 1: "N", 2: "N"},
            "actions": {0: "batch", 1: "K", 2: "T"},
            "logits": {0: "batch", 1: "T"},
        },
    )
    print(f"✅ ONNX 导出成功: {onnx_path}")

    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX 模型验证通过")

    # ONNX Runtime 推理
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_feed = {"S_0": S_0.numpy(), "actions": actions.numpy()}

    # warmup
    for _ in range(3):
        sess.run(None, input_feed)

    # 计时
    ort_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        ort_logits = sess.run(None, input_feed)[0]
        ort_times.append(time.perf_counter() - t0)

    ort_mean = np.mean(ort_times) * 1000
    ort_std = np.std(ort_times) * 1000
    print(f"ONNX CPU: {ort_mean:.1f} ± {ort_std:.1f} ms / step")

    # 数值一致性检查
    pt_logits = logits.numpy()
    max_diff = np.abs(pt_logits - ort_logits).max()
    print(f"数值差异: max |pt - ort| = {max_diff:.6e}")

except Exception as e:
    print(f"❌ ONNX 导出失败: {type(e).__name__}: {e}")
    print("\n原因分析: Mamba2 使用 triton 自定义内核, ONNX tracing 不支持。")
    print("解决方案: 1) 用 PyTorch CPU 推理 (已验证可行)")
    print("         2) 用 torch.jit.trace (script) 代替 onnx.export")
    print("         3) 把 Mamba2 换成纯 PyTorch 实现的 S4/S6 扫描")

    # 即使 ONNX 失败, PyTorch CPU 推理也是边缘部署的可行路径
    print(f"\n✅ PyTorch CPU 推理可行: {pt_mean:.1f} ms / step ({n_params/1e6:.2f}M 参数)")

# 5. 内存占用
import os
import psutil
proc = psutil.Process()
mem_mb = proc.memory_info().rss / 1024 / 1024
print(f"\n内存占用: {mem_mb:.0f} MB")

print("\n" + "=" * 60)
print("  结论")
print("=" * 60)
print(f"  参数: {n_params/1e6:.2f}M (边缘设备友好)")
print(f"  CPU 推理: {pt_mean:.1f} ms / step")
print(f"  → 边缘设备可跑 ({1000/pt_mean:.0f} FPS)")
