"""Mamba2 构造验证脚本（写架构代码前的预检）。

验证计划中的三链 Mamba2 参数组合能否正确构造 + forward + 双向 flip。
任何一项 FAIL 就必须在写架构代码前解决。
"""
import sys
import torch

try:
    from mamba_ssm import Mamba2
    print(f"[ok] mamba_ssm.Mamba2 导入成功  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
except ImportError as e:
    print(f"[FATAL] mamba_ssm 不可用: {e}")
    sys.exit(1)


def try_make(d_model, d_state, d_conv, expand, headdim, tag,
             seq_len=50, batch=2, bidir=True):
    """构造一个 Mamba2，跑 forward + 反向 flip 双向测试。"""
    try:
        m = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        ).cuda()
        x = torch.randn(batch, seq_len, d_model).cuda()
        y = m(x)
        # 双向：y_bi = y + flip(m(flip(x)))
        if bidir:
            y_back = m(torch.flip(x, dims=[1]))
            y_bi = y + torch.flip(y_back, dims=[1])
        else:
            y_bi = None
        n = sum(p.numel() for p in m.parameters())
        bi_shape = tuple(y_bi.shape) if y_bi is not None else None
        print(f"  [ok] {tag}: in{tuple(x.shape)}->out{tuple(y.shape)} bi{bi_shape} params={n:,}")
        return True
    except Exception as e:
        print(f"  [FAIL] {tag}: {type(e).__name__}: {str(e)[:200]}")
        return False


def main():
    print("\n=== 1. 修正后三链参数（d_model=256） ===")
    # 空间链: headdim=32 规避 causal_conv1d stride 对齐问题
    #   expand=1 → d_inner=256, nheads=256*1/32=8
    ok_s = try_make(256, 128, 4, 1, 32, "空间链(expand=1,headdim=32,d_state=128)")
    # 时间链: d_state=64, expand=2, headdim=64 → nheads=256*2/64=8
    ok_t = try_make(256, 64, 4, 2, 64, "时间链(expand=2,headdim=64,d_state=64)")
    # 因果链: d_state=32, expand=2, headdim=64 → nheads=8
    ok_c = try_make(256, 32, 4, 2, 64, "因果链(expand=2,headdim=64,d_state=32)")

    print("\n=== 2. 空间链 headdim=32 关键场景 ===")
    # T=150 长序列（OOD 外推长度）
    try_make(256, 128, 4, 1, 32, "空间链 T=150 双向", seq_len=150, bidir=True)
    # 大 batch reshape: 空间链实际用法 reshape(B*T, N², d)
    # B=8, T=100 → 800; N=8 → N²=64
    try_make(256, 128, 4, 1, 32, "空间链 batch=800 seq=64",
             seq_len=64, batch=800, bidir=True)
    # 更大 batch: B=8, T=150 → 1200
    try_make(256, 128, 4, 1, 32, "空间链 batch=1200 seq=64",
             seq_len=64, batch=1200, bidir=True)
    # N=12 → N²=144
    try_make(256, 128, 4, 1, 32, "空间链 seq=144(N=12)",
             seq_len=144, batch=100, bidir=True)

    print("\n=== 3. 时间链 T=150（OOD 长度） ===")
    try_make(256, 64, 4, 2, 64, "时间链 T=150 因果", seq_len=150, bidir=False)
    # 时间链大 batch: reshape(B*N², T, d), B=8, N=8 → 512
    try_make(256, 64, 4, 2, 64, "时间链 batch=512 seq=150",
             seq_len=150, batch=512, bidir=False)

    print("\n=== 4. 因果链 K 维扫描 ===")
    # 因果链: 对 act_emb (B, K, T, d) 的 K 维扫描
    # reshape(B*T, K, d), B=8, T=100 → 800, K=12
    try_make(256, 32, 4, 2, 64, "因果链 batch=800 seq=12(K=12)",
             seq_len=12, batch=800, bidir=False)

    print("\n=== 5. d_conv 边界确认（只支持 2-4） ===")
    for dc in [2, 3, 4]:
        try_make(256, 64, dc, 2, 64, f"d_conv={dc}", seq_len=30, bidir=False)

    print("\n=== 汇总 ===")
    print(f"  空间链(headdim=32): {'PASS' if ok_s else 'FAIL'}")
    print(f"  时间链(headdim=64): {'PASS' if ok_t else 'FAIL'}")
    print(f"  因果链(headdim=64): {'PASS' if ok_c else 'FAIL'}")
    if ok_s and ok_t and ok_c:
        print("\n  ✅ 三链修正参数全部可用，可以开始写架构代码")
    else:
        print("\n  ❌ 仍有核心参数 FAIL，需继续调整")


if __name__ == "__main__":
    main()
