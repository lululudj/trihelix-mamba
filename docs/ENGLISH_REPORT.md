# ThreeChainMamba3: Three-Chain DNA-Mamba3 with Pairwise Base Pair Coupling for Long-Range Spatiotemporal Extrapolation

> **To the Mamba3 development team at state-spaces/mamba** —
> We built upon your Mamba3 architecture (dt-RoPE complex state space + trapezoidal discretization) and extended it with a three-chain heteromorphic scanning framework inspired by DNA triple helix base pair coupling. This document summarizes our work and key findings.

---

## 1. What We Built

### 1.1 Three-Chain Architecture (ThreeChainMamba3)

We designed a **three-chain heteromorphic SSM** where three Mamba3 modules scan the same spatiotemporal tensor `x:(B, T, N², d)` from three different perspectives, exchanging information at every layer via residual fusion:

| Chain | Scan Dimension | Mamba3 Config | Physical Meaning |
|-------|---------------|---------------|------------------|
| Spatial | Row + Column (bidirectional) | d_state=64, expand=2, headdim=64 | 2D grid spatial structure |
| Temporal | T (causal) | d_state=64, expand=2, headdim=64 | Temporal causality |
| Causal | K/agent dim (causal) | d_state=32, expand=2, headdim=64 | Inter-agent causal interaction |

**Key design choices:**
- **Unified tensor**: All three chains scan the *same* tensor `x:(B,T,N²,d)`, not independent branches
- **Per-layer residual fusion**: `x = Norm(x + x_s + x_t + c_inject)` at every layer (not just final fusion)
- **Heteromorphic configs**: Each chain has independent `d_state`/`expand`/`headdim` tailored to its scanning dimension
- **No fixed time_embed**: Removed Mamba2-era `nn.Embedding(max_T, d_model)` — fully relies on Mamba3's dt-RoPE for position encoding, enabling natural length extrapolation

### 1.2 PairwiseBasePairMamba3 (BP v2.1)

Inspired by DNA triple helix base pair pairing, we introduced **three pairwise base pair couplings** between the three chains:

| Base Pair | Coupled Chains | Mechanism | Modulation Networks |
|-----------|---------------|-----------|---------------------|
| BP1 | Spatial ↔ Temporal | Cross-Delta modulation + spatial change-rate gating | st_t2s_delta, st_s2t_gate |
| BP2 | Temporal ↔ Causal | Reset gate memory gating + temporal phase injection | tc_c2t_reset, tc_t2c_phase |
| BP3 | Causal ↔ Spatial | FiLM affine (gamma/beta) + spatial reality check | cs_c2s_gamma/beta, cs_s2c_check |

**Critical fix — multiplicative gradient deadlock (v2 → v2.1):**

We discovered a subtle but devastating gradient deadlock in the v2 design. When two zero-initialized networks are multiplied together, their gradients mutually block:

```
v2 (deadlock):   x_t_new = x_t + reset_b * sgate_b * x_t   ← both reset_b and sgate_b start at 0
                                                                     → gradient of reset_b depends on sgate_b=0
                                                                     → gradient of sgate_b depends on reset_b=0
                                                                     → PERMANENT DEADLOCK (C500 measured: both = 0.0000)

v2.1 (fixed):    x_t_new = x_t + reset_b * x_t + sgate_b * x_t   ← additive independent modulation
                                                                     → gradient of reset_b flows through x_t (independent)
                                                                     → gradient of sgate_b flows through x_t (independent)
                                                                     → BOTH LEARN (C500 measured: avg=0.35)
```

**Mathematical proof:**
For `y = a·b·x` where `a(0)=0, b(0)=0`:
- `∂L/∂a = (∂L/∂y) · b · x` → at init, `b=0` → gradient = 0 → `a` never updates → `b` never updates → **deadlock**

For `y = a·x + b·x` where `a(0)=0, b(0)=0`:
- `∂L/∂a = (∂L/∂y) · x` → independent of `b` → gradient flows → `a` updates
- `∂L/∂b = (∂L/∂y) · x` → independent of `a` → gradient flows → `b` updates
- **No deadlock** ✓

## 2. Key Results

### 2.1 Long-Range Extrapolation (GridWorld, 100M → 1.13B parameters)

| Scale | Params | Steps | Seeds | Point Decay (T150) | Window Decay (T150) |
|-------|--------|-------|-------|---------------------|---------------------|
| 100M | 72.32M | 500 | 3 | +0.24% / +1.93% / +0.37% | -0.79% / +0.67% / -0.15% |
| 300M | 182.95M | 2000 | 3 | -1.69% / -1.66% / -0.28% | -0.36% / -0.15% / -0.16% |
| 700M | 724.51M | 2000 | 2 | +0.19% / -0.39% | -0.02% / -0.60% |
| **1.13B** | **1.13B** | 2000 | 3 | +0.28% / -0.12% / +0.38% | -0.01% / -0.39% / -0.30% |

**Main finding**: All converged experiments from 100M to 1.13B show both point decay and window decay within **±2%**. At 5× extrapolation (T=500), decay is only +0.36%. Larger models have *smaller* variance (1B spread=0.5% < 100M 1.7%).

### 2.2 C500 Domestic GPU Experiments (BP v2.1 Validation)

**Hardware**: MetaX C500 GPU (64GB VRAM), Triton 3.0.0+metax, mamba_ssm 2.2.4+metax, torch 2.6.0+metax

**Scale**: 32 scenarios × 3 seeds × 4 models = 384 training runs

| Model | val_ch_acc | ood_ch_acc | ood_decay | Train Time |
|-------|-----------|-----------|-----------|------------|
| Transformer | 30.07% | 30.86% | 2.75% | 10.3s |
| Mamba3 Single-Chain | 29.99% | 30.67% | 2.83% | 28.8s |
| Three-Chain Mamba3 | 56.91% | 57.14% | 0.41% | 60.5s |
| **Three-Chain Mamba3 + BP** | **57.50%** | **57.86%** | 0.60% | 53.7s |

### 2.3 BP v2.1 Deadlock Fix Verification

| Modulation Network | v2 Deadlock Value | v2.1 Fixed Value | Status |
|-------------------|-------------------|------------------|--------|
| bp1_gate | 0.0000 | avg=0.3512 (range: 0.26-0.48) | ✅ Fixed |
| bp2_reset | 0.0000 | avg=0.3500 (range: 0.26-0.47) | ✅ Fixed |
| bp3_gamma | avg=0.3689 | avg=0.3689 | Always normal |

**All 94/96 BP results** (2 missed due to OOM on largest scenario) show non-zero bp1_gate and bp2_reset — the additive fix completely resolves the multiplicative gradient deadlock.

### 2.4 BP Improvement by Scenario Type

| Scenario Type | Three-Chain Mamba3 | Three-Chain + BP | BP Improvement |
|--------------|-------------------|------------------|----------------|
| Random | 56.98% | 57.42% | +0.44% |
| Goal-Directed | 56.50% | 56.85% | +0.35% |
| **Adversarial** | 58.14% | **59.24%** | **+1.10%** |
| **Extreme** | 57.36% | **59.26%** | **+1.90%** |

**Key insight**: BP coupling provides the most benefit in adversarial and extreme scenarios (+1.1% to +1.9%), demonstrating that base pair coupling is most valuable in complex multi-agent interactions.

### 2.5 Real-World Data (Stanford Drone Dataset)

| Dataset | Videos | Model Scale | OOD Decay (window) | Key Finding |
|---------|--------|-------------|---------------------|-------------|
| SDD bookstore | 7 | 30M | +11.72% | No degradation, long-range enhancement |
| SDD nexus | 12 | 30M | -6.9% | Causal chain contributes -55%, learns agent identity (2.3× random) |

## 3. Architecture Details

### 3.1 Mamba3 Integration

We use the official Mamba3 kernel with the following configuration:
- **dt-RoPE**: Cumulative angles `cumsum(angle_raw * dt)` — no preset max_len, naturally supports arbitrary length extrapolation
- **Complex state space**: RoPE-rotated B/C projections create effective complex-valued state structure
- **Trapezoidal discretization**: Replaces Mamba2's ZOH, smaller discretization error at large step sizes

### 3.2 Bidirectional Mamba3

For the spatial chain, we implement bidirectional scanning:
```python
y_fwd = Mamba3(x)           # forward causal scan
y_bwd = Mamba3(flip(x))     # backward scan (shared parameters)
output = y_fwd + flip(y_bwd)  # bidirectional fusion
```

### 3.3 PairwiseBasePairMamba3 Forward Pass

```python
# Three pairwise base pair couplings (v2.1 additive formulation)

# BP1+BP3 → Spatial chain: FiLM affine increment
x_s_new = x_s + gamma * delta * x_s + beta

# BP1+BP2 → Temporal chain: additive independent modulation (v2.1 fix)
x_t_new = x_t + reset_b * x_t + sgate_b * x_t    # ← additive, not multiplicative

# BP2+BP3 → Causal chain: phase injection + spatial check
c_inject_new = c_inject + 0.1 * tphase + 0.1 * scheck * c_inject
```

All modulation networks use **zero initialization** on the last Linear layer, ensuring initial output = baseline (pure residual). Training then learns non-zero modulation.

## 4. Domestic GPU (MetaX C500) Adaptation

### 4.1 Zero-Modification Triton Backend

Mamba3's Triton kernels run on MetaX C500 with **zero code modification** via the Triton-MXMACA backend:
- `source /opt/maca/env.sh` sets MACA_HOME environment variable
- Triton automatically selects the metax backend for kernel compilation
- All Mamba3 forward/backward kernels compile and run correctly

### 4.2 Engineering Safeguards

| Issue | Solution |
|-------|----------|
| `__pycache__` stale cache | Force delete all `__pycache__` directories before deployment |
| `rq_qos_wait` kernel deadlock | `torch.cuda.synchronize()` after every training step |
| Large scenario OOM | `BATCH_SIZE=1` + gradient checkpointing (`use_reentrant=True`) |
| Experiment interruption | Checkpoint resume: result key = `"scenario|seedN|model"`, skip completed, save JSON immediately |

## 5. Comparison with Mamba3 Baselines

| Model | Parameters | OOD changed_acc @ T=150 | OOD Decay | Status |
|-------|-----------|------------------------|-----------|--------|
| Transformer (same params) | 3.43M | 0.0000 (collapse to all-zero) | — | ❌ Collapsed |
| Mamba3 Single-Chain | 3.26M | 0.2999 | 2.83% | ❌ Degrades |
| **ThreeChainMamba3 (ours)** | **168K** | **0.5714** | **0.41%** | ✅ No degradation |
| **ThreeChainMamba3 + BP (ours)** | **168K** | **0.5786** | **0.60%** | ✅ No degradation + BP boost |

## 6. Open Source

- **GitLink**: https://www.gitlink.org.cn/lulululudj/ThreeChainMamba3
- **GitHub**: https://github.com/lululudj/trihelix-mamba
- **License**: MIT
- **Tag**: v2.1.0

### Key Files
- `models/three_chain_mamba3.py` — Three-chain Mamba3 + PairwiseBasePairMamba3 (core architecture)
- `models/mamba3_ref.py` — Pure Python Mamba3 reference implementation (for non-CUDA backends)
- `battle32_c500_gpu.py` — 32-scenario benchmark script with C500 safeguards
- `docs/technical_report.md` — Full technical report
- `docs/performance_report.md` — Performance benchmark report

## 7. Acknowledgments

This work builds upon:
- **Mamba3** (dt-RoPE complex state space + trapezoidal discretization) from [state-spaces/mamba](https://github.com/state-spaces/mamba)
- **Mamba2** (SSD) as the predecessor architecture
- **Stanford Drone Dataset** for real-world multi-agent evaluation

We thank the Mamba3 development team for the excellent kernel design that enabled both long-range extrapolation and seamless domestic GPU deployment.

## 8. Citation

```bibtex
@misc{threechainmamba2026,
  title={ThreeChainMamba3: Three-Chain DNA-Mamba3 with Pairwise Base Pair Coupling for Long-Range Spatiotemporal Extrapolation},
  author={lulululudj},
  year={2026},
  url={https://github.com/lululudj/trihelix-mamba}
}
```
