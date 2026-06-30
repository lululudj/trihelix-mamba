# TriHelix-Mamba 🧬

### Three-Chain DNA-Mamba2 for Long-Range State Extrapolation

> **A unified-tensor SSM architecture that cures OOD degradation — 3 chains (time / space / causality) scanning a single tensor, with zero long-range decay.**

---

## 🎯 The Problem

State-space models (Mamba/SSD) degrade on **out-of-distribution (OOD) long sequences** — when test length exceeds training length, predictions collapse to all-zeros. This is the SSM equivalent of "forgetting how to predict" the longer you roll out.

| Model | Params | OOD changed_acc @ T=150 | OOD decay |
|---|---|---|---|
| Old ThreeChain (Mamba1) | 4.77M | 0.5433 | **std=0.0000 (bug illusion)** |
| **TriHelix-Mamba2 (ours)** | **3.26M** | **0.5952 ± 0.0034** | **−0.0042 (≈ zero)** |

- Trained on T=100 steps, tested on T=150 steps (50% longer, never seen)
- 5-seed benchmark; OOD decay ≈ 0 means **no degradation on unseen lengths**
- Fewer parameters, better OOD extrapolation

---

## 🧬 Architecture: Three Helical Chains on One Tensor

```
                  Unified Spatiotemporal Tensor  x : (B, T, N², d)
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
     ┌──────────┐            ┌──────────┐            ┌──────────┐
     │ TIME     │            │ SPACE    │            │ CAUSAL   │
     │ chain    │            │ chain    │            │ chain    │
     │ (causal  │            │ (bi-dir  │            │ (K-dim   │
     │  Mamba2) │            │  Mamba2) │            │  scan)   │
     └────┬─────┘            └────┬─────┘            └────┬─────┘
          │                       │                       │
          └───────────┬───────────┴───────────┬───────────┘
                      ▼   residual fusion     ▼
                   ┌─────────────────────────────┐
                   │   Layer × N (per-block)     │
                   └─────────────────────────────┘
                      │
                      ▼
                   cell logits
```

**Key design choices (lessons from 6 failure modes):**

1. **Unified tensor** — all 3 chains scan the same `x:(B,T,N²,d)`, not separate branches
2. **Residual fusion per layer** — chains exchange info every layer (not just at the end)
3. **Heterogeneous Mamba2 configs** — each chain has tuned `d_state`/`expand`/`headdim`
4. **No BasePair / No Bind / No Fusion-attention** — replaced by simple residual sum (Occam's razor)

---

## 📊 Key Results (5-seed, 2000 steps)

| Metric | TriHelix-Mamba2 | Old ThreeChain |
|---|---|---|
| Parameters | **3.26M** | 4.77M |
| Train changed_acc @100 | 0.5991 | 0.5991 |
| **OOD changed_acc @150** | **0.5952 ± 0.0034** | 0.5433 |
| **OOD decay (100→150)** | **−0.42% (≈0)** | — |
| Zero-ratio (degradation check) | 17.2% ✅ | 97.7% ❌ |

**Full report:** see [PROFESSIONAL_REPORT.md](PROFESSIONAL_REPORT.md) (9 sections + 6 figures)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies (needs CUDA for Mamba2)
pip install torch numpy pyyaml matplotlib
pip install triton mamba_ssm causal_conv1d

# 2. Generate GridWorld data
python data/gen_grid_world.py

# 3. Train
python train.py --config configs/default.yaml --model three_chain_mamba2 --seed 42

# 4. Degradation diagnosis (non-degeneration gate)
python probe_mamba2_brain.py --model three_chain_mamba2 --seed 42 --steps 2000

# 5. OOD evaluation (train T=100, test T=150)
python eval_ood.py --model three_chain_mamba2 --seed 42
```

---

## 📁 Repository Structure

```
trihelix-mamba/
├── models/                          # All model implementations
│   ├── three_chain_mamba2.py        # ★ Main: TriHelix-Mamba2 (3.26M)
│   ├── three_chain.py               # Old baseline (4.77M, degraded)
│   ├── three_chain_mamba2_bp.py     # Ablation: base-pair variant
│   ├── baselines.py                 # SingleChain, ConcatMamba, Transformer, GNN
│   └── common.py                    # Shared components
├── configs/                         # YAML configs (matched/scaled)
├── data/                            # Data generation (GridWorld)
├── train.py                         # Training loop
├── eval.py                          # In-distribution evaluation
├── eval_ood.py                      # OOD long-range evaluation
├── probe_mamba2_brain.py            # Degradation diagnosis tool
├── utils.py                         # Shared utilities
├── PROFESSIONAL_REPORT.md           # ★ Full results report (9 sections)
├── figures/                         # 6 cosmic-dark-style charts
├── MAMBA2_STATUS.md                 # Development status
└── requirements.txt
```

---

## 🔬 Reproducibility

All experiments use 5 seeds: `[42, 123, 456, 789, 1024]`.

**Non-degeneration gate** (a model must pass before benchmarking):
- `zero_ratio < 0.90` (not predicting all-zeros)
- `changed_acc > 0.30` (better than random 1/16 = 0.0625)
- OOD non-zero predictions > 5% of target

---

## 📝 Citation

```bibtex
@misc{trihelixmamba2026,
  title={TriHelix-Mamba: Three-Chain DNA-Mamba2 for Long-Range State Extrapolation},
  author={lululudj},
  year={2026},
  url={https://github.com/lululudj/trihelix-mamba}
}
```

Paper (arXiv preprint + NeurIPS Workshop submission) coming soon.

---

## 🗺️ Roadmap

- [x] **Stage 1** — Core model + benchmark + paper (this repo)
- [ ] **Stage 2** — Edge device specialization (ONNX export, CPU inference)
- [ ] **Stage 3** — Wargame prediction demo
- [ ] **Stage 4** — World model demo (video/robot state prediction)

---

## License

MIT — see [LICENSE](LICENSE).
