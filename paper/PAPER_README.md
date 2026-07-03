# Paper — TriHelix-Mamba2

arXiv-style LaTeX source for the TriHelix-Mamba2 paper. Branch `paper-submission`.

## Structure

```
paper/
├── main.tex              # master file (article class, arXiv-compatible)
├── references.bib        # bibliography
├── figures/              # 7 PNGs (copied from ../figures/)
└── sections/
    ├── 01_abstract.tex
    ├── 02_intro.tex
    ├── 03_related.tex
    ├── 04_method.tex
    ├── 05_experiments.tex
    ├── 06_discussion.tex
    ├── 07_conclusion.tex   # Limitations + Conclusion
    └── 08_appendix.tex     # Reproducibility + Hyperparameters + Per-seed metrics
```

## Compile

No LaTeX compiler is on this machine. Two ways to build:

**Option A — Overleaf (easiest):** zip the `paper/` folder, upload as a new
Overleaf project, set compiler to `pdflatex`, set main document to `main.tex`,
recompile.

**Option B — local TeX Live / MiKTeX:**
```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Packages used (all standard in TeX Live full / arXiv): `inputenc, fontenc,
microtype, amsmath, amssymb, amsthm, graphicx, booktabs, xcolor, hyperref,
cleveref, enumitem, geometry, caption`. `\alert` is defined locally in
`main.tex` (it is a Beamer macro, not defined in `article`).

## Scope (per author's instruction)

The paper describes the **A-version `ThreeChainMamba2`** (3.26M, the main
reported model) and its ablations. Per the author's directive, the following
components are **excluded from the paper** because they are not part of the
reported model's forward pass:
- **JEPA** (latent world predictor)
- **CTM** (oscillator)
- **.m3** snapshot manager
- **Eagle** (rollback guard)

This is factually accurate: in `models/three_chain_mamba2.py`, the A-version
`ThreeChainMamba2` class (lines 181–277) instantiates only `HeteroMamba2` +
the linear output head and uses only `balanced_ce_loss`. JEPA / M3SnapshotManager
/ Eagle / Bind appear only in the B-version `ThreeChainMamba2Lite` class
(lines 363+). So the 5-seed OOD results (0.5952±0.0034) were produced by a
model that genuinely does not use the excluded components — there is no
"described model ≠ trained model" gap.

## Verified numbers

All headline numbers were cross-checked against `results_wsl/` JSONs:
- 5-seed `ch_acc@150`: 0.5952 ± 0.0034 (seeds 42/123/456/789/1024) ✓
- 5-seed OOD decay: −0.0042 ✓
- Params: 3,263,440 ✓
- BPv1 (seed 42, `three_chain_mamba2_bp`): OOD 0.5430, decay −9.4% ✓
- BPv2 / Transformer-tiny: documented in README.md / PROFESSIONAL_REPORT.md
  (their raw JSONs were gitignored; numbers are consistent across the
  committed report and commit 470cb9c message).
- Old ThreeChain (Mamba1): OOD 0.5433, 5-seed std = 0.0000 (evaluation bug) ✓

Per-seed breakdown is in `sections/08_appendix.tex` Table 2.

## Figure mapping

| Figure | File | Section |
|---|---|---|
| OOD long-range curve (main) | fig2_ood_curve.png | §5.2 |
| Training dynamics | fig1_training_dynamics.png | §5.2 |
| Collapse diagnosis | fig5_collapse_diagnosis.png | §5.3 |
| 5-seed variance | fig6_seed_variance.png | §5.3 |
| Ablation overview | fig7_ablation.png | §5.4 |
| Parameter efficiency | fig4_param_efficiency.png | §5.4 |
| OOD decay comparison | fig3_ood_decay.png | §5.4 |

## Not yet done

- **Compile not run** (no local LaTeX). The source was manually reviewed:
  all 13 `\cref` targets resolve, all `\label`s defined, `\alert` defined
  locally, unused packages removed, balanced environments. First compile may
  still surface minor warnings (e.g. overfull hboxes) — fix as they appear.
- **Citations**: `lecun2022path`, `gpt`, `mnih2015human`, `jumper2021alphafold`
  are in `references.bib` but uncited (won't appear in output). Remove if
  desired, or cite where appropriate.
