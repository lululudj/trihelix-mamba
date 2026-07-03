# ThreeChainMamba2 长程外推不退化机制分析

> 阶段 3 机制深挖最终报告 (3.1 探针 + 3.2 消融 + 3.3 对照)
> 基于 3 组实验的 5 个假说验证 (H1-H5), 其中 H1/H3/H4 支持, H2/H5 推翻
> 研究诚信: 假说被推翻也诚实记录, 不事后合理化

## 1. 问题定义

**任务**: 离散符号网格世界轨迹预测 — 给定初始网格 S_0 和动作序列 actions, 预测 T 步轨迹 S_1...S_T。

**挑战**: OOD 长程外推 — 训练 T=100, 评估 T=150 (1.5× 外推)。模型不得在 OOD 区退化。

**核心发现**: ThreeChainMamba2 (SSM) 在 30M→1.13B 参数范围内长程外推不退化 (decay ±2%), 而同参数 Transformer 完全退化 (changed_acc≈0.055, 接近随机)。

**本文问题**: 为什么 SSM 不退化而 Transformer 退化? 机制差异何在?

## 2. 架构概述

ThreeChainMamba2 由三部分组成:

1. **AnchorInit2** (初始表示锚定): `x = cell_embed(S_0) + act_mean + time_emb`
   - cell_embed: 空间网格编码 (B, N², d)
   - act_mean: 动作序列均值编码 (B, T, d), 广播到 N²
   - time_emb: 时间位置编码 (B, T, d)
   - 三源加法融合, 提供 O(1) 复杂度的初始锚定

2. **三链并行 SSM** (HeteroMamba2): 每层包含三条 Mamba2 链
   - 空间链 (spatial): N² 维双向扫描, 建模网格内空间结构
   - 时间链 (temporal): T 维因果 SSM, 建模时间演化
   - 因果链 (causal): K 维 agent 因果序, 建模 agent 间交互
   - 残差融合: `x = norm_fuse(x + x_s + x_t + c_inject)`

3. **Fusion + 预测头**: CrossAttention 融合 + MLP 输出 cell 类型 logits

## 3. 实验证据

### 3.1 三链 Hidden State 探针 (H1/H2/H3)

**方法**: 加载 30M checkpoint, 在 OOD T=150 上抽取三链 (h_s/h_t/h_c) 全时序状态, 测量 Frobenius norm 随 t 演化。

**假说与结果**:
- H1: 时间链 h_t 在 OOD 区稳定 → **✅ 支持** (norm 变化 <15%)
- H2: 空间链 h_s 随 t 衰减 → **❌ 推翻** (空间链未显著衰减)
- H3: 因果链 h_c 极其稳定 → **✅ 支持** (norm 变化 <15%)

**结论**: 三链 norm 在 OOD 区都稳定 → 不退化来自**架构整体稳定性**, 而非单链独撑。

### 3.2 三链消融实验 (置零输出, 保持参数量)

**方法**: 在残差融合前置零某链输出 (`x_s = zeros`), 保持参数量 26.59M 不变, 训练 500 步 + OOD eval。

**结果**:

| 模型 | decay% | changed_acc@150 | 判定 |
|------|--------|-----------------|------|
| Baseline (三链全开) | -0.35% | 0.5921 | - |
| 消融空间链 | -7.54% | 0.5491 | 空间链有实质贡献 |
| 消融时间链 | -0.38% | 0.5924 | 时间链非主因 |
| 消融因果链 | -1.07% | 0.5876 | 因果链非主因 |
| **三链全消融** | -0.66% | 0.5900 | **不退化** |

**关键反常现象**: 单消融空间链恶化 (-7.54%) > 全消融 (-0.66%)。

**机制解读**:
- 单消融空间链: 时间链/因果链基于**残缺的 x** (缺少空间信息) 继续演化, 错误信息被放大 → 恶化
- 全消融: `x = norm_fuse(x + 0 + 0 + 0) = norm_fuse(x)`, 只剩 AnchorInit2 初始表示 → 仍稳定
- 空间链的作用是**维持 x 的正确性**供其他链演化, 而非直接承载长程信息

### 3.3 SSM vs Transformer 信号保留对照 (H4/H5)

**方法**: 对 30M SSM 和 30M Transformer (30.74M) 各注入扰动 (修改 S_0 的 3 个非零 cell), 测量敏感度 retention(t) = sensitivity(t) / sensitivity(1)。

**假说与结果**:

| 指标 | SSM (26.59M) | Transformer (30.74M) |
|------|-------------|---------------------|
| 信号保留率 @t=150 | 94.1% | **99.4%** |
| OOD decay | +0.2% | -6.6% |
| changed_acc @t=150 | **0.5929** | 0.0555 |
| val changed_acc @500步 | 0.5740 | 0.0715 |

- H4: SSM 保留率 > 70% → **✅ 支持** (94.1%)
- H5: Transformer 保留率 < 30% → **❌ 推翻** (99.4%, 甚至高于 SSM)

**反直觉发现**: Transformer **完美保留输入信号 (99.4%)** 但**预测完全崩溃 (changed_acc=0.0555)**。

原假说"Transformer 退化因为注意力稀释信号"被推翻。Transformer 记住了输入, 但无法将其转化为有效预测。

## 4. 机制结论: 三层保障

综合 3.1-3.3 的实验证据, ThreeChainMamba2 长程外推不退化的机制来自**三层保障**:

### 第一层: AnchorInit2 结构性锚定 (必需)

- **机制**: `x = cell_embed(S_0) + act_mean + time_emb` 将空间网格、动作序列、时间位置三源信息加法融合, 形成 O(1) 复杂度的初始锚定
- **证据**:
  - 3.2 三链全消融 (仅剩 AnchorInit2+残差) 仍不退化 (decay=-0.66%, changed_acc=0.5900)
  - 3.3 Transformer (无 AnchorInit2, 仅 cell_embed+自注意力) 完全退化 (changed_acc=0.0555), 即使保留 99.4% 输入信号
- **含义**: 不退化的**根因**是 AnchorInit2 的结构性锚定, 而非 SSM 的递推状态保持

### 第二层: 残差连接 + LayerNorm (重要)

- **机制**: `x = norm_fuse(x + x_s + x_t + c_inject)` 的残差结构确保即使三链输出为零, 初始表示 x 仍能传递到输出
- **证据**: 3.2 no_all 消融后 `x = norm_fuse(x)` (三链全零), changed_acc@150=0.5900 (仅降 0.4% vs baseline)
- **含义**: 残差连接是 AnchorInit2 信号传递到输出的**管道**, LayerNorm 确保数值稳定

### 第三层: 三链 SSM 演化 (锦上添花)

- **机制**: 三条并行 Mamba2 链 (空间/时间/因果) 在初始表示基础上进一步演化, 提升预测精度
- **证据**:
  - 3.1 三链 norm 在 OOD 区都稳定 (H1/H3 支持)
  - 3.2 单消融空间链恶化 (-7.54%), 说明空间链维持信息正确性供其他链演化
  - 时间链/因果链消融无影响 (非主因)
- **含义**: 三链演化是**优化层**, 提升精度但非长程外推的必需; 空间链对维持信息正确性最关键

## 5. 与 Transformer 的机制差异

| 维度 | ThreeChainMamba2 | Transformer |
|------|-----------------|-------------|
| 初始锚定 | AnchorInit2 (cell+action+time 三源加法) | cell_embed + 自注意力 |
| 信号保留 | 94.1% @t=150 | 99.4% @t=150 |
| 预测质量 | changed_acc=0.5929 | changed_acc=0.0555 |
| 退化? | 否 (decay +0.2%) | 是 (decay -6.6%) |
| 根因 | AnchorInit2 锚定 → 有效映射 | 无锚定 → 无法学习映射 |

**核心差异**: 不是信息保留能力 (两者都高), 而是**初始表示的结构性锚定**。AnchorInit2 提供了空间-动作-时间三源对齐的初始信号, 使模型即使 OOD 也能基于锚定预测。Transformer 仅靠 cell_embed + 自注意力, 缺乏这种结构性先验, 导致训练 500 步后 val changed_acc 仍仅 0.0715 — **根本没学会任务**, 而非 OOD 特异性退化。

## 6. 与 Mamba2 SSD 框架的理论联系

ThreeChainMamba2 使用 Mamba2 的 SSD (State Space Duality) 框架, 其核心是衰减矩阵 A:

- Mamba2 的 A 矩阵通过 `heavy_tail_activation` (从 Mamba3 移植) 提供长尾衰减特性
- 但 3.2/3.3 实验表明, SSM 的递推状态保持**不是**不退化的主因 (Transformer 同样保留信号)
- SSM 的优势在于**计算效率** (线性复杂度 O(T) vs Transformer 的 O(T²)) 和**三链分解的归纳偏置**, 而非信息保持能力本身

## 7. 结论

ThreeChainMamba2 长程外推不退化的机制是**三层保障**:

1. **AnchorInit2 结构性锚定** (必需) — 根因, 提供空间-动作-时间三源对齐的初始表示
2. **残差 + LayerNorm** (重要) — 信号传递管道, 确保初始锚定到达输出
3. **三链 SSM 演化** (锦上添花) — 精度优化, 空间链维持信息正确性

这一结论基于 5 个假说的严格验证 (3 支持 / 2 推翻), 其中 H5 的推翻揭示了"信号保留 ≠ 预测质量"的关键区分, 修正了"SSM 优势在于状态保持"的直觉认知。

---

## 附录: 假说验证汇总

| 假说 | 内容 | 验证方法 | 结论 |
|------|------|---------|------|
| H1 | 时间链 h_t 在 OOD 区稳定 | 3.1 norm 探针 | ✅ 支持 |
| H2 | 空间链 h_s 随 t 衰减 | 3.1 norm 探针 | ❌ 推翻 |
| H3 | 因果链 h_c 极其稳定 | 3.1 norm 探针 | ✅ 支持 |
| H4 | SSM 信号保留率 > 70% | 3.3 扰动探针 | ✅ 支持 (94.1%) |
| H5 | Transformer 信号保留率 < 30% | 3.3 扰动探针 | ❌ 推翻 (99.4%) |

## 产出物

- 3.1: `paper/figures/stage3_hidden_probes_*.png` + `results_stage3/probe_*.npz`
- 3.2: `paper/figures/stage3_ablation.png` + `paper/stage3_ablation.md`
- 3.3: `paper/figures/stage3_3_retention.png` + `paper/stage3_3_retention.md`
- 本报告: `paper/mechanism_analysis.md`
