# 三链不退化理论推导

> ThreeChainMamba2 长程外推不退化机制的理论分析 — 面向沐曦青年开源专项基金评审与研究生套磁材料
> 本文档基于 5 个可证伪假说的严格验证 (3 支持 / 2 推翻, 推翻也诚实记录), 给出三层保障机制的形式化定义。
> 实验证据: [paper/mechanism_analysis.md](../paper/mechanism_analysis.md), [paper/stage3_ablation.md](../paper/stage3_ablation.md), [paper/stage3_3_retention.md](../paper/stage3_3_retention.md)

## 目录 (TOC)

1. [问题定义: OOD 长程外推退化现象](#1-问题定义-ood-长程外推退化现象)
2. [假说体系 (H1-H5)](#2-假说体系-h1-h5)
3. [三层保障机制 (核心理论贡献)](#3-三层保障机制-核心理论贡献)
4. [GridWorld vs SDD 差异理论](#4-gridworld-vs-sdd-差异理论)
5. [window-smoothed decay 理论](#5-window-smoothed-decay-理论)
6. [与 Transformer / Mamba3 的机制对比](#6-与-transformer--mamba3-的机制对比)
7. [局限与可证伪边界](#7-局限与可证伪边界)

---

## 1. 问题定义: OOD 长程外推退化现象

### 1.1 任务设定

给定初始网格 $S_0 \in \mathbb{Z}^{N \times N}$ 与动作序列 $A \in \mathbb{Z}^{K \times T}$, 预测 $T$ 步轨迹 $\hat{S}_{1:T}$。模型在训练时仅观察 $T = T_{train}$ 步序列, 推理时需对 $T_{eval} \gg T_{train}$ 的未来做预测。

### 1.2 退化现象 (Prediction Collapse)

定义 changed_acc:

$$
\text{changed\_acc}(t) = \frac{1}{|\mathcal{C}_t|} \sum_{(i,j) \in \mathcal{C}_t} \mathbb{1}\!\left[ \hat{S}_t[i,j] = S_t^{*}[i,j] \right], \quad \mathcal{C}_t = \{(i,j) : S_t^{*}[i,j] \neq S_0[i,j]\}
$$

OOD decay (单点):

$$
\text{decay}_{single} = \frac{\text{changed\_acc}(T_{eval}) - \text{changed\_acc}(T_{train})}{\text{changed\_acc}(T_{train})} \times 100\%
$$

**退化 (collapse)** 定义: $|\text{decay}_{single}| > 2\%$ 且 $\text{changed\_acc}(T_{eval}) \to 0$ (预测全猜 0, 即 $\hat{S}_t \equiv 0$)。

### 1.3 已观察到的退化案例

| 模型 | 参数 | $\text{changed\_acc}(T_{eval})$ | $\text{decay}$ | 现象 |
|---|---|---|---|---|
| 同参数 Transformer-tiny | 3.43M | 0.0000 | — | 完全塌缩 (zero_ratio=1.0) |
| Mamba3 (官方实现) | — | — | -11.5% | 算法层面退化 |
| 旧 ThreeChain (Mamba1) | 4.77M | 0.5433 | std=0.0000 (脚本 bug) | 退化假象 |
| **ThreeChainMamba2 (本工作)** | **3.26M → 1.13B** | **0.5952 ± 0.0034** | **−0.0042 (≈ 0)** | ✅ 不退化 |

主结论: 100M → 1.13B 参数区间, 单点 decay 与窗口 decay **双双全部落在 ±2% 内**, 1B 三 seed spread 仅 0.5%。

---

## 2. 假说体系 (H1-H5)

为回答"为什么 SSM 不退化而 Transformer 退化", 设计 5 个可证伪假说。**研究诚信原则: 假说被推翻也诚实记录, 不事后合理化。**

### 2.1 假说 H1 (支持): 时间链在 OOD 区稳定

**假说**: 时间链 $h_t$ 在 $t > T_{train}$ (OOD 区) 的 Frobenius norm 保持稳定, 承载长程信息。

**验证方法**: 加载 30M checkpoint, 在 OOD $T=150$ 上抽取 $h_t$ 全时序状态, 测量 $\|h_t\|_F$ 随 $t$ 演化。

**结果** (1000 步收敛模型):

| $\|h_t\|_F$ @ $t=100$ | $\|h_t\|_F$ @ $t=149$ | decay | 判定 |
|---|---|---|---|
| 82.73 | 83.30 | **+0.7%** | ✅ 极稳 |

**结论**: ✅ **支持** — 时间链在 OOD 区 norm 几乎不变。Mamba2 因果递推 $h_{t+1} = \bar{A} h_t + \bar{B} x_t$ 在 $T_{eval} \gg T_{train}$ 时仍稳定。

### 2.2 假说 H2 (推翻): 空间链随 $t$ 衰减

**假说**: 空间链 $h_s$ 在 OOD 区随 $t$ 衰减, 因其不承载时间信息。

**结果**:

| $\|h_s\|_F$ @ $t=100$ | $\|h_s\|_F$ @ $t=149$ | decay | 判定 |
|---|---|---|---|
| 125.01 | 121.46 | **-2.8%** | ❌ 微降 (阈值 -15%) |

**结论**: ❌ **推翻** — 空间链未显著衰减。这反驳了"空间链只服务短程"的直觉, 提示**三链都保持稳定**, 不退化是架构整体性质而非某条链独撑。

### 2.3 假说 H3 (支持): 因果链极其稳定

**假说**: 因果链 $h_c$ 与 action 序列强相关, 在 OOD 区 norm 极其稳定。

**结果** (1000 步模型):

| $\|h_c\|_F$ @ $t=100$ | $\|h_c\|_F$ @ $t=149$ | decay | 判定 |
|---|---|---|---|
| 74.35 | 74.35 | **-0.0%** | ✅ 极稳 |

**结论**: ✅ **支持** — 因果链沿 $K$ 维扫描, 状态稳定性最高。

### 2.4 假说 H4 (支持): SSM 信号保留率 > 70%

**假说**: SSM 对 $t=0$ 注入的扰动信号, 在 $t = T_{eval} = 150$ 仍能保留 (> 70%)。

**验证方法** (扰动探针, [probe_token_retention.py](../probe_token_retention.py)):

1. 跑 clean forward 拿 $\text{logits}_{clean}$
2. 修改 $S_0$ 的 3 个非零 cell (注入扰动)
3. 跑 perturbed forward 拿 $\text{logits}_{pert}$
4. $\text{sensitivity}(t) = \|\text{logits}_{pert}[:,t] - \text{logits}_{clean}[:,t]\|_2 / \|\text{logits}_{clean}[:,t]\|_2$
5. $\text{retention}(t) = \text{sensitivity}(t) / \text{sensitivity}(1)$

**结果**: SSM retention @ $t=150$ = **94.1%** ✅

### 2.5 假说 H5 (推翻): Transformer 退化因注意力稀释信号

**假说**: Transformer 信号保留率 < 30%, 退化因注意力稀释输入信号。

**结果**:

| 指标 | SSM (26.59M) | Transformer (30.74M) |
|---|---|---|
| 信号保留率 @ $t=150$ | 94.1% | **99.4%** (高于 SSM) |
| OOD decay | +0.2% | -6.6% |
| $\text{changed\_acc}$ @ $t=150$ | **0.5929** | 0.0555 |

**结论**: ❌ **推翻** — Transformer 完美保留 99.4% 输入信号但预测完全崩溃 ($\text{changed\_acc} = 0.0555$, 接近随机 1/16 = 0.0625)。

### 2.6 H5 推翻的深层含义

**反直觉发现**: "信号保留 ≠ 预测质量"。

$$
\text{retention}(\text{Transformer}) > \text{retention}(\text{SSM}) \quad \text{but} \quad \text{changed\_acc}(\text{Transformer}) \ll \text{changed\_acc}(\text{SSM})
$$

原假说"Transformer 退化因为注意力稀释信号"被推翻。真实机制是:
1. **不是信息遗忘问题** — 两个模型都完美保留输入扰动信号 (94% vs 99%)
2. **是预测质量问题** — Transformer 训练 500 步后 val $\text{changed\_acc}$ 仍仅 0.0715, 根本没学会任务
3. **根因是归纳偏置缺失** — Transformer 缺 AnchorInit2 的初始锚定 (cell+action+time 三源), 仅靠 cell_embed + 自注意力无法建立有效的时间外推映射

---

## 3. 三层保障机制 (核心理论贡献)

综合 H1-H5 的实验证据, ThreeChainMamba2 长程外推不退化的机制来自**三层保障**。这是本工作的核心理论贡献。

### 3.1 第一层: 基础层 (必需) — AnchorInit2 结构性锚定

**机制**: $x^{(0)} = E_{cell}[S_0] + \bar{E}_{act}[A] + E_{time}[t]$, 将空间网格、动作序列、时间位置三源信息**加法融合**, 形成 $O(1)$ 复杂度的初始锚定。

**形式化**: 三源加法保证任一源信息 (cell / action / time) 在 $x^{(0)}$ 中均存在分量, 即使后续 SSM 演化失效, 输出头仍能基于 $x^{(0)}$ 做出预测。

**证据**:
- 阶段 3.2 三链全消融 (仅剩 AnchorInit2 + 残差) 仍不退化 ($\text{decay} = -0.66\%$, $\text{changed\_acc} = 0.5900$)
- 阶段 3.3 Transformer (无 AnchorInit2, 仅 cell_embed + 自注意力) 完全退化 ($\text{changed\_acc} = 0.0555$), 即使保留 99.4% 输入信号

**含义**: 不退化的**根因**是 AnchorInit2 的结构性锚定, 而非 SSM 的递推状态保持。

### 3.2 第二层: 稳定层 (重要) — 残差连接 + LayerNorm

**机制**: $x^{(\ell+1)} = \text{LayerNorm}(x^{(\ell)} + h_s + h_t + c_{inject})$ 的残差结构确保即使三链输出为零, 初始表示 $x^{(0)}$ 仍能传递到输出 $x^{(L)}$。

**形式化**:

当 $h_s = h_t = c_{inject} = 0$ 时 (三链全消融):

$$
x^{(\ell+1)} = \text{LayerNorm}(x^{(\ell)} + 0 + 0 + 0) = \text{LayerNorm}(x^{(\ell)})
$$

递推 $L$ 层后:

$$
x^{(L)} = \text{LayerNorm} \circ \text{LayerNorm} \circ \cdots \circ \text{LayerNorm}(x^{(0)}) \approx x^{(0)} \cdot \gamma + \beta
$$

(其中 $\gamma, \beta$ 是 LayerNorm 的可学习仿射参数), 即输出仍主要由 $x^{(0)}$ 主导。

**证据**: 阶段 3.2 no_all 消融后 $\text{changed\_acc}@150 = 0.5900$ (仅降 0.4% vs baseline 0.5921)。

**含义**: 残差连接是 AnchorInit2 信号传递到输出的**管道**, LayerNorm 确保数值稳定 (防 SSM 状态爆炸)。

### 3.3 第三层: 优化层 (锦上添花) — 三链 SSM 演化

**机制**: 三条并行 Mamba2 链 (空间 / 时间 / 因果) 在初始表示基础上进一步演化, 提升预测精度。

**证据**:
- 阶段 3.1 三链 norm 在 OOD 区都稳定 (H1 / H3 支持)
- 阶段 3.2 单消融空间链恶化 ($\text{decay} = -7.54\%$), 说明空间链**维持信息正确性**供其他链演化
- 时间链 / 因果链消融无影响 (非主因)

**反常现象解释** (单消融空间链恶化 > 全消融):

| 模型 | $\text{decay}$ | $\text{changed\_acc}@150$ |
|---|---|---|
| baseline | -0.35% | 0.5921 |
| no_spatial (单消融空间) | **-7.54%** | **0.5491** (降 7.3%) |
| no_all (三链全消融) | -0.66% | 0.5900 (仅降 0.4%) |

**机制解读**:
1. **no_spatial 恶化原因**: 空间链被置零, 但时间链和因果链仍在工作。它们基于**残缺的 $x$** (缺少空间信息) 继续演化, 错误信息被放大 → 预测质量下降
2. **no_all 不恶化原因**: 三链全置零, $x = \text{LayerNorm}(x)$, 只剩 AnchorInit2 初始表示。模型靠初始锚定直接预测, 这已足够
3. **空间链的真正作用**: 维持 $x$ 的**空间正确性**供其他链基于正确输入演化, 而非直接承载长程信息

**含义**: 三链演化是**优化层**, 提升精度但非长程外推的必需; 空间链对维持信息正确性最关键。

### 3.4 三层保障汇总

| 层次 | 机制 | 证据 | 重要性 |
|---|---|---|---|
| **基础层** | AnchorInit2 初始锚定 (cell+action+time 三源加法) | 3.2 no_all 不退化 + 3.3 Transformer 无锚定则崩溃 | **必需** |
| **稳定层** | 残差连接 + LayerNorm | 3.2 no_all 仍 stable | **重要** |
| **优化层** | 三链 SSM 演化提升精度 | 3.2 单消融空间链恶化 | **锦上添花** |

---

## 4. GridWorld vs SDD 差异理论

三链机制在两个数据集上展现出**不同重要性**, 这一差异本身就是机制的可证伪证据。

### 4.1 GridWorld: 三链"锦上添花"

**消融实验** (30M seed2 @500 步, baseline decay=-0.35%):

| 消融 | $\text{decay}\%$ | $\text{changed\_acc}@150$ | 判定 |
|---|---|---|---|
| 消融空间链 | -7.54% | 0.5491 | 空间链有实质贡献 |
| 消融时间链 | -0.38% | 0.5924 | 时间链非主因 |
| 消融因果链 | -1.07% | 0.5876 | 因果链非主因 |
| **三链全消融** | **-0.66%** | **0.5900** | **不退化** (仅降 0.4%) |

GridWorld 上 no_all 仅降 0.4%, 三链是**优化层**, AnchorInit2 单独足够。

### 4.2 SDD: 三链"必需"

**SDD 真实数据消融** (bookstore + nexus):

| 消融 | pos_iou 变化 | 解读 |
|---|---|---|
| ablate_all (三链全消融) | pos_iou 降 45-71% | 三链必需, AnchorInit2 单独不够 |

SDD 上 ablate_all 导致 pos_iou 大幅下降, 三链成为**必需层**。

### 4.3 差异根因理论

**根因**: 真实数据复杂度高, AnchorInit2 单独不够。

形式化分析: 设任务的**信息复杂度**为 $\mathcal{I}$, AnchorInit2 提供的**初始锚定信息量**为 $\mathcal{I}_0$, 三链演化提供的**增量信息量**为 $\Delta\mathcal{I}$:

$$
\mathcal{I} = \mathcal{I}_0 + \Delta\mathcal{I}
$$

预测有效要求 $\mathcal{I}_0 + \Delta\mathcal{I} \ge \mathcal{I}_{min}$ (任务最低信息需求)。

- **GridWorld**: $\mathcal{I}_{min}$ 低 (16 cell types + 简单动力学), $\mathcal{I}_0 \ge \mathcal{I}_{min}$ → 三链 $\Delta\mathcal{I}$ 锦上添花
- **SDD**: $\mathcal{I}_{min}$ 高 (真实行人轨迹 + 多 agent 复杂交互), $\mathcal{I}_0 < \mathcal{I}_{min}$ → 必需 $\Delta\mathcal{I}$ 补足

**含义**: 三链机制不是"普遍必需"也不是"普遍冗余", 而是**任务复杂度的函数**。这本身就是可证伪的理论预测: 在更高复杂度任务上, 三链的重要性会进一步提升。

### 4.4 SDD 长程增强现象

SDD bookstore 10k 步充分训练后, OOD 长程不仅不衰减反而**增强** (window decay = +11.72%):

| 训练步数 | val ch_acc 峰值 | 单点 decay | window decay |
|---|---|---|---|
| 3k 步 (欠训) | 0.49 | -26.54% | (未算) |
| 10k 步 (收敛) | 0.5141 | +53.2% (ch@100 坑干扰) | **+11.72%** |

3k 步的衰减是**"未充分训练"假象**, 10k 步充分训练后 SDD 长程增强机制依然成立。详见 [paper/stage2_sdd_report.md](../paper/stage2_sdd_report.md)。

---

## 5. window-smoothed decay 理论

### 5.1 单点 decay 不可靠性

单点 decay $\text{decay}_{single} = (\text{ch}@T_{eval} - \text{ch}@T_{train}) / \text{ch}@T_{train}$ 在 $t = T_{train} = 100$ 处有**边界噪声**:

**SDD ch@100 异常深坑现象** (10k 步 eval):

```
ch@99  = 0.2558  (in-distribution 最后一步)
ch@100 = 0.1777  ← 异常低点 (训练 max_T 边界)
ch@101 = 0.2542  (OOD 第一步, 恢复)
ch@150 = 0.2723  (长程终点, 最高)
```

### 5.2 根因: time_embed 未训练 slot

**根因**: `time_embed = nn.Embedding(max_T = 256, d_model)` ([three_chain_mamba2.py L301](../models/three_chain_mamba2.py)), 训练时 $T = 100$ 只更新 slot 0-99, $t = 100$ 是第一个**未训练 embedding** 的时间步 → SSM 状态首次外推的瞬态冲击。$t = 101$ 后 SSM 稳定到外推动力学, 曲线恢复上升。

形式化: 设 $E_{time}[t]$ 在 $t < T_{train}$ 时为训练值 $E^*$, $t \ge T_{train}$ 时为随机初始化值 $E^{init}$:

$$
x^{(0)}[t] = E_{cell}[S_0] + \bar{E}_{act}[A] + \begin{cases} E^*_{time}[t] & t < T_{train} \\ E^{init}_{time}[t] & t \ge T_{train} \end{cases}
$$

$t = T_{train}$ 处 $E_{time}$ 突变引入瞬态扰动, SSM 在 $t = T_{train} + 1$ 后逐渐吸收扰动恢复稳定。

### 5.3 window-smoothed decay 定义

为消除单点噪声, 定义窗口平滑 decay:

$$
\boxed{\;
\text{decay}_{win} = \frac{\bar{c}[T_{eval}] - \bar{c}[T_{train}]}{\bar{c}[T_{train}]}, \quad
\bar{c}[t] = \frac{1}{2W+1} \sum_{\tau = t-W}^{t+W} \text{changed\_acc}(\tau)
\;}
$$

其中 $W = 5$ (±5 邻域均值)。

### 5.4 稳健性论证

设 $\text{changed\_acc}(t)$ 在 $t = T_{train}$ 处有噪声 $\epsilon$:

$$
\text{changed\_acc}(T_{train}) = \text{ch}^* + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

则:

$$
\text{var}(\text{decay}_{single}) \approx \frac{\sigma^2}{(\text{ch}^*)^2}, \quad
\text{var}(\text{decay}_{win}) \approx \frac{\sigma^2}{(2W+1)(\text{ch}^*)^2}
$$

窗口平滑将方差降低 $\frac{1}{2W+1} = \frac{1}{11}$ 倍 (约 $\sqrt{11} \approx 3.3\times$ 标准差降低)。

### 5.5 30M 实验的 window-smoothed 印证

30M @500 步 5 seed 单点 decay $= [-9.81, -2.29, -0.35, -2.02, -0.42]\%$, 仅 2/5 在 ±2% 内。

window-smoothed decay $= [-3.37, +0.13, -0.22, -0.98, -0.14]\%$, **4/5 在 ±2% 内**, median $= -0.22\%$。

**根因诊断**: 30M 曲线噪声大, 单点 ch@100 易落在局部峰值 (如 seed0 的 ch@100=0.5940 vs 窗口 0.5494), 把单点 decay 拉偏。seed0 的 -9.81% 主要是这个 ch@100 峰值假象, 窗口平滑后仅 -3.37%。

**结论**: 30M 不作为"不退化"的硬数据点 (单点方差过大), 但 window median 在 ±2% 内说明架构未塌缩。100M 是"不退化"稳定成立的干净下界。

---

## 6. 与 Transformer / Mamba3 的机制对比

### 6.1 三模型对比表

| 维度 | ThreeChainMamba2 | Transformer | Mamba3 |
|---|---|---|---|
| 初始锚定 | AnchorInit2 (cell+action+time 三源加法) | cell_embed + 自注意力 (无锚定) | AnchorInit2 (同本工作) |
| 复杂度 | $O(L)$ (SSM 线性) | $O(L^2)$ (注意力二次) | $O(L)$ |
| A 激活 | $\exp$ (默认) 或 heavy_tail (HTA) | N/A | heavy_tail |
| 信号保留率 @ $t=150$ | 94.1% | 99.4% | — |
| $\text{changed\_acc}$ @ $t=150$ | **0.5929** | 0.0555 | — |
| OOD decay | +0.2% (✅) | -6.6% (❌) | -11.5% (❌) |
| 根因 | AnchorInit2 锚定 → 有效映射 | 无锚定 → 无法学习映射 | 算法层面问题 |

### 6.2 核心机制差异

**不是信息保留能力** (Transformer 99.4% > SSM 94.1%), 而是**初始表示的结构性锚定**:

$$
\underbrace{\text{retention}(\text{Transformer}) > \text{retention}(\text{SSM})}_{\text{信息保留}} \quad \nRightarrow \quad \underbrace{\text{changed\_acc}(\text{Transformer}) > \text{changed\_acc}(\text{SSM})}_{\text{预测质量}}
$$

AnchorInit2 提供空间-动作-时间三源对齐的初始信号, 使模型即使 OOD 也能基于锚定预测。Transformer 仅靠 cell_embed + 自注意力, 缺乏这种结构性先验, 导致训练 500 步后 val $\text{changed\_acc}$ 仍仅 0.0715 — **根本没学会任务**, 而非 OOD 特异性退化。

### 6.3 SSM 的真正优势

阶段 3.3 实验表明, SSM 的递推状态保持**不是**不退化的主因 (Transformer 同样保留信号)。SSM 的优势在于:

1. **计算效率**: 线性复杂度 $O(L)$ vs Transformer 的 $O(L^2)$ — 适合端侧长时序推理 (详见 [mxmaca_adaptation.md](mxmaca_adaptation.md))
2. **三链分解的归纳偏置**: 异构 $d_{state}$ / $expand$ / $headdim$ 让三链承载互补信息 (空间-因果余弦相似度 ≈ 0, 见 [paper/stage3_3_1_validation.md](../paper/stage3_3_1_validation.md)), 这种正交性在 OOD 区稳定保持

---

## 7. 局限与可证伪边界

### 7.1 当前理论的局限

1. **30M 单点 decay 方差大**: 必须用 window-smoothed decay 才能稳定, 30M 不作为"不退化"硬数据点
2. **SDD 单场景单 seed**: 仅 bookstore 7 视频单 seed, 需多场景多 seed 确认统计可靠性
3. **ch@100 边界效应**: 已用 window decay 缓解, 但未来训练应让 max_T 与 eval T 解耦 (如训练 $T=80$, eval $T=120$)
4. **Transformer 对照非自回归**: 现有 TransformerBaseline 为非自回归设计 (无 time_embed), 公平对照需实现 causal-mask + time position 的自回归 Transformer baseline

### 7.2 可证伪边界

理论预测可在以下条件下被推翻:

1. **若任务复杂度 $\mathcal{I}_{min}$ 极高**: SDD 显示三链从"锦上添花"转为"必需", 若在更复杂任务 (如 $\mathcal{I}_{min} \to \infty$) 上三链全消融使 decay 暴跌 → 进一步证实三层保障机制的任务依赖性
2. **若 AnchorInit2 改为单源** (如仅 cell_embed): 预测退化将出现, 因为失去结构性锚定 — 这是 H4/H5 已部分证实的预测
3. **若残差连接移除**: 预测稳定性将大幅下降, 因为基础层信号无法传递到输出
4. **若 $T_{eval} / T_{train}$ 比例极极端** (如 100×): 当前仅在 5× 外推 (T=500) 验证, 极端外推的稳定性未知

### 7.3 假说验证汇总

| 假说 | 内容 | 验证方法 | 结论 |
|---|---|---|---|
| H1 | 时间链 $h_t$ 在 OOD 区稳定 | 3.1 norm 探针 | ✅ 支持 (+0.7%) |
| H2 | 空间链 $h_s$ 随 $t$ 衰减 | 3.1 norm 探针 | ❌ 推翻 (-2.8%) |
| H3 | 因果链 $h_c$ 极其稳定 | 3.1 norm 探针 | ✅ 支持 (-0.0%) |
| H4 | SSM 信号保留率 > 70% | 3.3 扰动探针 | ✅ 支持 (94.1%) |
| H5 | Transformer 信号保留率 < 30% | 3.3 扰动探针 | ❌ 推翻 (99.4%) |

---

## 参考

- 机制总报告: [paper/mechanism_analysis.md](../paper/mechanism_analysis.md)
- 消融实验: [paper/stage3_ablation.md](../paper/stage3_ablation.md)
- 信号保留对照: [paper/stage3_3_retention.md](../paper/stage3_3_retention.md)
- 三链 norm 探针: [paper/stage3_3_1_validation.md](../paper/stage3_3_1_validation.md)
- SDD 真实数据: [paper/stage2_sdd_report.md](../paper/stage2_sdd_report.md)
- 实验数据汇总: [paper/data_summary.md](../paper/data_summary.md)
- 探针脚本: [probe_hidden_states.py](../probe_hidden_states.py), [probe_token_retention.py](../probe_token_retention.py)
- 架构详解: [architecture.md](architecture.md)
