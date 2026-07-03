# 三链架构详解与数学方程

> ThreeChainMamba2 架构白皮书 — 面向沐曦青年开源专项基金评审与研究生套磁材料
> 本文档给出完整的数学定义、三链 SSM 参数表与因果性分析, 所有符号与代码严格对齐。
> 实现代码: [three_chain_mamba2.py](../models/three_chain_mamba2.py)

## 目录 (TOC)

1. [架构总览](#1-架构总览)
2. [AnchorInit2 初始锚定方程](#2-anchorinit2-初始锚定方程)
3. [三链扫描数学定义](#3-三链扫描数学定义)
4. [Mamba2 (SSD) 核心方程](#4-mamba2-ssd-核心方程)
5. [残差融合与输出头](#5-残差融合与输出头)
6. [三链参数表](#6-三链参数表)
7. [因果性分析](#7-因果性分析)
8. [实现硬约束](#8-实现硬约束)

---

## 1. 架构总览

ThreeChainMamba2 在统一时空张量 $x:(B, T, N^2, d)$ 上以三条异构 Mamba2 (SSD) 链做并行扫描, 每层残差融合, 解决旧架构六个失败模式 (详见 [three_chain_mamba2.py](../models/three_chain_mamba2.py) 模块 docstring)。

### 1.1 三链架构图 (ASCII Art)

```
                  输入: S_0 (B,N,N),  actions (B,K,T)
                              │
                ┌─────────────┴──────────────┐
                │       AnchorInit2          │   x = cell_emb(S_0)
                │  x = cell + act_mean + t   │     + act_mean(actions)
                │     (B,T,N²,d)            │     + time_emb(t)
                └─────────────┬──────────────┘
                              │ x,  act_emb (B,K,T,d)
                              ▼
       ┌──────────────────────┴──────────────────────┐
       │              HeteroMamba2 Layer × L          │
       │                                              │
       │  ┌──────────┐    ┌──────────┐    ┌──────────┐│
       │  │ 空间链    │    │ 时间链    │    │ 因果链    ││
       │  │ (双向    │    │ (因果    │    │ (K 维    ││
       │  │  Mamba2) │    │  Mamba2) │    │  扫描)   ││
       │  │ 行+列    │    │ 沿 T    │    │ 沿 K    ││
       │  │  h_s     │    │  h_t    │    │  h_c    ││
       │  └────┬─────┘    └────┬─────┘    └────┬─────┘│
       │       │               │               │      │
       │       └───────┬───────┴───────┬───────┘      │
       │               ▼               ▼              │
       │       x_s = Linear([row; col])  c_inject     │
       │               │               │              │
       │               ▼ 残差融合       ▼              │
       │      x = LayerNorm(x + x_s + x_t + c_inject)│
       └──────────────────────┬───────────────────────┘
                              │ x^(L)
                              ▼
                ┌─────────────────────────┐
                │   输出头: Linear(d→C)    │
                │   logits (B,T,N,N,C)    │
                └─────────────────────────┘
```

### 1.2 张量形状约定

| 符号 | 含义 | 形状 |
|---|---|---|
| $B$ | batch size | — |
| $T$ | 时间步数 (训练 $T_{train}$, OOD 评估 $T_{eval}$) | — |
| $N$ | 网格边长 (GridWorld $N=12$, SDD $N=24$) | — |
| $N^2$ | 网格 cell 总数 | — |
| $K$ | agent 数 (SDD $K=8$) | — |
| $d$ | 模型隐藏维度 $d_{model}$ | — |
| $C$ | cell 类别数 (GridWorld $C=16$) | — |
| $L$ | 三链残差融合层数 $n\_layers$ | — |

---

## 2. AnchorInit2 初始锚定方程

AnchorInit2 用三源加法融合生成初始统一张量, 是不退化机制的**基础层**:

$$
\boxed{\; x^{(0)} = E_{cell}[S_0] \;+\; \bar{E}_{act}[A] \;+\; E_{time}[t] \;}
$$

其中:

- $E_{cell}[S_0]$: cell embedding, $S_0 \in \mathbb{Z}^{B \times N^2}$, $E_{cell} \in \mathbb{R}^{C \times d}$, 输出 $(B, N^2, d)$
- $\bar{E}_{act}[A]$: 动作 embedding 对 $K$ 维取均值, $A \in \mathbb{Z}^{B \times K \times T}$, $\bar{E}_{act}[A] = \tfrac{1}{K}\sum_{k=1}^{K} E_{act}[A_{\cdot,k,\cdot}]$, 输出 $(B, T, d)$, 广播到 $N^2$
- $E_{time}[t]$: 时间位置 embedding, $t \in \{0, 1, \ldots, T-1\}$, $E_{time} \in \mathbb{R}^{T_{max} \times d}$, 输出 $(T, d)$, 广播到 $B, N^2$

完整广播: $x^{(0)} \in \mathbb{R}^{B \times T \times N^2 \times d}$

对应代码: [`three_chain_mamba2.py` L348-358](../models/three_chain_mamba2.py)

**关键设计点**:
1. 三源**加法**融合 (非拼接非注意力), 复杂度 $O(1)$ per token
2. cell 信息广播到全部 $T$ 步 (静态环境信息共享)
3. act_mean 广播到 $N^2$ (动作全局信号注入所有 cell)
4. time_emb 沿 $t$ 变化 (唯一的时间位置先验)

---

## 3. 三链扫描数学定义

三链在统一张量 $x^{(\ell)}$ 上以三种**视角**重新排列后扫描, 每链独立配置 $d_{state}$ / $expand$ / $headdim$。

### 3.1 空间链 (Spatial Chain, 双向 Mamba2)

对每个时间步 $t$, 将 $N^2$ 个 cell 视为序列, 分别按行优先 (row-major) 与列优先 (column-major) 顺序双向扫描:

$$
\begin{aligned}
h_s^{row} &= \text{BiMamba2}_{row}\!\left( \text{reshape}(x^{(\ell)}, (B \cdot T, N^2, d)) \right) \\
h_s^{col} &= \text{BiMamba2}_{col}\!\left( \text{reshape}\circ\text{transpose}_{N\times N}(x^{(\ell)}) \right) \\
h_s &= W_s \cdot \big[\, h_s^{row};\, h_s^{col}\, \big] + b_s
\end{aligned}
$$

其中 $\text{BiMamba2}(x) = \text{Mamba2}(x) + \text{flip}(\text{Mamba2}(\text{flip}(x)))$, 共享参数以控制参数量。$W_s \in \mathbb{R}^{d \times 2d}$ 为行/列融合线性投影。

### 3.2 时间链 (Temporal Chain, 因果 Mamba2)

对每个 cell $(i, j)$, 沿时间轴 $T$ 做因果扫描:

$$
h_t = \text{CausalMamba2}\!\left( \text{reshape}(x^{(\ell)}, (B \cdot N^2, T, d)) \right)
$$

输出 $h_t \in \mathbb{R}^{B \times T \times N^2 \times d}$ (重排回原形状)。Mamba2 默认因果 (左→右), 严格保证 $t$ 时刻输出只依赖 $\le t$ 时刻输入。

### 3.3 因果链 (Causal Chain, agent 维因果 Mamba2)

对每个时间步 $t$, 沿 agent 维 $K$ 做因果扫描, 直接扫描 $act\_emb$ (不再 mean 派生):

$$
\begin{aligned}
h_c &= \text{CausalMamba2}\!\left( \text{reshape}(act\_emb, (B \cdot T, K, d)) \right) \\
act\_emb^{(\ell+1)} &= h_c \quad \text{(残差更新 act_emb)} \\
c_{inject} &= W_{inj} \cdot \left( \tfrac{1}{K}\sum_{k=1}^{K} h_c[\cdot,\cdot,k,\cdot] \right)
\end{aligned}
$$

其中 $W_{inj} \in \mathbb{R}^{d \times d}$ 将 agent 维均值池化投影回 $d$ 维, 广播到 $N^2$ 注入主张量。

### 3.4 完整三链方程汇总

$$
\boxed{
\begin{aligned}
x^{(0)} &= E_{cell}[S_0] + \bar{E}_{act}[A] + E_{time}[t] \\
\text{for } & \ell = 0, 1, \ldots, L-1: \\
& h_s = \text{BiMamba2}_{row/col}(x^{(\ell)}) \quad \text{(空间链, } N^2 \text{ 维双向)} \\
& h_t = \text{CausalMamba2}(x^{(\ell)}) \quad \text{(时间链, } T \text{ 维因果)} \\
& h_c = \text{CausalMamba2}(act\_emb^{(\ell)}) \quad \text{(因果链, } K \text{ 维因果)} \\
& x^{(\ell+1)} = \text{LayerNorm}\!\left( x^{(\ell)} + h_s + h_t + c_{inject} \right)
\end{aligned}
}
$$

---

## 4. Mamba2 (SSD) 核心方程

ThreeChainMamba2 基于 Mamba2 的 SSD (State Space Duality) 框架, 下面给出 SSM 的递推形式与 SSD 的结构化矩阵形式。

### 4.1 连续时间状态空间方程 (S4 范式)

$$
\begin{aligned}
h'(t) &= A\, h(t) + B\, x(t) \\
y(t) &= C\, h(t) + D\, x(t)
\end{aligned}
$$

其中 $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$, $D \in \mathbb{R}$。

### 4.2 离散化递推 (S6 / Mamba1)

零阶保持 (ZOH) 离散化:

$$
\bar{A} = \exp(\Delta \cdot A), \quad \bar{B} = (\Delta A)^{-1}(\bar{A} - I) \cdot \Delta B
$$

状态递推:

$$
\boxed{\; h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t \;}, \qquad y_t = C\, h_t + D\, x_t
$$

### 4.3 SSD (State Space Duality) 分块形式 (Mamba2)

Mamba2 将递推改写为**结构化矩阵**形式, 允许通过矩阵乘法并行训练 (前向) 与递推 (推理) 双模式:

$$
\begin{aligned}
h_{t+1} &= \bar{A}\, h_t + \bar{B}\, x_t \\
y_t &= C\, h_t + D\, x_t
\end{aligned}
\quad \Longleftrightarrow \quad
\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_L \end{bmatrix}
= \text{SSD}(M)\cdot
\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_L \end{bmatrix}
$$

其中 $M$ 为**结构化衰减矩阵** (structured mask matrix):

$$
M_{ij} = \begin{cases}
\prod_{k=j+1}^{i} \bar{A}_k & \text{if } i \ge j \\
0 & \text{if } i < j \quad \text{(因果下三角)}
\end{cases}
$$

SSD 的对偶性: 同一个模型既可写成 $O(L)$ 递推形式 (推理快, $h_{t+1} = \bar{A} h_t + \bar{B} x_t$), 也可写成 $O(L^2)$ 注意力形式 (训练并行), 两者数学等价。Mamba2 在多 head / 多状态 (multihead SSD) 下分块计算, 故名 SSD。

### 4.4 heavy_tail_activation (HTA, 从 Mamba3 移植)

A 矩阵的激活函数从 $\exp$ 替换为 heavy_tail:

$$
f(x) = \text{heavy\_tail}(x) = \begin{cases}
x + 1 & \text{if } x \ge 0 \quad \text{(线性, 梯度恒为 1, 不饱和)} \\
\dfrac{1}{1 - x} & \text{if } x < 0 \quad \text{(重尾, } >1 \text{ 但有限)}
\end{cases}
$$

性质: 总是 $> 0$, 连续可微, 比 $\exp$ 更稳 ($\exp$ 在大值爆炸, heavy_tail 正侧线性不爆)。

A 的计算:

$$
A = -\exp(A_{log}) \quad \xrightarrow{\text{HTA parametrization}} \quad A = -\text{heavy\_tail}(r)
$$

其中 $r$ 是原始可学习参数, $A_{log} = \log(\text{heavy\_tail}(r))$ 由 `register_parametrization` 注入, 不修改 mamba_ssm 系统包源码。详见 [three_chain_mamba2.py L59-95](../models/three_chain_mamba2.py)。

---

## 5. 残差融合与输出头

### 5.1 每层残差融合

第 $\ell$ 层三链输出与输入残差融合, 经 LayerNorm:

$$
\boxed{\; x^{(\ell+1)} = \text{LayerNorm}\!\left( x^{(\ell)} + h_s^{(\ell)} + h_t^{(\ell)} + c_{inject}^{(\ell)} \right) \;}
$$

**关键性质**:
1. 三链在**每层**都交换信息 (而非只在最后融合)
2. 残差连接保证三链全消融时 $x^{(\ell+1)} = \text{LayerNorm}(x^{(\ell)})$, 仍能稳定传播 (机制见 [theory.md](theory.md))
3. LayerNorm 数值稳定, 防 SSM 状态爆炸

### 5.2 输出头

最后第 $L$ 层输出 $x^{(L)} \in \mathbb{R}^{B \times T \times N^2 \times d}$ 经 MLP 投影到 $C$ 类:

$$
\boxed{\; \hat{S} = \text{Linear}\!\left( \text{GELU}\!\left( \text{Linear}( \text{LayerNorm}(x^{(L)}) ) \right) \right) \in \mathbb{R}^{B \times T \times N \times N \times C} \;}
$$

具体结构: `LayerNorm(d) → Linear(d, d/2) → GELU → Linear(d/2, C)`, 然后重排 $(B, T, N^2, C) \to (B, T, N, N, C)$。

对应代码: [three_chain_mamba2.py L308-313](../models/three_chain_mamba2.py)。

**奥卡姆剃刀设计**: 输出头用 `Linear` 直接投影, 不用 cross-attention 融合 (旧架构的 Fusion 假注意力失败模式)。

---

## 6. 三链参数表

经 `_probe_mamba2.py` 验证的 Mamba2 三链配置 (参数严格对齐 mamba_ssm 2.x 算子约束):

| 链 | 扫描维度 | 方向 | $d_{state}$ | $d_{conv}$ | $expand$ | $headdim$ | $nheads$ | 备注 |
|---|---|---|---|---|---|---|---|---|
| **空间链 (spatial)** | $N^2$ (行+列) | 双向 | 128 | 4 | 1 | **32** | 8 | $expand=1$ 时 $headdim$ 必须为 32 (causal_conv1d stride 对齐) |
| **时间链 (temporal)** | $T$ | 因果 (左→右) | 64 | 4 | 2 | 64 | 8 | Mamba2 默认因果 |
| **因果链 (causal)** | $K$ (agent) | 因果 (低 ID → 高 ID) | 32 | 4 | 2 | 64 | 8 | 直接扫描 $act\_emb$ 的 $K$ 维 |

每条链的行/列版本各 1 个 Mamba2 (空间链共 2 个), 加每层 1 个 Linear 融合 + 1 个因果注入投影 + 1 个 LayerNorm。

### 6.1 异构设计的物理动机

- **空间链 $d_{state}=128$**: 网格空间结构需要较大状态容量记录 $N^2$ 个 cell 的相对位置
- **时间链 $d_{state}=64$**: 时间动力学相对低维, 因果约束限制了状态演化自由度
- **因果链 $d_{state}=32$**: agent 间交互在 $K$ 较小时信息量小, 小状态足够
- **headdim 异构 (32 vs 64)**: 空间链 $expand=1$ 必须用 32 规避 `causal_conv1d` stride 对齐 bug; 时间/因果链 $expand=2$ 用 64 提升单 head 容量

### 6.2 参数量规模 (30M baseline)

| 配置 | $d_{model}$ | $n_{layers}$ | 参数量 | 用途 |
|---|---|---|---|---|
| 30M baseline | 768 | 2 | 26.59M | 主力 + 消融实验 |
| 100M | 1024 | 4 | 72.32M | 规模扩展 |
| 300M | 1536 | 4 | 182.95M | 规模扩展 |
| 700M | 2048 | 6 | 724.51M | 规模扩展 |
| 1B (gradient_ckpt) | 2560 | 8 | 1.13B | 极限规模 (use_checkpoint=True) |
| deep_n8 | 768 | 8 | 105.39M | 深度对照 |

完整规模-decay 数据见 [paper/data_summary.md](../paper/data_summary.md)。

---

## 7. 因果性分析

三链各自承担不同的因果约束, 是设计层面的归纳偏置:

### 7.1 时间因果 (Temporal Causality, 时间链)

$$
\forall t: \quad \text{logits}[t] = f\!\left( S_0,\; actions_{[:, :, 0:t+1]} \right)
$$

- 实现: Mamba2 默认因果 (左→右), $\bar{A}$ 为下三角
- 含义: $t$ 时刻预测只依赖 $S_0$ 与 $\le t$ 步动作, 严格不自回归预测未来动作
- OOD 外推时: 时间因果保证 $T_{eval} \gg T_{train}$ 仍可前向 (无未来信息泄漏)

### 7.2 空间双向 (Spatial Bidirectionality, 空间链)

$$
\forall (i, j) \in N \times N: \quad h_s[(i,j), t] = f\!\left( \big\{ x[(i', j'), t] \big\}_{(i', j') \in N \times N} \right)
$$

- 实现: `BiMamba2 = Mamba2(x) + flip(Mamba2(flip(x)))`, 行优先 + 列优先各一次
- 含义: 同时间步内所有 cell 互相可见, **无空间因果序** (网格内 cell 并行存在, 不像时间维有序)
- 与 Transformer self-attention 的等价性: 都是无序的 set-to-set 映射, 但 BiMamba2 是 $O(N^2)$ 线性, self-attention 是 $O(N^4)$

### 7.3 Agent 因果 (Agent Causality, 因果链)

$$
\forall k \in \{1, \ldots, K\}: \quad h_c[k] = f\!\left( act\_emb[1], \ldots, act\_emb[k] \right)
$$

- 实现: Mamba2 沿 $K$ 维因果扫描
- 含义: 低 ID agent 的动作先于高 ID agent 被处理, 匹配**冲突解决规则** (低 ID 优先)
- 物理动机: 多智能体系统中, agent 优先级序通常由 ID 决定 (如机器人调度), 因果链将这一先验注入架构

### 7.4 三链因果对比表

| 链 | 扫描方向 | 因果掩码 | $O(L)$ 复杂度 | 物理意义 |
|---|---|---|---|---|
| 空间链 | 双向 (BiMamba2) | 无 (全可见) | $O(N^2 \cdot d)$ | 空间网格无因果序 |
| 时间链 | 单向因果 | 下三角 $\bar{A}$ | $O(T \cdot d)$ | 时间不可逆 |
| 因果链 | 单向因果 | 下三角 $\bar{A}$ | $O(K \cdot d)$ | agent 优先级序 |

---

## 8. 实现硬约束

来自 mamba_ssm 算子层的硬约束 (违反则 build / runtime 报错):

1. **$d_{conv} \in \{2, 3, 4\}$**: `causal_conv1d` 仅支持这三种 kernel size
2. **空间链 $headdim = 32$**: $expand = 1$ 时 $headdim$ 必须为 32, 否则 `causal_conv1d` stride 对齐错位 (参见 mamba_ssm issue tracker)
3. **$headdim$ 整除 $d_{model} \cdot expand$**: 多 head SSD 分块要求
4. **1B 训练需 gradient checkpointing**: $d_{model} = 2560$ 时 forward 激活值 ~25GB > 24GB, 必须 `use_checkpoint=True` 将内存从 $O(L)$ 降至 $O(1)$ (代价: backward 慢 ~1.5×)
5. **SDD $N^2 = 576$ (4× GridWorld)**: `triton SSD backward` 在 24GB GPU 上 OOM, 必须用 `batch=1 + use_checkpoint=True`

gradient checkpointing 实现: [three_chain_mamba2.py L41-51](../models/three_chain_mamba2.py) (`_ckpt_call` 包装)。

---

## 参考

- 实现代码: [models/three_chain_mamba2.py](../models/three_chain_mamba2.py)
- 共享组件 (AnchorInit / Bind / Fusion / JEPA / M3Snapshot): [models/common.py](../models/common.py)
- 训练入口: [train.py](../train.py)
- OOD 评估: [eval_ood.py](../eval_ood.py)
- 理论推导: [theory.md](theory.md)
- 国产 GPU 适配: [mxmaca_adaptation.md](mxmaca_adaptation.md)
- 实验数据汇总: [paper/data_summary.md](../paper/data_summary.md)
