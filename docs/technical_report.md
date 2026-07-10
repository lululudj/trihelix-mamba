# 三链 DNA-Mamba3 + BP v2.1 碱基对耦合 技术说明文档

> 本文档详细说明 ThreeChainMamba3 的技术架构、核心算法、BP v2.1 加法修复原理、参数量表、实验设计与国产 GPU 适配方案。
> 关联代码: [models/three_chain_mamba3.py](../models/three_chain_mamba3.py) · [models/mamba3_ref.py](../models/mamba3_ref.py) · [battle32_c500_gpu.py](../battle32_c500_gpu.py)

---

## 目录

1. [项目技术架构详解](#1-项目技术架构详解)
2. [核心算法说明](#2-核心算法说明)
3. [BP v2.1 加法修复技术原理](#3-bp-v21-加法修复技术原理)
4. [模型参数量表](#4-模型参数量表)
5. [32 场景实验设计](#5-32-场景实验设计)
6. [国产 GPU (C500/MXMACA) 适配技术细节](#6-国产-gpu-c500mxmaca-适配技术细节)
7. [性能优化措施](#7-性能优化措施)

---

## 1. 项目技术架构详解

### 1.1 架构总览

ThreeChainMamba3 是在统一时空张量 `x: (B, T, N², d)` 上以三条异构 Mamba3 链做并行扫描的架构。与上一代 ThreeChainMamba2 相比，核心升级是将 Mamba2 (SSD) 替换为 Mamba3 (梯形离散化 + dt-RoPE 复数状态)，并引入 BP v2.1 两两碱基对耦合机制。

```
              输入: S_0 (B,N,N),  actions (B,K,T)
                          │
            ┌─────────────┴──────────────┐
            │       AnchorInit2          │   x = cell_emb(S_0)
            │  x = cell + act_mean       │     + act_mean(actions)
            │     (B,T,N²,d)            │     (不加 time_embed!)
            └─────────────┬──────────────┘
                          │ x,  act_emb (B,K,T,d)
                          ▼
   ┌──────────────────────┴──────────────────────┐
   │           HeteroMamba3 Layer × L             │
   │                                              │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
   │  │ 空间链    │  │ 时间链    │  │ 因果链    │   │
   │  │ (双向    │  │ (因果    │  │ (K 维    │   │
   │  │  Mamba3) │  │  Mamba3) │  │  扫描)   │   │
   │  │ 行+列    │  │ 沿 T    │  │ 沿 K    │   │
   │  │  h_s     │  │  h_t    │  │  h_c    │   │
   │  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
   │       │             │             │         │
   │       └──────┬──────┴──────┬──────┘         │
   │              ▼             ▼                 │
   │     [BP v2.1 两两碱基对耦合] (可选)           │
   │     BP1: 空间↔时间  BP2: 时间↔因果  BP3: 因果↔空间 │
   │              │             │                 │
   │              ▼ 残差融合     ▼                 │
   │     x = LayerNorm(x + x_s + x_t + c_inject) │
   └──────────────────────┬───────────────────────┘
                          │ x^(L)
                          ▼
            ┌─────────────────────────┐
            │   输出头: Linear(d->C)    │
            │   logits (B,T,N,N,C)    │
            └─────────────────────────┘
```

### 1.2 三链异构分工

| 链 | 扫描维度 | Mamba3 配置 | 方向 | 建模目标 |
|---|---|---|---|---|
| 空间链 (Spatial) | N² (行+列各一次) | d_state=64, expand=2, headdim=64 | 双向 | N×N 网格的局部空间结构 |
| 时间链 (Temporal) | T (沿时间轴) | d_state=64, expand=2, headdim=64 | 因果 | 长程时间依赖 |
| 因果链 (Causal) | K (沿 agent 维) | d_state=64, expand=2, headdim=64 | 因果 | agent 间优先级序 (低 ID 优先) |

### 1.3 核心升级点 (Mamba2 → Mamba3)

1. **移除固定长度 time_embed**: 删除 Mamba2 时代遗留的 `nn.Embedding(max_T, d_model)`，这是外推衰减的罪魁祸首——T > max_T 时要么越界要么用未训练的随机向量，还会干扰 Mamba3 内部 RoPE 的位置信号。
2. **完全依赖 Mamba3 内核的 dt-RoPE**: 位置感知完全交给 Mamba3 内部的 dt-scaled RoPE，支持 T → 32K+ 无衰减外推。
3. **复数状态空间**: 通过 RoPE 旋转 B/C (key/query) 投影，SSM 状态获得等效复值结构，可以追踪旋转/振荡/周期模式。
4. **梯形离散化**: trapezoidal rule 替代 Mamba2 的 ZOH，大步长下离散化误差更小。

### 1.4 AnchorInit2 初始锚定

```python
cell_emb = self.cell_embed(S_0.reshape(B, N²))   # (B, N², d)
act_emb = self.action_embed(actions)              # (B, K, T, d)
act_mean = act_emb.mean(dim=1)                    # (B, T, d)
x = cell_emb.unsqueeze(1) + act_mean.unsqueeze(2) # (B, T, N², d)
# 注意: 不加 time_embed! 位置感知完全由 Mamba3 内部 dt-RoPE 提供
```

---

## 2. 核心算法说明

### 2.1 Mamba3 dt-RoPE 复数状态

Mamba3 不使用传统固定位置 RoPE，而是基于时间步 `dt` 自适应累积角度：

```
angle_increments = angle_raw × dt                    # (B, L, H, S) 每步角度增量
cumulative_angles = cumsum(angle_increments, dim=1)   # (B, L, H, S) 累积角度
```

其中 `angle_raw` 是每步学习的旋转速率，`dt` 是经 softplus 激活的时间步。累积角度用于旋转 B 和 C 投影的前 `split_tensor_size` 维（复数对），剩余维度保持实数。

**RoPE 旋转操作** (对 B/C 投影):

```
对每对维度 (a, b) 旋转角度 θ:
    a' = a × cos(θ) - b × sin(θ)
    b' = a × sin(θ) + b × cos(θ)
```

这赋予了 SSM 状态等效复值结构——实数状态无法表示的振荡/周期模式（如运行计数的奇偶性）可以通过旋转角度自然编码。

**关键参数**:
- `rope_fraction = 0.5`: 一半的 d_state 维度参与旋转
- `split_tensor_size = d_state × 0.5` (取偶数)
- `num_rope_angles = split_tensor_size / 2`

### 2.2 梯形离散化 (Exponential-Trapezoidal Discretization)

Mamba2 使用零阶保持 (ZOH / exponential-Euler) 将连续 SSM 转为递推，这是一阶近似，在大步长 `dt` 下精度损失显著。Mamba3 改用梯形法则：

```
h_t = exp(A × dt_t) × h_{t-1} + dt_t × trap_t × (B_t × x_t + B_{t-1} × x_{t-1}) / 2
```

其中 `trap_t` 是学习的 sigmoid 门 (`∈ [0, 1]`)：
- `trap = 0`: 退化为标准 Euler/ZOH 更新 (Mamba2 风格)
- `trap = 1`: 完全梯形混合 (当前和前一步 B×x 的平均)

**混合公式**:

```
Bx_blended = (1 - trap) × Bx_curr + trap × 0.5 × (Bx_curr + Bx_prev)
h_t = exp(ADT) × h_{t-1} + DT × Bx_blended
```

完整递推 (SISO 模式):

```
decay = exp(A × dt)                          # 状态衰减
Bx_curr = einsum("bhp,bhd->bhpd", x_t, B_t)  # B×x 外积
Bx_blended = (1-trap)×Bx_curr + trap×0.5×(Bx_curr + Bx_prev)
h = decay × h + dt × Bx_blended              # 状态更新
y = einsum("bhd,bhpd->bhp", C_t, h) + D × x  # 输出 + skip
```

### 2.3 PairwiseBasePairMamba3 三对碱基对

碱基对耦合模块实现了三链之间的两两直接耦合，借鉴 DNA 双螺旋的碱基对互补配对思想：

| 碱基对 | 耦合方向 | 调制类型 | 机制描述 |
|---|---|---|---|
| BP1 | 空间 ↔ 时间 | Cross-Delta | 时间链生成 dt 系数调制空间链; 空间变化率门控时间链 |
| BP2 | 时间 ↔ 因果 | Reset Gate | 因果链能量门控时间链记忆; 时间相位注入因果链 |
| BP3 | 因果 ↔ 空间 | FiLM 仿射 | 因果链生成 gamma/beta 仿射空间链; 空间校验因果链 |

**三链全局表示提取** (在 N² 和 T 维上 pool):

```python
g_s = x_s.mean(dim=(1, 2))           # (B, d) 空间全局
g_t = x_t.mean(dim=(1, 2))           # (B, d) 时间全局
g_c = c_inject.squeeze(2).mean(dim=1) # (B, d) 因果全局
s_diff = (x_s[:, 1:] - x_s[:, :-1]).mean(dim=(1, 2))  # (B, d) 空间变化率
```

**v2 纯残差应用** (零初始化保证初始 = baseline):

```python
# BP1+BP3 -> 空间链: FiLM 仿射增量
x_s_new = x_s + gamma × delta × x_s + beta

# BP1+BP2 -> 时间链: reset gate + 空间变化率门控 (v2.1 加法独立调制)
x_t_new = x_t + reset_gate × x_t + s_gate × x_t

# BP2+BP3 -> 因果链: 时间相位注入 + 空间校验
c_inject_new = c_inject + 0.1 × t_phase + 0.1 × s_check × c_inject
```

---

## 3. BP v2.1 加法修复技术原理

### 3.1 问题背景：v1 的 alpha 死锁

v1 版本使用 `alpha` (初始 0) 门控整个调制项：

```
x_s_new = x_s + alpha × (gamma × delta × x_s + beta - x_s)
```

**两个致命问题**:

1. **梯度信号极弱**: `alpha` 初始为 0，梯度需通过 `alpha × 调制项` 反传，但 `alpha=0` 时梯度为 0，导致 `alpha` 永久卡死。实测 100 步后 `alpha` 均值仅 0.0007。
2. **信号缩小**: 公式含 `-x_s` 项，即使 `alpha > 0` 但 `gamma/beta = 0` 时，输出 = `x_s × (1 - alpha)`，会缩小原始信号，非中性。

### 3.2 v2 的改进：去掉 alpha，改纯残差 + 零初始化

v2 去掉 `alpha` 门控，改纯残差形式：

```
x_s_new = x_s + gamma × delta × x_s + beta
```

- 调制网络末层零初始化 → 初始 `gamma=0, beta=0` → 输出 = `x_s` (等价 baseline)
- 训练后调制网络直接生效，无 alpha 瓶颈
- 纯残差形式，不含 `-x_s` 项，不会缩小原始信号

### 3.3 v2 残留问题：乘法梯度死锁

v2 在时间链上使用**乘法**组合两个零初始化网络：

```
# v2 (有 bug): 乘法组合
x_t_new = x_t + reset_gate × s_gate × x_t
```

其中 `reset_gate` 和 `s_gate` 都是零初始化网络的输出。**乘法导致梯度互相阻塞**：

```
∂L/∂reset_gate = ∂L/∂x_t_new × s_gate × x_t
∂L/∂s_gate     = ∂L/∂x_t_new × reset_gate × x_t
```

由于 `reset_gate = 0` 且 `s_gate = 0`（零初始化），两个梯度均为 0，形成**梯度死锁**——两个网络都无法获得梯度信号来更新自身。C500 实测两者均为 0.0000，从未增长。

### 3.4 v2.1 修复：加法独立调制

v2.1 将乘法改为加法，使两个调制项梯度独立：

```
# v2.1 (修复): 加法独立调制
x_t_new = x_t + reset_gate × x_t + s_gate × x_t
```

**数学推导**:

```
∂L/∂reset_gate = ∂L/∂x_t_new × x_t    (不依赖 s_gate)
∂L/∂s_gate     = ∂L/∂x_t_new × x_t    (不依赖 reset_gate)
```

两个梯度独立可流过，不再互相阻塞。

**初始条件验证**:

```
初始时 reset_gate = 0, s_gate = 0 (零初始化):
    x_t_new = x_t + 0 × x_t + 0 × x_t = x_t  ✓  (等价 baseline)

训练后 reset_gate = r, s_gate = s:
    x_t_new = x_t + r × x_t + s × x_t = x_t × (1 + r + s)  ✓
```

**公平对照保证**: 关闭 BP 时参数量不变; 开启 BP 时初始 = baseline (靠零初始化)，训练后调制网络自然学到非零值。

### 3.5 调制强度诊断

v2 无 `alpha` 参数，使用调制网络末层权重范数衡量 BP 激活程度：

```python
def modulation_strength(self):
    """返回各调制网络的权重范数 (用于诊断 BP 是否生效)。"""
    return {
        "bp1_delta": ‖st_t2s_delta 末层权重‖,
        "bp1_gate":  ‖st_s2t_gate 末层权重‖,
        "bp2_reset": ‖tc_c2t_reset 末层权重‖,
        "bp2_phase": ‖tc_t2c_phase 末层权重‖,
        "bp3_gamma": ‖cs_c2s_gamma 末层权重‖,
        "bp3_beta":  ‖cs_c2s_beta 末层权重‖,
        "bp3_check": ‖cs_s2c_check 末层权重‖,
    }
```

训练日志中每 20 步输出 `γ_norm` (BP3 gamma 网络权重范数)，用于实时监控 BP 是否在学习。

---

## 4. 模型参数量表

### 4.1 三链 Mamba3 参数表 (d_model=64, n_layers=1)

| 组件 | 参数量 | 说明 |
|---|---|---|
| cell_embed | 1,024 | `nn.Embedding(16, 64)` |
| action_embed | 320 | `nn.Embedding(5, 64)` |
| 空间链 (行) × 1 层 | ~34,690 | `BidirectionalMamba3(d=64, d_state=64, expand=2, headdim=64)` |
| 空间链 (列) × 1 层 | ~34,690 | 同上 (独立参数) |
| fusion_s × 1 层 | 8,256 | `Linear(128, 64)` |
| 时间链 × 1 层 | ~34,690 | `Mamba3(d=64, d_state=64, expand=2, headdim=64)` |
| 因果链 × 1 层 | ~34,690 | 同上 |
| norm_fuse × 1 层 | 128 | `LayerNorm(64)` |
| causal_inject × 1 层 | 4,160 | `Linear(64, 64)` |
| 输出头 | 2,736 | `LayerNorm + Linear(64,32) + GELU + Linear(32,16)` |
| **三链 Mamba3 总计** | **168,705** | **d_model=64, n_layers=1** |

### 4.2 BP v2.1 附加参数 (enable_bp=True 时)

| 组件 | 参数量 | 说明 |
|---|---|---|
| BP1: st_t2s_delta | 1,153 | `Linear(64,16) + SiLU + Linear(16,1) + Sigmoid` |
| BP1: st_s2t_gate | 1,216 | `Linear(64,16) + SiLU + Linear(16,64)` |
| BP2: tc_c2t_reset | 1,216 | 同上结构 |
| BP2: tc_t2c_phase | 1,216 | 同上结构 |
| BP3: cs_c2s_gamma | 1,216 | 同上结构 |
| BP3: cs_c2s_beta | 1,216 | 同上结构 |
| BP3: cs_s2c_check | 1,281 | 末层多一个 Tanh (无额外参数) |

### 4.3 对照模型参数 (battle32 实验, d_model=64)

| 模型 | n_layers | 参数量 | 说明 |
|---|---|---|---|
| Transformer | 3 | ~167K | 3 层自注意力, 4 头 |
| Mamba3 单链 | 1 | ~84K | 单条 Mamba3 链 |
| 三链 Mamba3 | 1 | 168,705 | 三链异构, 无 BP |
| 三链 Mamba3 + BP | 1 | ~177K | 三链 + BP v2.1 碱基对耦合 |

### 4.4 Mamba3 内部参数约束

```
d_inner = expand × d_model = 2 × 64 = 128
nheads = d_inner / headdim = 128 / 64 = 2
约束: d_inner 必须被 headdim 整除
注意: Mamba3 无 d_conv 参数 (用梯形离散化代替卷积)
```

---

## 5. 32 场景实验设计

### 5.1 实验矩阵

32 场景 × 4 模型 × 3 seeds = 384 组实验，覆盖四类场景：

| 场景类型 | 场景数 | 描述 | 动作生成方式 |
|---|---|---|---|
| random | 8 | 完全随机动作 | `rng.integers(0, 5, size=(K, T))` |
| goal_directed | 12 | 每智能体有目标格，贪心选最短路径 | 70% 贪心 + 30% 随机 |
| adversarial | 8 | 两两配对，pursuer 追 escaper | 70% 追逃 + 30% 随机 |
| extreme | 4 | 极端配置 (最小最快/最大最密等) | 混合 |

### 5.2 场景配置详表

每场景配置为 `(name, N, K, T_train, T_ood, p_transfer, scenario_type, description)`:

**random (8 场景)**:
| 名称 | N | K | p_transfer | 描述 |
|---|---|---|---|---|
| rand_n6_k4 | 6 | 4 | 0.0 | 随机:小网格少agent |
| rand_n6_k8 | 6 | 8 | 0.15 | 随机:小网格中agent |
| rand_n8_k4 | 8 | 4 | 0.0 | 随机:中网格少agent |
| rand_n8_k8 | 8 | 8 | 0.15 | 随机:中网格中agent |
| rand_n8_k12 | 8 | 12 | 0.3 | 随机:中网格多agent |
| rand_n12_k8 | 12 | 8 | 0.15 | 随机:大网格中agent |
| rand_n12_k12 | 12 | 12 | 0.3 | 随机:大网格多agent |
| rand_n6_k12 | 6 | 12 | 0.6 | 随机:高密度强转移 |

**goal_directed (12 场景)**:
| 名称 | N | K | p_transfer | 描述 |
|---|---|---|---|---|
| goal_n6_k4 | 6 | 4 | 0.0 | 目标:小网格少agent |
| goal_n6_k8 | 6 | 8 | 0.15 | 目标:小网格中agent |
| goal_n8_k4 | 8 | 4 | 0.05 | 目标:中网格少agent |
| goal_n8_k8 | 8 | 8 | 0.15 | 目标:中网格中agent |
| goal_n8_k12 | 8 | 12 | 0.3 | 目标:中网格多agent |
| goal_n12_k4 | 12 | 4 | 0.05 | 目标:大网格少agent |
| goal_n12_k8 | 12 | 8 | 0.15 | 目标:大网格中agent |
| goal_n12_k12 | 12 | 12 | 0.3 | 目标:大网格多agent |
| goal_n6_k4_p6 | 6 | 4 | 0.6 | 目标:强转移 |
| goal_n8_k8_p6 | 8 | 8 | 0.6 | 目标:中密度强转移 |
| goal_n12_k4_l | 12 | 4 | 0.0 | 目标:大网格长程 |
| goal_n6_k12_l | 6 | 12 | 0.3 | 目标:高密度长程 |

**adversarial (8 场景)**:
| 名称 | N | K | p_transfer | 描述 |
|---|---|---|---|---|
| adv_n6_k4 | 6 | 4 | 0.0 | 对抗:小网格少agent |
| adv_n6_k8 | 6 | 8 | 0.15 | 对抗:小网格中agent |
| adv_n8_k4 | 8 | 4 | 0.05 | 对抗:中网格少agent |
| adv_n8_k8 | 8 | 8 | 0.15 | 对抗:中网格中agent |
| adv_n8_k12 | 8 | 12 | 0.3 | 对抗:中网格多agent |
| adv_n12_k8 | 12 | 8 | 0.15 | 对抗:大网格中agent |
| adv_n12_k12 | 12 | 12 | 0.3 | 对抗:大网格多agent |
| adv_n6_k12_p6 | 6 | 12 | 0.6 | 对抗:高密度强转移 |

**extreme (4 场景)**:
| 名称 | N | K | p_transfer | scenario_type | 描述 |
|---|---|---|---|---|---|
| ext_n6_k4 | 6 | 4 | 0.0 | goal_directed | 极端:最小最快 |
| ext_n12_k12 | 12 | 12 | 0.6 | adversarial | 极端:最大最密 |
| ext_n8_k8_hp | 8 | 8 | 0.6 | random | 极端:高转移随机 |
| ext_n8_k4_drone | 8 | 4 | 0.05 | goal_directed | 极端:无人机轨迹 |

### 5.3 数据划分

每场景独立生成:
- **train**: 60 样本, T = T_train (100 步)
- **val**: 16 样本, T = T_train (100 步)
- **ood**: 16 样本, T = T_ood (200 步, 测试长程外推)

### 5.4 评估指标

| 指标 | 定义 | 意义 |
|---|---|---|
| val_acc | val 集全部 cell 准确率 | 基础预测能力 |
| val_ch_acc | val 集变化 cell 准确率 | 核心区分指标 (排除不变 cell 的 trivial 正确) |
| ood_acc | OOD 集全部 cell 准确率 | 长程外推基础能力 |
| ood_ch_acc | OOD 集变化 cell 准确率 | 长程外推核心指标 |
| ood_decay_pct | `(ood_ch_acc - val_ch_acc) / val_ch_acc × 100%` | 外推衰减率 (越接近 0 越好) |

### 5.5 统计检验

使用 Welch's t-test (不等方差) 对三链 Mamba3+BP vs 其他 3 个模型做统计检验:
- **t 统计量**: 衡量均值差异的显著性
- **p 值**: `< 0.05` 标记为显著 (`*`), `< 0.01` (`**`), `< 0.001` (`***`)
- **Cohen's d**: 效应量 (`|d| > 0.8` 为大效应)
- **95% CI**: 均值差的置信区间

全场景合并样本量 n = 32 场景 × 3 seed = 96。

---

## 6. 国产 GPU (C500/MXMACA) 适配技术细节

### 6.1 MXMACA 软件栈架构

```
┌─────────────────────────────────────────────────┐
│  ThreeChainMamba3 (应用层)                       │
│  - models/three_chain_mamba3.py (PyTorch Module)│
│  - battle32_c500_gpu.py                          │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  PyTorch (MXMACA 适配版)                         │
│  - torch.nn: Linear, LayerNorm, Embedding, GELU │
│  - torch.utils.checkpoint                       │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  mamba_ssm Triton 内核 / mamba3_ref 纯Python     │
│  - selective_scan_fn (Mamba2 内核, 兼容)         │
│  - mamba3_ref 纯Python SSM scan (C500 主力)     │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  Triton-MXMACA 编译后端                          │
│  - Triton IR → 沐曦 GPU 指令                    │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│  MXMACA cu-bridge (92.94% CUDA 兼容)            │
└──────────────────────┬──────────────────────────┘
                       ▼
│  沐曦曦云 C500 GPU                               │
```

### 6.2 关键环境变量

```bash
source /opt/maca/env.sh
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib
export MAMBA3_FORCE_REF=1           # 强制使用 mamba3_ref 纯Python版 (C500 主力)
export MAMBA3_FORCE_GPU_SCAN=1      # mamba3_ref 的 scan 在 GPU 上跑 (C500 无 TDR 限制)
export MAMBA3_USE_TRITON_SCAN=0     # 禁用 selective_scan_fn Triton 内核 (用纯Python GPU scan)
export PYTORCH_MACA_ALLOC_CONF=max_split_size_mb:128  # 内存分配优化
```

### 6.3 Mamba3 后端自动选择策略

```python
def make_mamba3(d_model, d_state, expand, headdim, chunk_size=64):
    """自动选择后端:
        - 官方 mamba_ssm 2.3.x (Triton内核快) -> 传 chunk_size/is_outproj_norm
        - mamba3_ref 纯Python (C500/本地验证) -> 省略Triton特有参数
    """
    force_ref = os.environ.get("MAMBA3_FORCE_REF", "0") == "1"
    if HAS_OFFICIAL_MAMBA3 and not force_ref:
        return Mamba3(d_model, d_state, expand, headdim, ...)  # 官方 Triton
    else:
        return Mamba3Ref(d_model, d_state, expand, headdim, ...)  # 纯Python
```

**C500 上使用 `MAMBA3_FORCE_REF=1`** 的原因:
1. 官方 Mamba3 Triton 内核在 MXMACA 后端可能编译失败 (edge case)
2. `mamba3_ref` 纯 Python 版本在 GPU 上跑 (通过 `MAMBA3_FORCE_GPU_SCAN=1`)，数学等价且更稳定
3. 纯 Python scan 在 C500 上无 Windows TDR 限制 (WSL2 需要 CPU 回退)

### 6.4 WSL2 vs C500 的 scan 设备选择

```python
# mamba3_ref.py 中的设备选择逻辑
_force_gpu = os.environ.get("MAMBA3_FORCE_GPU_SCAN", "0") == "1"
_is_maca = "MACA_PATH" in os.environ
if _force_gpu or _is_maca:
    device = orig_device          # C500: GPU 上跑
else:
    x = x.cpu()                   # WSL2: CPU 上跑 (规避 Windows TDR)
    device = torch.device("cpu")
```

---

## 7. 性能优化措施

### 7.1 cuda.synchronize 防护

C500 (MXMACA) 上存在 `rq_qos_wait` 死锁风险，通过每步训练后显式同步防护：

```python
# battle32_c500_gpu.py 中的防护策略
logits, info = model(S_0, actions)
loss = compute_loss(model, logits, S_t, info)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
torch.cuda.synchronize()  # ★ 每步同步, 防止 kernel 队列堆积
```

**同步点设置**:
- 每步训练后: `torch.cuda.synchronize()`
- 模型构建后: `torch.cuda.synchronize()`
- 评估前后: `torch.cuda.synchronize()`
- 模型间切换: `synchronize() + gc.collect() + empty_cache() + sleep(0.3)`

### 7.2 BATCH_SIZE = 1

C500 显存有限，强制 `BATCH_SIZE = 1` 保证内存安全：

```python
D_MODEL = 64
BATCH_SIZE = 1      # C500 内存安全
MAX_STEPS = 100
```

配合 `GroupedByNSampler` 保证每个 batch 内 N、K、T 一致。

### 7.3 梯度检查点 (Gradient Checkpointing)

对于大模型 (d_model=256+)，使用梯度检查点突破 OOM：

```python
def _ckpt_call(mamba, x, use_checkpoint=False):
    """梯度检查点包装 Mamba3 调用。
    backward 时重算 forward, 内存 O(L)->O(1)。
    eval 时 (no_grad) 自动直通, 不影响推理速度。
    """
    if use_checkpoint and torch.is_grad_enabled() and x.requires_grad:
        return cp.checkpoint(mamba, x, use_reentrant=True)
    return mamba(x)
```

**注意**: Mamba3 Triton 内核的 backward 多次访问 `ctx.saved_tensors`，与 `use_reentrant=False` 不兼容，必须使用 `use_reentrant=True`。

d_model=64 时 T=200 峰值仅 1.49GB，通常无需开启 `use_checkpoint`。

### 7.4 OOM 恢复机制

```python
except RuntimeError as e:
    emsg = str(e).lower()
    if "out of memory" in emsg:
        log(f"[OOM] step {step}, skip batch")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        continue                    # 跳过当前 batch, 继续训练
    if "device-side assert" in emsg or "cuda error" in emsg or "maca" in emsg:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        raise                        # GPU 硬件错误, 上抛重试
```

### 7.5 连续错误重置

连续 5 个错误后自动重置 GPU 状态：

```python
errors_row += 1
if errors_row >= 5:
    log("!!! 连续5个错误, 尝试重置GPU状态 !!!")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    errors_row = 0
```

Shell 层面还有 `mx-smi -r` GPU reset 和最多 10 次自动重试 (见 `c500_run_battle32_v2.sh`)。

### 7.6 断点续跑

结果文件 `battle32_results.json` 持久化保存，支持断点续跑：

```python
if RESULTS_PATH.exists():
    all_results = json.load(open(RESULTS_PATH))
    log(f"已有结果: {len(all_results)} 条, 断点续跑")

# 每完成一个实验就保存
for scen in SCENARIOS_32:
    for seed in SEEDS:
        for model in MODELS:
            key = f"{scen_name}|seed{seed}|{display_name}"
            if key in all_results and "error" not in all_results[key]:
                log(f"[skip] {key}")      # 跳过已完成
                continue
            # ... 训练 ...
            all_results[key] = result
            json.dump(all_results, open(RESULTS_PATH, "w"))  # 即时保存
```
