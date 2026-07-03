# 相关工作文献定位 (related_work)

> 本文件用于定位 ThreeChainMamba2 与现有工作的差异,不是综述。每条记录聚焦三件事:核心贡献 / 与本工作关系 / 需对比指标。
> 检索日期: 2026-07-02。所有 arXiv 链接经 WebSearch 实测可见。

---

## 1. Mamba 系列 SSM

### 1.1 Mamba (Gu & Dao 2023)
- 论文: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (arXiv:2312.00752, ICML 2024)
- 核心贡献: 提出 S6 选择性状态空间模型。让 SSM 的 B、C、Δ 参数成为输入的函数,使模型能"按内容选择性记忆/遗忘",同时用硬件感知的并行扫描算法保持 O(N) 复杂度。在语言、音频、基因组建模上首次让 SSM 在质量上匹敌甚至超越同规模 Transformer。
- 与本工作关系: 三链架构中每条链的底层都是选择性 SSM 的变体。本工作的因果/空间/时间三链可以理解为把 Mamba 单链的"选择性"机制沿三个正交维度分别实例化。
- 需对比指标: O(N) 线性复杂度基线、长程序列建模能力、selective copy / induction head 合成任务表现 (本工作 GridWorld toy 任务在精神上同属"信息密集离散序列")。

### 1.2 Mamba-2 (Dao & Gu 2024) — 本工作基线
- 论文: [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) (arXiv:2405.21060)
- 核心贡献: 提出结构化状态空间对偶 (SSD) 框架,证明选择性 SSM 与半可分矩阵、结构化掩码注意力 (SMA) 的等价性。Mamba-2 把 A 的结构从对角进一步简化为"标量 × 单位阵",用矩阵乘法重写 selective scan,训练速度比 Mamba-1 快 2-8×,允许 8× 以上的更大状态维度。
- 与本工作关系: **本工作的直接基线**。三链中的每条链都是 Mamba2 block。本工作在 100M→1.13B 参数区间做外推,需要和 Mamba-2 单链在同等参数下对照。
- 需对比指标: 100M/300M/700M/1.13B 各档位的 in-distribution loss、OOD decay (%),训练吞吐 (tokens/s),状态利用率。这是论文 Table 1 的核心对照。

### 1.3 Mamba-3 (Lahoti et al., ICLR 2026 Oral)
- 论文: [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569) (arXiv:2603.15569; OpenReview: [HwCvaJOiCj](https://openreview.net/forum?id=HwCvaJOiCj), ICLR 2026 Oral)
- 核心贡献: 从"推理优先"视角引入三项改进 — (1) 指数-梯形离散化 (exponential-trapezoidal) 取代 ZOH/Euler,获得更有表达力的递推;(2) 复值状态转移提升状态跟踪能力 (如奇偶性);(3) 多输入多输出 (MIMO) 提高解码并行度。1.5B 规模下游准确率比 Gated DeltaNet 高 1.8 个百分点。
- 与本工作关系: **本工作的算法层退化对照**。项目报告 Mamba3 在 OOD 上 decay=-11.5%,是"算法层面退化"的负面对照,用以说明仅升级单链算法 (而不引入三链并行) 不足以保住长程外推。
- 需对比指标: 同参数下 OOD decay、状态跟踪任务准确率、推理算术强度。
- ⚠️ **重要说明 (待用户核实)**: 任务书把 Mamba3 描述为 "heavy_tail_activation 相关"。但实测检索到的官方 Mamba-3 论文 (ICLR 2026 Oral) 并不涉及 heavy-tail activation,而是离散化/复值/MIMO。"heavy-tailed activations" 出现在另一篇独立工作 TVMamba ( ternary 量化 Visual Mamba, [OpenReview wbqPX3jOS4](https://openreview.net/pdf?id=wbqPX3jOS4)) 中,与本工作的 Mamba3 退化对照是否同一对象需用户确认。本条按官方 Mamba-3 记录。

---

## 2. JEPA 系列

### 2.1 LeCun 2022 立场论文 — JEPA 理论源头
- 论文: [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) (OpenReview position paper, 2022-06-27)
- 核心贡献: 提出联合嵌入预测架构 (JEPA) 作为世界模型骨架。核心主张: 不在像素/token 空间做生成式预测,而在学到的抽象表征空间做预测,从而丢弃不可预测的细节。引入能量模型视角和表征塌缩 (collapse) 分类法,提出 H-JEPA 分层预测蓝图。
- 与本工作关系: 本工作的 "JEPA 钩子" 机制在精神上继承 LeCun "在表征空间预测" 的思想,但有重要差异 (见下)。LeCun 的 JEPA 是自监督表征学习框架,目标函数是预测误差能量;本工作的钩子是**在已训练好的 SSM 状态空间上挂接辅助预测头**,用作长程外推的稳定化信号,而非表征学习主目标。
- 需对比指标: 钩子位置 (latent vs raw)、是否需要 target encoder (EMA)、塌缩防护机制。
- 备注: 任务书将本条标为 "arXiv:2301.08243",但该 arXiv 号实为 I-JEPA (2023, 见 2.2)。LeCun 2022 原始立场论文发表于 OpenReview 而非 arXiv,故此处用 OpenReview 链接。

### 2.2 I-JEPA (Assran et al. 2023)
- 论文: [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) (arXiv:2301.08243, CVPR 2023; LeCun 为共同作者)
- 核心贡献: JEPA 在图像上的首个具体实现。从单图 context block 预测多个 target block 的表征,无需手工数据增强。关键设计: target block 要足够大 (semantic),context block 要空间分布,以避免表征塌缩并产出语义表征。
- 与本工作关系: 提供了 JEPA 落地的工程范式 (context-target 划分、target encoder EMA、masking 策略)。本工作的"钩子"如采用 JEPA 式目标,可参考其塌缩防护。
- 需对比指标: 是否使用 EMA target encoder、mask 比例、塌缩检测指标 (方差/协方差)。

### 2.3 V-JEPA / V-JEPA 2 (Bardes et al. 2024, 2025)
- 论文: [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) (V-JEPA, arXiv:2404.08471); [V-JEPA 2](https://arxiv.org/abs/2506.09985) (arXiv:2506.09985, 2025)
- 核心贡献: 把 JEPA 从静态图像扩展到视频时空块。在掩码的时空区域上预测表征,学到含动态语义的特征。V-JEPA 2 扩展到十亿参数,成为后续 VLA/VL-JEPA 的视觉骨干。
- 与本工作关系: V-JEPA 处理的是"时空块预测",与本工作的"时间链 + 空间链"在结构上有共鸣。但 V-JEPA 是自监督预训练,本工作是监督下的世界模型外推,目标不同。
- 需对比指标: 时空 mask 策略、是否预测未来 vs 预测掩码区域。

---

## 3. 世界模型

### 3.1 Ha & Schmidhuber 2018 — World Models 原始论文
- 论文: [World Models](https://arxiv.org/abs/1803.10122) (arXiv:1803.10122, NeurIPS 2018)
- 核心贡献: 把 agent 拆为 V (VAE 视觉) + M (MDN-RNN 记忆/动力学) + C (线性控制器) 三模块。在学到的世界模型 M 内部 ("梦"中) 训练控制器,再零样本迁移到真实环境。CarRacing 达到人类水平,VizDoom Take Cover 超越人类。
- 与本工作关系: **本工作的应用场景定位参照**。本工作的 GridWorld 离散符号动力学世界模型,在精神上继承 V+M+C 的"压缩空间 + 时间动力学"分解。差异: 本工作的 M 是三链并行 SSM 而非 MDN-RNN,且本工作的核心命题是 M 本身的长程外推稳定性,而非 C 的策略学习。
- 需对比指标: 是否在"梦"中训练、动力学模型类型、外推 horizon。

### 3.2 DreamerV3 (Hafner et al. 2023) — 当前 SOTA 世界模型
- 论文: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (arXiv:2301.04104)
- 核心贡献: 基于 RSSM (Recurrent State-Space Model) 的通用世界模型 RL 算法,固定超参数跨 150+ 任务。引入 symlog/symexp 变换、two-hot 编码、KL free bits 等鲁棒性技巧。首个无人工数据/课程从零在 Minecraft 收集钻石的算法。
- 与本工作关系: **世界模型方向的 SOTA 对照**。DreamerV3 的 RSSM 用 GRNN+离散潜变量做长程动力学,而本工作用三链 Mamba2 做长程动力学。两者都关注跨域鲁棒性,但 DreamerV3 关注"策略学习跨域",本工作关注"动力学外推跨尺度"。
- 需对比指标: 跨域鲁棒性、长程预测误差、潜状态维度效率。

### 3.3 长程时空外推相关工作
- 检索结果: 在 "long-horizon extrapolation" / "OOD temporal extrapolation world model" 方向,未找到与本项目 GridWorld toy 任务直接对标的、以 OOD decay (%) 为核心指标的工作。DreamerV3 的跨域评估最接近但任务设定不同。
- 标记: **未找到 (待补)**。本工作在 OOD 时空外推指标上的直接对照,目前主要靠内部 sanity check (随机标签 decay=-6.07%) 而非外部基线。

---

## 4. 记忆系统

### 4.1 Neural Turing Machines (Graves et al. 2014)
- 论文: [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (arXiv:1410.5401)
- 核心贡献: 把神经网络控制器与外部可微分记忆矩阵耦合,通过注意力式软读写实现端到端训练。NTM 能从输入输出示例推断复制、排序、关联回忆等简单算法,并外推到训练时未见的 2× 序列长度。
- 与本工作关系: **外置记忆范式起点**。NTM 的"控制器 + 外部记忆"分离思想,对应本工作中"SSM 隐状态 = 压缩记忆"的设计。但 NTM 用显式外部矩阵,本工作用 SSM 的隐状态向量做隐式记忆,容量受 d_state 限制。
- 需对比指标: 长度外推倍数 (NTM 2×)、算法任务泛化 (copy/sort)。

### 4.2 End-to-End Memory Networks (Sukhbaatar et al. 2015)
- 论文: [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895) (arXiv:1503.08895, NeurIPS 2015)
- 核心贡献: 在外部记忆上做多次"跳" (hop) 注意力读取,端到端可训练。比 Memory Networks 弱监督需求更低,在问答和语言建模上接近 LSTM。
- 与本工作关系: "多跳记忆"思想可对照本工作的"多链并行"。差异: MemNN 的多跳是串行精炼,本工作的三链是并行解耦 (因果/空间/时间)。
- 需对比指标: 跳数 vs 链数、计算开销、长程依赖捕获。

### 4.3 Transformer-XL (Dai et al. 2019)
- 论文: [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) (arXiv:1901.02860)
- 核心贡献: 引入段级递归 (segment-level recurrence) 把前段隐状态缓存为记忆供下段使用,配以相对位置编码,突破固定上下文窗口。学到比 RNN 长 80%、比 vanilla Transformer 长 450% 的依赖,评估快 1800×。
- 与本工作关系: **Transformer 阵营的长上下文经典对照**。本工作对照实验中的 Transformer-tiny 完全退化 (zero_ratio=1.0),正说明固定窗口 + 无外置记忆的 Transformer 在本任务长程外推上失败 — Transformer-XL 的段级递归是 Transformer 阵营修补此问题的代表路径。
- 需对比指标: 有效上下文长度、外推时是否窗口化、相对 vs 绝对位置。

### 4.4 Compressive Transformer (Rae et al. 2019)
- 论文: [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507) (arXiv:1911.05507, ICLR 2020)
- 核心贡献: 在 Transformer-XL 基础上增加"压缩记忆"层 — 旧记忆不丢弃而是被压缩成更少向量存入二级记忆。双记忆 (短期细粒度 + 长期粗粒度) 机制。WikiText-103 取得 17.1 ppl,Enwik8 0.97 bpc,对稀有词建模显著改善。提出 PG-19 长程基准。
- 与本工作关系: **"压缩记忆"思想的代表**。Compressive Transformer 用有损压缩扩展时间感受野,与本工作用 SSM 隐状态做有损压缩在动机上一致。差异: Compressive Transformer 的压缩是显式子网络,本工作的压缩是 SSM 递推的隐式结果。
- 需对比指标: 压缩比、稀有 token 召回、记忆层级数。

---

## 5. SSM 长程 / OOD 外推相关工作

> 这是与本工作命题最直接相关的方向。本工作的核心主张 "OOD decay 全部落在 ±2% 内" 直接回应了这一系列工作揭示的 Mamba 长度外推失败问题。

### 5.1 DeciMamba (Ben-Kish et al., ICLR 2025)
- 论文: [DeciMamba: Exploring the Length Extrapolation Potential of Mamba](https://arxiv.org/abs/2406.14528) (arXiv:2406.14528, ICLR 2025)
- 核心贡献: 识别 Mamba 长度外推瓶颈源于"受限的有效感受野 (ERF)",由训练长度决定。提出 DeciMamba — 基于 S6 层内嵌的隐式过滤机制,按 Δ_t 大小筛掉不重要 token,使训练好的模型无需再训练即可外推到更长上下文。
- 与本工作关系: **Mamba 长度外推失败的诊断对照**。DeciMamba 认为问题在 ERF;本工作的三链并行可视为另一种补救 — 用空间/时间链分担单因果链的压力,而非单链内做 token 抽取。
- 需对比指标: ERF 测量、外推倍数、LongBench 子任务。

### 5.2 MambaExtend (Azizi et al., ICLR 2025)
- 论文: [MambaExtend: A Training-Free Approach to Improve Long Context Extension of Mamba](https://openreview.net/forum?id=LgzRo1RpLS) (ICLR 2025 Poster)
- 核心贡献: 把 Mamba 长上下文失败归因于 OOD 离散化步长 (Δ_t)。提出免训练方法,用梯度/零阶优化仅为每层校准 Δ 的缩放因子,实现 2k→64k (32×) 上下文扩展。
- 与本工作关系: **OOD 诊断的 Δ 派 vs A 派之争的代表**。MambaExtend 归因于 Δ,Mamba Modulation (5.3) 归因于 A 的谱。本工作的 OOD decay 指标可作为判断这两种归因是否完备的实证参照 — 若三链架构在不动 Δ/不缩放 A 的情况下就能把 decay 压到 ±2%,说明归因可能还有遗漏 (比如"单链信息瓶颈")。
- 需对比指标: 外推倍数、perplexity 增量、参数更新量。

### 5.3 Mamba Modulation / On the Length Generalization of Mamba (Lu et al., NeurIPS 2025) — 最相关
- 论文: [Mamba Modulation: On the Length Generalization of Mamba](https://arxiv.org/abs/2509.19633) (arXiv:2509.19633, NeurIPS 2025; code: [github.com/gnepul-ace/mamba_modulation](https://github.com/gnepul-ace/mamba_modulation))
- 核心贡献: **直接以 "OOD behavior of state-space dynamics" 为题**。证明 Mamba 长度外推失败的根源是状态转移矩阵 A 的谱 — 当 λ→1 状态爆炸,λ→0 状态消失。定理 4.2 给出状态范数随 t→∞ 的收敛行为。提出谱缩放 (spectrum scaling) 方法,对预训练 Mamba 的 A 矩阵谱做压缩大特征值/膨胀小特征值,在仅调 Δ 失效的场景下恢复长上下文泛化。
- 与本工作关系: **命题最贴近的对照工作**。两者都把"长程外推时 SSM 状态动力学 OOD"作为核心问题。差异: (a) Mamba Modulation 通过后处理缩放 A 的谱来修补单链;本工作通过三链并行架构来从根本上避免单链状态爆炸/消失。(b) Mamba Modulation 处理语言建模 perplexity;本工作处理 GridWorld 离散动力学的 OOD decay。(c) Mamba Modulation 是训练后免训练修补;本工作是架构层重设计。
- 需对比指标: OOD decay (%)、状态范数随长度变化曲线、A 谱分布。这是论文 Related Work 章节必须详细对比的核心文献。

### 5.4 Stuffed Mamba (Chen et al., COLM 2025)
- 论文: [Stuffed Mamba: Oversized States Lead to the Inability to Forget](https://arxiv.org/abs/2410.07145) (arXiv:2410.07145, COLM 2025)
- 核心贡献: 把 Mamba-2 长上下文失败归因于"状态过参数化导致学不会遗忘"。状态太大 → 训练损失无需遗忘即可最小化 → 上下文超长时信息干扰 → 性能塌缩。证明最小训练长度随状态大小线性增长,passkey 可准确检索的最大上下文随状态大小指数增长。
- 与本工作关系: **"状态容量"视角的对照**。本工作的三链架构把单链状态切成三个并行小状态 (causal/spatial/temporal),可视为对 Stuffed Mamba 论点的回应 — 与其用一个塞满的大状态,不如用三个分工的小状态。需验证本工作在 1.13B 大状态时是否也出现"学不会遗忘",以及三链是否延迟了该塌缩点。
- 需对比指标: 遗忘曲线 (首 token 保留强度)、状态大小 vs 训练长度标度律、passkey 检索准确率。

### 5.5 Leveraging SSMs in Long Range Genomics (Popov et al. 2025) — 正面证据
- 论文: [Leveraging State Space Models in Long Range Genomics](https://arxiv.org/abs/2504.06304) (arXiv:2504.06304)
- 核心贡献: 在 50M 参数规模下对比 Caduceus/Hawk (SSM) 与 Transformer 基线在基因组长程任务上的表现。发现 SSM 匹配 Transformer 质量,且展现惊人的零样本外推 — 处理比训练长 10-100× 的上下文,提示 SSM 的表征更适合长复杂序列。可在单 GPU 处理 1M token 序列。
- 与本工作关系: **SSM 长程外推的正面证据**。与 5.1-5.4 揭示的失败案例形成张力 — 在合适任务 (基因组) 和合适规模 (50M) 下,SSM 反而能零样本外推 1-2 个数量级。本工作的 100M 起点恰在该工作验证过的"SSM 友好"区间,需说明本任务的 GridWorld 离散动力学是否同样属于 SSM 友好分布。
- 需对比指标: 零样本外推倍数、参数规模拐点、任务类型 (连续 vs 离散)。

---

## 6. 本工作差异化定位 (一句话)

**在 O(N) 线性复杂度下,通过把单链选择性 SSM 重构为因果/空间/时间三链并行架构,在 100M→1.13B 参数区间实现长程时空外推不退化 (OOD decay 全部落在 ±2% 内) — 这是对 Mamba 系列单链外推失败 (DeciMamba/MambaExtend/Mamba Modulation/Stuffed Mamba 各自从 ERF/Δ/A 谱/遗忘角度诊断的同一问题) 的架构层而非修补式回应,与 Transformer-XL/Compressive Transformer 的外置记忆路径、与 Ha&Schmidhuber/DreamerV3 的世界模型路径、与 LeCun JEPA 的表征空间预测路径均正交。**
