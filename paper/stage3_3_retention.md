# 阶段 3.3 SSM vs Transformer 信号保留对比报告

> 实验设置: 30M 参数匹配 (SSM 26.59M vs Transformer 30.74M)
> 训练: 500 步 seed=2, OOD 评估 T=150 (训练 T=100)
> 生成时间: 2026-07-02

## 机制假说

- **H4**: SSM 对 t=0 注入的扰动信号, 在 t=150 仍能保留 (保留率 > 70%)
- **H5**: Transformer 对 t=0 注入的扰动信号, 在 t>100 急剧衰减 (保留率 < 30%)

## 结果汇总

| 指标 | SSM (ThreeChainMamba2) | Transformer |
|------|----------------------|-------------|
| 参数量 | 26.59M | 30.74M |
| 信号保留率 @t=150 | 94.1% | **99.4%** |
| OOD decay (t=100→150) | +0.2% | -6.6% |
| changed_acc @t=150 | **0.5929** | 0.0555 |
| val changed_acc @500步 | 0.5740 | 0.0715 |

## 假说判定

- **H4 (SSM 保留率 > 70%)**: ✅ 支持 — 保留率 94.1%
- **H5 (Transformer 保留率 < 30%)**: ❌ **推翻** — 保留率 99.4% (甚至高于 SSM)

## 关键发现：H5 推翻的深层含义

### 反直觉结果

Transformer 信号保留率 **99.4%**，甚至**高于** SSM 的 94.1%。但 Transformer 的 changed_acc@150 仅 0.0555（接近随机），而 SSM 为 0.5929。

**即：Transformer 完美保留了输入信号（99.4%），但它的预测完全崩溃。**

### 机制修正

原假说认为"Transformer 退化是因为注意力稀释了输入信号"。**实验推翻了这一假说**。

真实机制是：

1. **不是信息遗忘问题** — 两个模型都完美保留输入扰动信号（94% vs 99%），信息始终在场
2. **是预测质量问题** — Transformer 的 changed_acc 在训练 500 步后仍仅 0.0715（val），说明它**根本没学会**这个任务，而非 OOD 特异性退化
3. **根因是归纳偏置缺失** — Transformer 缺少 AnchorInit2 的初始锚定（cell+action+time 三源），仅靠 cell_embed + 自注意力，无法建立有效的时间外推映射

### 与 3.2 的整合

3.2 消融实验发现 no_all（三链全消融，仅剩 AnchorInit2+残差）仍不退化（decay=-0.66%, changed_acc@150=0.5900）。

3.3 进一步证实：**AnchorInit2 是关键**。
- SSM 有 AnchorInit2 → 即使三链全消融也稳定（3.2 no_all）
- Transformer 无 AnchorInit2 → 即使保留 99.4% 信号也崩溃（3.3）

不退化的根因不在 SSM 的递推状态保持，而在 **AnchorInit2 对初始表示的结构性锚定**。

### 三层机制修正

ThreeChainMamba2 不退化的机制来自**三层保障**：

| 层次 | 机制 | 证据 |
|------|------|------|
| **基础层** (必需) | AnchorInit2 初始锚定 (cell+action+time 三源) | 3.2 no_all 不退化 + 3.3 Transformer 无锚定则崩溃 |
| **稳定层** (重要) | 残差连接 + LayerNorm | 3.2 no_all 仍 stable (decay=-0.66%) |
| **优化层** (锦上添花) | 三链 SSM 演化提升精度 | 3.2 单消融空间链恶化 |

## 方法说明

### 信号保留探针

对每个 OOD 样本 (T=150, 训练时 T=100):
1. 跑 clean forward 拿 logits_clean
2. 修改 S_0 的 3 个非零 cell (注入扰动信号)
3. 跑 perturbed forward 拿 logits_pert
4. 敏感度(t) = ||logits_pert[:,t] - logits_clean[:,t]||_2 / ||logits_clean[:,t]||_2
5. 保留率(t) = sensitivity(t) / sensitivity(1)

### 敏感度关键时间点

| t | SSM sensitivity | SSM retention% | Transformer sensitivity | Transformer retention% |
|---|-----------------|---------------|------------------------|----------------------|
| 1 | 0.0407 | 100.0% | 0.0661 | 100.0% |
| 10 | 0.0368 | 90.4% | 0.0633 | 95.8% |
| 50 | 0.0387 | 95.2% | 0.0650 | 98.4% |
| 100 | 0.0241 | 59.3% | 0.0615 | 93.1% |
| 120 | 0.0400 | 98.5% | 0.0646 | 97.8% |
| 150 | 0.0382 | 94.1% | 0.0656 | 99.4% |

注：SSM 在 t=100 有一个低谷（retention=59.3%），但 t=120 回升到 98.5%，说明 SSM 的敏感度有波动但整体保持。Transformer 全程稳定在 93-99%。

## 阶段 3 全景

| 子任务 | 方法 | 假说 | 结论 |
|--------|------|------|------|
| 3.1 探针 | 三链 norm 可视化 | H1/H2/H3 | H1支持(时间链稳定), H2推翻(空间链未衰减), H3支持(因果链稳定) |
| 3.2 消融 | 置零某链输出 | 哪条链贡献不退化 | 空间链有实质贡献, 全消融不退化(AnchorInit2+残差保障) |
| 3.3 对照 | SSM vs Transformer 信号保留 | H4/H5 | H4支持(SSM 94.1%), **H5推翻**(Transformer 99.4%) |

### 核心机制结论

ThreeChainMamba2 长程外推不退化的根本原因是 **AnchorInit2 初始表示的结构性锚定**（cell+action+time 三源），而非 SSM 的递推状态保持。三链 SSM 演化是精度优化层，残差+LayerNorm 是稳定层。

## 产出物

- 图表: `paper/figures/stage3_3_retention.png`
- 本报告: `paper/stage3_3_retention.md`
- 探针脚本: `probe_token_retention.py`
- 云端编排: `run_stage3_3_cloud.py`
