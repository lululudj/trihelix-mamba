# 论文发表任务 — 项目交接

> 时间：2026-07-01
> 分支：paper-submission（基于 main）
> 任务：把三螺旋 DNA-Mamba2 研究成果写成论文发表
> 状态：素材齐备，待动笔

---

## 一句话任务

把已有的三螺旋 DNA-Mamba2 实验结果写成论文，先 arXiv 预印本占坑，再投 NeurIPS Workshop，数据扎实后冲 ICLR 正会。

---

## 发表计划（3 级目标）

| 级别 | 目标 | 篇幅 | 时间窗 | 难度 |
|---|---|---|---|---|
| 🥉 立即 | arXiv 预印本占坑 | 8-12 页 | 当天上线 | 低 |
| 🥈 近期 | NeurIPS Workshop 投稿 | 4-6 页 | 看 Workshop 截止 | 中 |
| 🥇 远期 | ICLR 正会 | 8-10 页 | 下个投稿窗 | 高 |

**推荐顺序**：先 arXiv（占坑+建立发表记录）→ 再 NeurIPS Workshop（命中率高的故事性投稿）→ 数据补齐后 ICLR。

---

## 论文核心论点（已迭代 3 版，定稿）

### 论点演化
1. **初版**："3链SSM+碱基对约束 治好退化"
2. **二版**："简单统一张量架构 > 复杂配对约束"（奥卡姆剃刀，BP 实验后转向）
3. **定版**（强化）："三链 SSM 长程外推 > Attention；约束加错地方有害，加对地方中性，简单架构最优"

### 三大杀器证据
1. **主结果**：三螺旋 mamba2 5 seed 2000 步，OOD changed_acc@150 = 0.5952±0.0034，OOD decay ≈ -0.0042（几乎不衰减）
2. **消融对照**：
   - BPv1（单链内约束）：OOD -5.6%，严重衰减 → 约束加错地方有害
   - BPv2（跨链耦合）：OOD 持平，alpha 学到负值 → 加对地方中性，模型用碱基对维持多样性而非共识
3. **杀手级对照**：Transformer-tiny 同参数（3.43M vs 3.26M）同训练同数据，OOD zero_ratio=1.000（100% 全猜 0），完全退化 → SSM 长程外推 > Attention

---

## 已有素材（全部齐备，直接用）

### 代码与数据
- GitHub 仓库：https://github.com/lululudj/trihelix-mamba（commit 470cb9c）
- 可复现 OOD 测试集：`data/ood_T150/`（72 样本）
- 全部 JSON 指标：`results_wsl/`
- 核心模型代码：`models/three_chain_mamba2.py`（A 版主力）

### 文档
- [PROFESSIONAL_REPORT.md](file:///e:/three_chain_v3/PROFESSIONAL_REPORT.md) — 10 章专业测评报告（可直接当论文素材）
- [REPRODUCE.md](file:///e:/three_chain_v3/REPRODUCE.md) — 复现指南
- [MAMBA2_STATUS.md](file:///e:/three_chain_v3/MAMBA2_STATUS.md) — 状态跟踪

### 图表（7 张，宇宙深色风，ASCII 安全字符）
- `figures/fig1_training_dynamics.png` — 训练动态
- `figures/fig2_ood_curve.png` — OOD 长程曲线（主图）
- `figures/fig3_ood_decay.png` — OOD 衰减对比
- `figures/fig4_seed_variance.png` — 5 seed 方差
- `figures/fig5_old_vs_new.png` — 旧版 vs 新版对照
- `figures/fig6_brain_probe.png` — 退化诊断
- `figures/fig7_ablation.png` — 消融实验（BPv1/BPv2/Transformer vs baseline）

---

## 关键数据速查表

| 模型 | 参数 | 训练 ch_acc@100 | OOD ch_acc@150 | OOD decay | 退化? |
|---|---|---|---|---|---|
| ThreeChainMamba2 (baseline) | 3.26M | 0.5991 | 0.5989 | -0.0042 | ❌ 不退化 |
| ThreeChainMamba2BPv1 | 3.39M | 0.5994 | 0.5430 | -9.4% | ⚠️ 衰减 |
| ThreeChainMamba2BPv2 | 3.40M | 0.5906 | 0.5991 | ≈0 | ❌ 不退化（持平） |
| Transformer-tiny | 3.43M | 0.04~0.58 乱跳 | zero_ratio=1.0 | N/A | ❌ 完全退化 |
| 旧 ThreeChain (Mamba1) | 4.77M | — | std=0.0000 | — | ❌ 评估 bug 假象 |

5 seed baseline 详细：OOD changed_acc@150 = 0.5952 ± 0.0034（seed: 42/123/456/789/1024）

---

## 论文结构建议（arXiv 版，10 页）

```
1. Abstract — 三链 SSM 治退化 + 杀手级 Transformer 对照
2. Introduction — 长程外推痛点 + 三链直觉 + 贡献清单
3. Related Work — Mamba/SSM、世界模型、GridWorld benchmark
4. Method
   4.1 三链架构（时间/空间/因果 + 统一时空张量）
   4.2 HeteroMamba2 三链残差融合
   4.3 （消融）碱基对约束 v1/v2
5. Experiments
   5.1 Setup（GridWorld N=8, K=8, T=100 train / 150 OOD）
   5.2 主结果（5 seed + OOD decay）
   5.3 消融（BPv1/BPv2）
   5.4 杀手级对照（Transformer-tiny 完全退化）
   5.5 退化诊断（zero_ratio/changed_acc 分析）
6. Discussion — 为什么 SSM 赢 Attention、碱基对中性原因
7. Limitations — 棋盘规模、对话能力、单任务
8. Conclusion
附录: 复现指南、超参表、5 seed 完整指标
```

---

## NeurIPS Workshop 故事包装（4-6 页版）

**故事钩子**："为什么同参数的 Transformer 在长程外推上完全退化，而三链 SSM 不会？"

**压缩结构**：
- 1 页 Intro + 三链直觉
- 1.5 页 Method（只讲 baseline 架构，消融简述）
- 1.5 页 Experiments（主结果 + Transformer 对照，消融放附录）
- 1 页 Discussion + Conclusion

**Workshop 命中率高的原因**：SSM vs Attention 的对照实验是当下热点，"简单架构赢复杂约束"的反直觉结论有传播性。

---

## 写作工具建议

- **arXiv 预印本**：LaTeX，用 `arxiv.sty` 模板
- **NeurIPS Workshop**：官方 `neurips_2026.sty`（4-6 页严限）
- **图表**：现有 7 张 PNG 可直接用，必要时重制 PDF 矢量版
- **引用管理**：Zotero 或 BibTeX，SSM/Mamba 论文必引（Gu+2023, Albert+2023, Mamba2 2024）

---

## Git 工作流

```bash
# 当前分支
git branch  # 应显示 paper-submission

# 论文写作在这个分支进行，不影响主线代码
# 写作产物建议放：
#   paper/           # LaTeX 源码
#   paper/figures/   # 论文用图（可软链 figures/）
#   paper/sections/  # 分节 .tex 文件

# 推送分支到 GitHub（首次）
git push -u origin paper-submission
```

**注意**：从 Windows PowerShell 推送需配代理
```
git config http.proxy http://127.0.0.1:7892
```

---

## 风险与注意事项

| 风险 | 缓解 |
|---|---|
| arXiv 审核被拒（格式/内容） | 严格按 arXiv 模板，先小修后投 |
| NeurIPS Workshop 截止已过 | 查 2026 各 Workshop 截止日，错过则等下个或转 ICLR |
| Transformer 对照被审稿质疑"不公平" | 强调同参数同训练同数据，只换架构，最严格对照 |
| 旧 ThreeChain std=0 评估 bug | 论文里如实报告，作为"暴露评估陷阱"的贡献，不掩饰 |
| 论点"简单架构最优"被反驳 | 用 BPv1/BPv2 双消融支撑，不是单点结论 |

---

## 给新会话的启动提示词

> "我在 three_chain_v3 项目做论文发表任务，分支 paper-submission。先读 PAPER_HANDOFF.md 了解全貌，再读 PROFESSIONAL_REPORT.md 当素材。第一步：搭 arXiv LaTeX 骨架，按 PAPER_HANDOFF.md 里的 10 页结构列大纲。"

---

## 与主线任务的关系

- **主线（master/main 分支）**：三螺旋架构扩展（外置 transformer + .m3 格式 + JEPA 移植），进行中
- **本任务（paper-submission 分支）**：论文发表，独立进行
- **交叉点**：如果主线扩展跑出新结果（比如 JEPA 移植提升 OOD），可补充进论文再投 ICLR；arXiv 版不阻塞主线

两个任务可并行，互不阻塞。论文用现有结果先占坑，主线扩展跑出新结果后再升级版本。
