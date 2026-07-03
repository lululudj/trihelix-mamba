# ThreeChainMamba2 文档目录索引

本目录 (`docs/`) 是 ThreeChainMamba2 项目的 Wiki 技术文档中心, 面向沐曦青年开源专项基金评审与研究生套磁材料, 体现科研深度。

> 🌳 项目主页: [../README.md](../README.md) | 论文 LaTeX: [../paper/main.tex](../paper/main.tex) | 实验数据: [../paper/data_summary.md](../paper/data_summary.md)

## 文档清单

| 文档 | 内容 | 主要读者 |
|---|---|---|
| **[Home.md](Home.md)** | Wiki 首页, 项目简介 + 文档索引 + 快速开始 + 沐曦基金申报信息 | 所有访问者 (入口) |
| **[architecture.md](architecture.md)** | 三链架构详解 + 完整 LaTeX 数学方程 + Mamba2 (SSD) 核心方程 + 三链参数表 + 因果性分析 | 评审 / 研究生 (技术深度) |
| **[theory.md](theory.md)** | 三链不退化理论推导: 5 假说验证 + 三层保障机制 + GridWorld vs SDD 差异理论 + window-smoothed decay 理论 | 研究生 / 评审 (科研深度) |
| **[mxmaca_adaptation.md](mxmaca_adaptation.md)** | MXMACA 国产 GPU 适配方案: 软件栈架构 + 零修改适配证明 + 部署步骤 + 端侧推理 benchmark 计划 + 落地场景 | 沐曦基金评审 (国产化) |

## 推荐阅读顺序

1. **[Home.md](Home.md)** — 项目概览 (5 分钟)
2. **[architecture.md](architecture.md)** — 架构与数学方程 (20 分钟)
3. **[theory.md](theory.md)** — 理论推导与机制分析 (30 分钟)
4. **[mxmaca_adaptation.md](mxmaca_adaptation.md)** — 国产 GPU 适配方案 (15 分钟, 沐曦基金评审重点)

## 上游资源

- 项目 README: [../README.md](../README.md)
- 复现指南: [../REPRODUCE.md](../REPRODUCE.md)
- 研究路线图: [../RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md)
- 实现代码: [../models/three_chain_mamba2.py](../models/three_chain_mamba2.py)
- 共享组件: [../models/common.py](../models/common.py)
- 训练入口: [../train.py](../train.py)
- OOD 评估: [../eval_ood.py](../eval_ood.py)

## 论文与报告

- 完整技术报告: [../paper/technical_report.md](../paper/technical_report.md)
- 机制分析: [../paper/mechanism_analysis.md](../paper/mechanism_analysis.md)
- 消融实验报告: [../paper/stage3_ablation.md](../paper/stage3_ablation.md)
- 信号保留对照: [../paper/stage3_3_retention.md](../paper/stage3_3_retention.md)
- 三链 norm 探针: [../paper/stage3_3_1_validation.md](../paper/stage3_3_1_validation.md)
- SDD 真实数据报告: [../paper/stage2_sdd_report.md](../paper/stage2_sdd_report.md)
- 实验数据汇总: [../paper/data_summary.md](../paper/data_summary.md)
- 论文 LaTeX 源: [../paper/main.tex](../paper/main.tex)

## 文档约定

- **数学方程**: LaTeX 语法, 行内 `$...$`, 独立行 `$$...$$` (GitLink Wiki 与 GitHub 都支持)
- **语言**: 中文为主, 技术术语保留英文 (如 SSM, SSD, Mamba2, MXMACA)
- **代码引用**: 相对路径链接, 如 `[three_chain_mamba2.py](../models/three_chain_mamba2.py)`
- **每个文档开头**: 含目录 (TOC) 便于导航
- **可追溯性**: 所有数字可追溯到 `results_cloud/` / `results_stage2/` 原始 JSON 文件

## 许可证

MIT — 见 [../LICENSE](../LICENSE)。

> 本目录文档面向沐曦青年开源专项基金申请开源, 便于评审复现与国产 GPU 适配验证。
