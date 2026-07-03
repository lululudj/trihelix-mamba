# 爱丽丝 (Alice) — 端侧 Mamba 对话助手

## 项目概述

爱丽丝是一个运行在端侧（小米手机）的对话助手，基于 Mamba 架构实现高效推理。

## 当前状态

- [x] 设计文档完成（2 处修改已应用：改名爱丽丝、改小米真机）
- [ ] 代码实现（零代码，待实现）
- [ ] 模型移植/部署
- [ ] 测试验证

## 设计文档

设计文档位于: `docs/superpowers/specs/2026-06-30-mamba-assistant-design.md`

### 已应用修改
1. **改名爱丽丝**: 项目中文名定为"爱丽丝"
2. **目标设备**: 小米真机

## 关键决策

待补充 — 需从设计文档中提取。

## 下一步

1. 阅读设计文档，确认所有设计决策
2. 启动实现计划 (writing-plans)
3. 分阶段编码实现

## Git 状态

```
470cb9c (HEAD -> main, origin/master, origin/main) Add ablation (BPv2 + Transformer-tiny), bilingual README, reproducible data
cec8af1 Initial release: TriHelix-Mamba2 (3-chain DNA-Mamba2)
```

当前仓库基于 `three_chain_v3`，爱丽丝项目在此基础上启动。

## 启动语

新会话启动时说:
> 我在做爱丽丝项目，设计文档在 docs/superpowers/specs/2026-06-30-mamba-assistant-design.md，先读设计文档，然后启动 writing-plans
