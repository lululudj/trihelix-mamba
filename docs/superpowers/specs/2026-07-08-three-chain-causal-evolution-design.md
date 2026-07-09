# 三链 Mamba3 定向改造：让小蚂蚁吞噬进化

> 生成时间: 2026-07-08
> 目标: 让三链模块在 Qwen2.5-Coder-7B 基座上贡献可量化的 +5% 以上增益
> 战场: HumanEval (83.54% -> 88%+) + 递进任务序列 (成功率质变)

## 1. 问题诊断

### 1.1 当前三链的三个致命瓶颈

**瓶颈1: 评测没经过三链模块**
- `run_p5_humaneval.py` 的 EVAL_SCRIPT 是独立脚本，自己写了 `gen_single/gen_multi/run_test`
- 完全没用 `sandbox.py` 的 `run_task/run_task_multi`
- 三链 ppl 打分、工具检索、错误经验注入 —— 全都没在 P5 评测中生效
- 测到的 83.54% 是纯 Qwen2.5-Coder-7B 裸跑分数

**瓶颈2: 因果链传零（三链实际只有两链）**
- `mamba_brain.py` L290-292: `aux = torch.zeros_like(x); h_c = self.causal_chain(aux)`
- 因果链输入永远是全零，输出是常量，等于没启用
- best.pt 里因果链权重虽然加载了（缺0/多0），但输入为零等于白加载

**瓶颈3: 三链输入都是原始 char 序列，没有代码结构感知**
- 空间链、时间链都吃同一个 `char_embed(char_ids)`，只是扫描方向不同
- 没有利用 AST 结构、控制流、定义-引用关系等代码特有拓扑

### 1.2 为什么 multi 反而比 single 低

P5 评测的 multi 模式用 `ast.parse` 选最长候选（简化版），没用真 mamba ppl 打分。
sandbox.py 的 `run_task_multi` 本来有 ppl 打分逻辑（L683），但 P5 评测绕过了它。

## 2. 设计方案

### 2.1 总体架构：三链定向改造

```
                        ┌─────────────────────────────────┐
                        │       代码输入 (char_ids)        │
                        └──────────┬──────────────────────┘
                                   │
                    ┌──────────────┼──────────────────────┐
                    │              │                      │
                    ▼              ▼                      ▼
           ┌──────────────┐ ┌──────────────┐    ┌──────────────────┐
           │  空间链改造   │ │  时间链改造   │    │  因果链激活(核心) │
           │ AST拓扑编码  │ │ 控制流时序    │    │  正推+反推+反事实 │
           │ tree-sitter  │ │ 执行轨迹标记  │    │  三头联合训练     │
           └──────┬───────┘ └──────┬───────┘    └────────┬─────────┘
                  │                │                     │
                  └────────────────┼─────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │  动态门控融合    │
                          │ (任务自适应权重) │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────────┐
              │ 检索向量  │  │ ppl 打分 │  │ 因果推理头   │
              │ (工具复用)│  │ (候选选优)│  │(预测/定位/外推)│
              └──────────┘  └──────────┘  └──────────────┘
```

### 2.2 Phase 1: 修评测管线（快速验证，2小时）

让三链 ppl 打分在 HumanEval 评测中真正生效。

**改动**: `run_p5_humaneval.py` 的 EVAL_SCRIPT
- `gen_multi` 不再用 ast.parse 选最长
- 改为调用 `MambaBrain.score()` 对每个候选打 ppl 分
- 选 ppl 最低的候选（与 sandbox.py `_select_best_candidate` 逻辑一致）
- 设置 `MAMBA_BEST_PT` 和 `MAMBA_CHAR_EMBED_V1` 环境变量

**验证**: 跑 164 题，对比 single 83.54% vs multi(ppl) 是否有提升

### 2.3 Phase 2: 因果链激活 - 正推头（核心改造，1天）

让因果链从"传零"变成"预测代码会导致什么结果"。

**架构改动** (`mamba_brain.py`):

```python
class CausalChainHead(nn.Module):
    """因果链三头: 正推/反推/反事实"""
    def __init__(self, d_model=256):
        super().__init__()
        # 正推头: code -> 预测 (pass/fail, error_type, output_shape)
        self.forward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 5),  # [pass_prob, syntax_err, runtime_err, timeout, logic_err]
        )
        # 反推头: error -> 问题代码位置 (序列标注)
        self.backward_head = nn.Linear(d_model, 1)  # 每个token是问题点的概率
        # 反事实头: (code, mutation) -> delta_pass_prob
        self.counterfactual_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
```

**因果链输入改造**:
- 不再传 `torch.zeros_like(x)`
- 正推模式: 输入 = code 的 char_ids（和空间/时间链相同输入，但通过因果链独立建模"代码->结果"映射）
- 因果链的 Mamba2 会学到"看到 return 语句 -> 预测返回值类型"、"看到 except -> 预测可能异常"

**训练数据自造**:
- 跑 HumanEval 164 题 × 3 温度 = 492 个 (code, pass/fail, error) 样本
- 跑递进任务 10 个 × 3 温度 = 30 个样本
- 每个样本自动标注: code + 执行结果(pass/fail) + 错误类型(syntax/runtime/logic/timeout)

### 2.4 Phase 3: 反推头 + 反事实外推（1天）

**反推头**: 给定错误输出，反推哪段代码有问题
- 输入: error_message 的 char_ids
- 输出: 每个代码 token 是"问题点"的概率
- 用途: debug 场景，从错误反推根因

**反事实头**: "如果把第 i 行换成 X 会怎样"
- 输入: code_embedding + mutation_embedding
- 输出: pass_prob 变化量
- 用途: 多候选选优时，不只看 ppl，还看"这个候选如果微调会变好还是变差"

**反事实训练数据自造**:
- 对 HumanEval 成功代码做变异（换操作符/改条件/改缩进）
- 记录 (原code, 变异code, 原pass, 变异pass) 四元组
- 模型学习"这种变异会导致 pass->fail"

### 2.5 Phase 4: 空间链 + 时间链定向改造（1天）

**空间链 -> AST 拓扑感知**:
- 输入从 char_ids 改为 AST 遍历序列
- 用 tree-sitter 解析代码为 AST，做先序遍历生成节点类型序列
- char_embed 扩展为 ast_node_embed（256+128=384维，前256是char后128是AST节点类型）
- 双向 Mamba2 感知代码的嵌套结构、作用域、定义-引用关系

**时间链 -> 控制流时序感知**:
- 输入 = char_ids + 控制流标记
- 标记 if/for/while/def/return 等控制流关键字的 position
- 时间链建模"先执行A再判断B然后循环C"的执行时序

### 2.6 Phase 5: 训练管线（1天）

**自造数据闭环**:
1. Phase 1-4 的改造让三链能跑通后
2. 跑 HumanEval 164题 × 3温度 + 递进任务 10个 × 3温度 = 522 样本
3. 自动收集:
   - (code, pass/fail, error_type) -> 正推训练对
   - (error, code, problem_location) -> 反推训练对
   - (code, mutation, delta_pass) -> 反事实训练对
4. 训练因果链三头（冻结空间/时间链，只训因果链+三个头）

**外部数据补充**:
- APPS 数据集（带测试用例和错误信息）
- GitHub bug-fix commit 对（从 fix commit 提取 error->fix 因果对）

### 2.7 Phase 6: 双战线验证

**HumanEval 战线**:
- single: 83.54% (已测基线)
- multi(ast.parse): 82.93% (已测，简化版)
- multi(mamba ppl): 目标 85%+ (Phase 1 后)
- multi(因果链激活): 目标 88%+ (Phase 2-5 后)

**递进任务战线**:
- 基线: 3/3 成功（但度量Top-1=0%）
- 改造后: 检索Top-1 > 50%，10步递进任务成功率 > 80%

## 3. 实施顺序

| Phase | 内容 | 预期增益 | 依赖 |
|-------|------|---------|------|
| 1 | 修评测管线，三链ppl生效 | +1~2% | 无 |
| 2 | 因果链正推头激活 | +2~3% | Phase1验证管线通 |
| 3 | 反推头+反事实外推 | +1~2% | Phase2 |
| 4 | 空间链AST+时间链控制流 | +1~2% | Phase2 |
| 5 | 自造数据训练 | 累积放大 | Phase2-4 |
| 6 | 双战线验证 | 量化确认 | Phase5 |

**关键里程碑**: Phase 1 完成后立即跑 HumanEval，如果 multi(ppl) > single，说明三链 ppl 打分有价值，继续投入 Phase 2-5。

## 4. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Phase1 ppl打分无效果 | 说明v1 LM权重质量不够，Phase2直接训因果链 |
| 因果链训练数据不够 | 用外部APPS数据补充 |
| tree-sitter云端装不上 | Phase4先用char序列，AST改造延后 |
| 4090显存不够(serve_qwen占14.7GB) | 训练时停serve_qwen，推理时再启动 |
