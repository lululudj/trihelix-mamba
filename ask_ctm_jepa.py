import requests, json, sys
sys.stdout.reconfigure(encoding="utf-8")

prompt = """你是SSM/神经动力学/JEPA专家。我有一个三流异构Mamba架构(空间/因果/时间三链,各有不同d_state/d_conv/expand,已实现碱基对耦合)。

现在要融合两个组件到核心大脑层:

1. CTM(Continuous Time Manifold)神经动力学:
- 多频段神经振荡替代人工硬隔离
- 连续态演化(非离散跳变)
- 内生自校验(多频共振偏差检测)

2. JEPA(Joint Embedding Prediction Architecture):
- 在潜空间做状态推演而非Token空间
- 预测未来N步世界状态快照
- 反事实推理(改一个变量推算连锁后果)
- 参考LeWorldModel: 1500万参数、单卡几小时训完

约束:
- RTX 4060 8G显存
- 三链Mamba参数4.69M,不能再翻倍
- 必须保持O(N)线性复杂度

请直接给出:
1. CTM最简实现:如何在三链中嵌入多频振荡?频段如何分配?
2. JEPA最简实现:潜空间预测头怎么设计?和现有强制分工辅助头什么关系?
3. CTM+JEPA如何协同?谁提供训练信号给谁?
4. 参数量增幅预估

简短回答,给代码骨架。"""

resp = requests.post(
    'https://api.deepseek.com/v1/chat/completions',
    headers={'Authorization': 'Bearer sk-559779dee4d04a049f5d2c46a2192ae9', 'Content-Type': 'application/json'},
    json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.7, 'max_tokens': 2000},
    timeout=60
)
print(resp.json()['choices'][0]['message']['content'])
