import requests, json, sys
sys.stdout.reconfigure(encoding="utf-8")

prompt = """你是SSM/Mamba架构专家。我有一个三流并行Mamba架构，每条流处理不同类型的数据:

Stream_S(空间): 输入(B, N^2, d)，2D网格展平为1D序列，需要理解空间邻接关系
Stream_C(因果): 输入(B, K, d)，K个agent的动作嵌入，agent之间会交互
Stream_T(时间): 输入(B, T, d)，标准时序序列

当前三条流都用相同的vanilla Mamba(d_model=256, d_state=16, d_conv=4, expand=2)。
问题: 三流架构性能不如单流Mamba。

我认为vanilla Mamba不适合所有三条流，需要为每条流定制Mamba变体:
- 空间流: 需要2D感知（如Vision Mamba的双向扫描，或row+column交叉扫描）
- 因果流: 需要跨agent交互感知（如agent间门控）
- 时间流: 可能需要多尺度时间感受野

请直接给出:
1. 每条流最适合的Mamba变体是什么？参数量会增多少？
2. 有没有论文支持这种"异构Mamba"思路？
3. 给出最简实现方案（保持显存可控，RTX 4060 8G）

简短回答，不要客套。"""

resp = requests.post(
    'https://api.deepseek.com/v1/chat/completions',
    headers={'Authorization': 'Bearer sk-559779dee4d04a049f5d2c46a2192ae9', 'Content-Type': 'application/json'},
    json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.7, 'max_tokens': 2000},
    timeout=60
)
print(resp.json()['choices'][0]['message']['content'])
