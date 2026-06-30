import requests, json, sys
sys.stdout.reconfigure(encoding="utf-8")

prompt = """你是Mamba/SSM底层专家。我有一个三流架构，每条流需要定制SSM核心超参数:

三条流的输入特性:
- Stream_S(空间): 8x8=64个位置的2D网格展平，需要捕捉邻接关系(上下左右)。序列长度64，但信息密度高——每个位置都有关联。
- Stream_C(因果): 4-8个agent的动作序列，agent间有交互。序列长度短(K=4~8)，但每个token含义重。
- Stream_T(时间): 100步的时间序列，需要同时捕捉短期节奏和长期趋势。标准1D时序。

Mamba四个核心超参数:
d_state: 状态空间维度(默认16)——越大记忆越强但越慢
d_conv: 1D卷积核大小(默认4)——控制局部感受野
expand: 通道扩展比(默认2)——控制特征丰富度
dt_rank: delta投影秩(默认=d_model/16)

请针对每条流给出精确推荐值，并解释原因。格式:
Stream_S: d_state=X, d_conv=X, expand=X, dt_rank=X
原因: ...
Stream_C: ...
原因: ...
Stream_T: ...
原因: ...

另外: 三流都用相同的d_model=256，这样做参数异构但输出维度统一，合理吗？

简短直接。"""

resp = requests.post(
    'https://api.deepseek.com/v1/chat/completions',
    headers={'Authorization': 'Bearer sk-559779dee4d04a049f5d2c46a2192ae9', 'Content-Type': 'application/json'},
    json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.7, 'max_tokens': 2000},
    timeout=60
)
r = resp.json()
print(r['choices'][0]['message']['content'])
print(f"\n--- tokens: {r.get('usage',{}).get('total_tokens','?')} ---")
