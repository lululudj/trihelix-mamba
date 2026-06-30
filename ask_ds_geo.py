import requests, json, sys
sys.stdout.reconfigure(encoding="utf-8")

prompt = """你是非线性动力学+深度学习专家。我有一个三流Mamba架构(空间/因果/时间三链),每条链输出(B, L, d)隐状态。

当前我已实现:
1. 三链异构Mamba核心(不同d_state/d_conv/expand)
2. 碱基对耦合(Cross-Delta调制+Reset Gate+FiLM,无Attention)
3. 三链正交性已达<0.04(近乎完美解耦)
4. changed_acc从0.380提升到0.406

现在考虑加几何正则化损失:强制三链在相空间形成"旋转等边三角形+圆柱面包络":
- 等边约束: 三链两两余弦相似度趋近-0.5(120度相位差)
- 圆柱包络: 三链L2范数之和在时间维保持恒定
- 旋转平滑: 相邻时间步三角形旋转角度变化率最小化

请直接判断:
1. 这个几何正则化有理论价值吗?与现有的正交性约束(已达<0.04)是否冲突?
2. 120度相位差(-0.5余弦)是否合理?还是保持当前的正交(~0)更好?
3. 圆柱包络约束会不会和任务loss冲突,导致模型为了"好看"牺牲精度?
4. 如果值得做,给最简实现(别太重)

简短回答。"""

resp = requests.post(
    'https://api.deepseek.com/v1/chat/completions',
    headers={'Authorization': 'Bearer sk-559779dee4d04a049f5d2c46a2192ae9', 'Content-Type': 'application/json'},
    json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.7, 'max_tokens': 1500},
    timeout=60
)
print(resp.json()['choices'][0]['message']['content'])
