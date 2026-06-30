import json, urllib.request

API_KEY = "sk-559779dee4d04a049f5d2c46a2192ae9"
API_URL = "https://api.deepseek.com/v1/chat/completions"

prompt = """你刚才给我的Tri-Helix DNA-Mamba测评报告打了2分，指出了5个致命问题。现在我需要你给出一份具体的整改实施方案。

硬件约束：
- RTX 4060 Laptop 8GB VRAM
- WSL2 Ubuntu
- 单卡，不能分布式训练
- 每轮训练约7分钟（2000步），不能跑太久
- 模型：Tri-Helix (三链Mamba) 4.77M, Single Mamba, Transformer

需要你给出：
1. 参数对齐方案：如何让三个模型的参数量精确匹配到同一水平（建议锁定多少参数？哪些超参调？）
2. 最小可行的实验矩阵：在8GB显存约束下，多少个种子、多少步、几个OOD难度能既让审稿人闭嘴又不跑一个月？
3. 消融实验怎么设计才有说服力？（不是简单的concat vs single）
4. OOD任务怎么设计才能排除"恒等映射假象"？
5. 如果我最多只能跑50轮实验（每轮~10分钟），最优实验分配是什么？

请给出可以直接执行的方案，每一条都要具体到数字。不要说"更多种子"，要说"15个种子"。不要说"更多数据集"，要说"在X和Y数据集上"。"""
data = json.dumps({
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.3,
    "max_tokens": 3000
}).encode("utf-8")

req = urllib.request.Request(API_URL, data=data, headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
})
resp = urllib.request.urlopen(req, timeout=120)
result = json.loads(resp.read().decode("utf-8"))
print(result["choices"][0]["message"]["content"])
