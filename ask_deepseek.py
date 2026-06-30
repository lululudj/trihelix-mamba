import json, urllib.request

API_KEY = "sk-559779dee4d04a049f5d2c46a2192ae9"
API_URL = "https://api.deepseek.com/v1/chat/completions"

report = open("/mnt/e/three_chain_v3/results_wsl/FINAL_BENCHMARK_REPORT.md", "r", encoding="utf-8").read()

prompt = f"""你是一位顶会审稿人（NeurIPS/ICLR级别）。以下是一个三螺旋DNA-Mamba架构的基准测试报告。请从专业角度给出尖锐、诚实的评价：

1. 这个实验结果在学术上有什么亮点和不足？
2. OOD稳定性零方差是否足够惊艳？有没有可能是实验设计缺陷导致的假象？
3. 与Single Mamba和Transformer对比，Tri-Helix的真正优势在哪里？
4. 如果要把这个工作投NeurIPS，还需要补什么实验？
5. 给一个1-10分的评分。

=== 报告内容 ===
{report}
=== 报告结束 ===
"""

data = json.dumps({
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7,
    "max_tokens": 3000
}).encode("utf-8")

req = urllib.request.Request(API_URL, data=data, headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
})

resp = urllib.request.urlopen(req, timeout=120)
result = json.loads(resp.read().decode("utf-8"))
print(result["choices"][0]["message"]["content"])
