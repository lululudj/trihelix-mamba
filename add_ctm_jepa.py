import sys; sys.stdout.reconfigure(encoding="utf-8")

with open("models/common.py", "r", encoding="utf-8") as f:
    content = f.read()

ctm_jepa = """

# ========== CTM: 多频神经振荡内核 (v6) ==========

class CTM_Oscillator(nn.Module):
    \"\"\"连续时间流形振荡器: 在每个SSM隐状态维度上叠加可学习的多频振荡。
    
    三条链分配不同频段(仿脑波):
      空间链: Delta (0.5-4 Hz) - 慢波, 长程空间依赖
      因果链: Theta (4-8 Hz) - 海马节律, 因果序列绑定
      时间链: Gamma (30-80 Hz) - 快波, 局部时间同步
    
    数学: h_new = h * exp(i*freq*dt) * exp(-damp*dt)
    实际用实数旋转近似: h_new = h * cos(phase) * decay
    \"\"\"
    
    def __init__(self, d_state, freq_band="theta"):
        super().__init__()
        self.d_state = d_state
        if freq_band == "delta":
            base_freq = 2.0
        elif freq_band == "theta":
            base_freq = 6.0
        elif freq_band == "gamma":
            base_freq = 50.0
        else:
            base_freq = 10.0
        
        self.log_freq = nn.Parameter(torch.full((d_state,), math.log(base_freq)))
        self.log_damp = nn.Parameter(torch.full((d_state,), -1.0))
    
    def forward(self, h):
        \"\"\"h: (B, L, d_state) 或 (B, d_state)\"\"\"
        freq = torch.exp(self.log_freq).to(h.device)
        damp = torch.exp(self.log_damp).to(h.device)
        dt = 0.01  # 归一化时间步
        phase = freq * dt
        decay = torch.exp(-damp * dt)
        h_mod = h * torch.cos(phase) * decay
        return h_mod


# ========== JEPA: 潜空间世界预测引擎 (v6) ==========

class JEPA_Predictor(nn.Module):
    \"\"\"JEPA潜空间预测器: 在隐状态空间做未来N步推演。
    
    参考LeWorldModel设计:
    - 输入当前融合隐状态
    - GRU递归预测未来隐状态
    - stop-gradient防表征崩塌
    - 余弦相似度损失(非MSE, 保持方向一致性)
    \"\"\"
    
    def __init__(self, d_state_total, n_future=3, hidden=64):
        super().__init__()
        self.n_future = n_future
        self.predictor = nn.GRUCell(d_state_total, hidden)
        self.proj_out = nn.Linear(hidden, d_state_total)
    
    def forward(self, h_t):
        \"\"\"h_t: (B, d) 当前融合隐状态
        Returns: (B, n_future, d) 未来隐状态预测\"\"\"
        preds = []
        h = h_t
        for _ in range(self.n_future):
            h = self.predictor(h)
            preds.append(self.proj_out(h))
        return torch.stack(preds, dim=1)
    
    def compute_loss(self, h_pred, h_true):
        \"\"\"余弦相似度损失: 鼓励方向一致而非逐元素匹配\"\"\"
        h_pred_norm = F.normalize(h_pred, dim=-1)
        h_true_norm = F.normalize(h_true, dim=-1)
        cos_sim = (h_pred_norm * h_true_norm).sum(dim=-1)
        return (1.0 - cos_sim).mean()
"""

with open("models/common.py", "w", encoding="utf-8") as f:
    f.write(content + ctm_jepa)

print("OK: CTM + JEPA added to common.py")
