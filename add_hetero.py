import sys; sys.stdout.reconfigure(encoding="utf-8")

with open("models/common.py", "r", encoding="utf-8") as f:
    content = f.read()

hetero_mamba = """

# ========== HeteroMamba: 三链异构核心 (v4) ==========

class HeteroMamba(nn.Module):
    \"\"\"三链异构Mamba: 每条链独立的核心超参数+外围扫描策略
    
    空间链: d_state=64 d_conv=7 expand=1 + 双向交叉扫描
    因果链: d_state=8  d_conv=3 expand=4 + Agent间门控
    时间链: d_state=32 d_conv=5 expand=2 + 多尺度膨胀卷积
    \"\"\"
    
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        # === 空间链: 大状态+宽卷积+双向扫描 ===
        self.mamba_s_row = Mamba(d_model=d_model, d_state=64, d_conv=7, expand=1)
        self.mamba_s_col = Mamba(d_model=d_model, d_state=64, d_conv=7, expand=1)
        self.fusion_s = nn.Linear(d_model * 2, d_model)
        self.norm_s = nn.LayerNorm(d_model)
        
        # === 因果链: 小状态+高扩展+Agent门控 ===
        self.mamba_c = Mamba(d_model=d_model, d_state=8, d_conv=3, expand=4)
        self.agent_gate = nn.Linear(d_model * 2, d_model)
        self.norm_c = nn.LayerNorm(d_model)
        
        # === 时间链: 中状态+中卷积+多尺度 ===
        self.mamba_t = Mamba(d_model=d_model, d_state=32, d_conv=5, expand=2)
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 4]
        ])
        self.fusion_t = nn.Linear(d_model * 3, d_model)
        self.norm_t = nn.LayerNorm(d_model)
    
    def forward(self, h_s, h_c, h_t):
        \"\"\"
        h_s: (B, N^2, d)  空间序列
        h_c: (B, K, d)    因果序列
        h_t: (B, T, d)    时间序列
        Returns: (h_s', h_c', h_t') 形状不变
        \"\"\"
        B = h_s.shape[0]
        N2 = h_s.shape[1]
        N = int(N2 ** 0.5)
        D = self.d_model
        
        # --- 空间链: 双向交叉扫描 ---
        # 行优先扫描
        row_out = self.mamba_s_row(h_s)  # (B, N^2, d)
        # 列优先扫描: 转置2D网格再展平
        h_s_2d = h_s.view(B, N, N, D)  # (B, N, N, d)
        col_seq = h_s_2d.permute(0, 2, 1, 3).reshape(B, N2, D)  # 列优先展平
        col_out = self.mamba_s_col(col_seq)
        # 融合行+列
        h_s_out = self.fusion_s(torch.cat([row_out, col_out], dim=-1))
        h_s_out = self.norm_s(h_s + h_s_out)  # 残差
        
        # --- 因果链: Agent间门控 ---
        c_out = self.mamba_c(h_c)  # (B, K, d)
        # 其他agent均值(排除自身)
        agent_sum = c_out.sum(dim=1, keepdim=True)  # (B, 1, d)
        K = h_c.shape[1]
        agent_mean = (agent_sum - c_out) / max(1, K - 1)  # (B, K, d)
        # 门控融合
        gate = torch.sigmoid(self.agent_gate(torch.cat([c_out, agent_mean], dim=-1)))
        h_c_out = self.norm_c(h_c + gate * c_out + (1 - gate) * agent_mean)
        
        # --- 时间链: 多尺度膨胀卷积 ---
        t_out = self.mamba_t(h_t)  # (B, T, d)
        t_conv = t_out.permute(0, 2, 1)  # (B, d, T)
        multi_scale = [conv(t_conv) for conv in self.dilated_convs]  # 3 x (B, d, T)
        t_fused = torch.cat(multi_scale, dim=1).permute(0, 2, 1)  # (B, T, 3d)
        h_t_out = self.fusion_t(t_fused)  # (B, T, d)
        h_t_out = self.norm_t(h_t + h_t_out)  # 残差
        
        return h_s_out, h_c_out, h_t_out
"""

with open("models/common.py", "w", encoding="utf-8") as f:
    f.write(content + hetero_mamba)

print("OK: HeteroMamba added to common.py")
