import sys; sys.stdout.reconfigure(encoding="utf-8")

with open("models/common.py", "r", encoding="utf-8") as f:
    content = f.read()

base_pair_module = """

# ========== TriHelixBasePairCoupling: 三螺旋碱基对耦合 (v5) ==========

class TriHelixBasePairCoupling(nn.Module):
    \"\"\"三螺旋碱基对两两耦合模块。
    
    三条链(空间/因果/时间)通过局部参数调制实现相位同步,
    不使用任何Attention, 保持O(L)线性复杂度。
    
    三对碱基对:
    1. 空间-时间: Cross-Delta调制 (频率锚定)
    2. 时间-因果: Reset Gate (记忆重置)  
    3. 因果-空间: FiLM仿射变换 (逻辑具身)
    \"\"\"
    
    def __init__(self, d_model=256):
        super().__init__()
        D = d_model
        
        # === 碱基对1: 空间 <-> 时间 (Cross-Delta) ===
        # 时间->空间: h_time生成dt调制系数
        self.st_t2s_delta = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, 1),
            nn.Sigmoid()
        )
        # 空间->时间: 空间变化率门控时间更新
        self.st_s2t_gate = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
            nn.Sigmoid()
        )
        
        # === 碱基对2: 时间 <-> 因果 (Memory Reset) ===
        # 因果->时间: 因果能量生成Reset Gate
        self.tc_c2t_reset = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
            nn.Sigmoid()
        )
        # 时间->因果: 时间相位拼接作为位置先验
        self.tc_t2c_phase = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
        )
        
        # === 碱基对3: 因果 <-> 空间 (FiLM) ===
        # 因果->空间: 生成gamma和beta做仿射变换
        self.cs_c2s_gamma = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
        )
        self.cs_c2s_beta = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
        )
        # 空间->因果: 空间坐标作为现实校验器
        self.cs_s2c_check = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, D),
            nn.Tanh()
        )
        
        # 归一化层
        self.norm_s = nn.LayerNorm(D)
        self.norm_c = nn.LayerNorm(D)
        self.norm_t = nn.LayerNorm(D)
    
    def forward(self, h_s, h_c, h_t):
        \"\"\"
        h_s: (B, L_s, d) 空间序列
        h_c: (B, L_c, d) 因果序列  
        h_t: (B, L_t, d) 时间序列
        Returns: (h_s', h_c', h_t') 同形状
        \"\"\"
        # 池化到全局表示 (B, d)
        g_s = h_s.mean(dim=1)  # 空间全局
        g_c = h_c.mean(dim=1)  # 因果全局
        g_t = h_t.mean(dim=1)  # 时间全局
        
        # === 碱基对1: 空间 <-> 时间 ===
        # 时间调制空间(Cross-Delta): delta_mod = 0.5 + 0.5*sigmoid
        delta_mod = 0.5 + 0.5 * self.st_t2s_delta(g_t)  # (B, 1)
        # 空间变化率门控时间
        s_diff = torch.diff(h_s, dim=1).mean(dim=1)  # (B, d) 空间一阶差分
        s_gate = self.st_s2t_gate(s_diff)  # (B, d)
        
        # === 碱基对2: 时间 <-> 因果 ===
        # 因果能量→Reset Gate (能量越高重置越多)
        c_energy = torch.norm(g_c, dim=-1, keepdim=True) / (g_c.shape[-1] ** 0.5)
        reset_gate = self.tc_c2t_reset(g_c) * c_energy.sigmoid()  # (B, d)
        # 时间相位→因果位置先验
        t_phase = self.tc_t2c_phase(g_t)  # (B, d)
        
        # === 碱基对3: 因果 <-> 空间 ===
        # 因果→FiLM参数
        gamma = self.cs_c2s_gamma(g_c)  # (B, d)
        beta = self.cs_c2s_beta(g_c)    # (B, d)
        # 空间→现实校验
        s_check = self.cs_s2c_check(g_s)  # (B, d)
        
        # === 应用耦合 ===
        # 空间链: delta调制 + FiLM仿射
        h_s_out = h_s * gamma.unsqueeze(1) + beta.unsqueeze(1)
        h_s_out = self.norm_s(h_s_out)
        
        # 因果链: 时间相位注入 + 现实校验抑制
        h_c_out = h_c + t_phase.unsqueeze(1) * 0.1  # 轻量注入
        h_c_out = h_c_out * (1.0 + 0.1 * s_check.unsqueeze(1))  # 空间校验
        h_c_out = self.norm_c(h_c_out)
        
        # 时间链: Reset门控 + 空间差分门控
        h_t_out = h_t * reset_gate.unsqueeze(1)  # 因果重置
        h_t_out = h_t_out * s_gate.unsqueeze(1)   # 空间变化门控
        h_t_out = self.norm_t(h_t_out)
        
        return h_s_out, h_c_out, h_t_out, {
            "delta_mod_mean": delta_mod.mean().item(),
            "reset_gate_mean": reset_gate.mean().item(),
            "s_gate_mean": s_gate.mean().item(),
        }
"""

with open("models/common.py", "w", encoding="utf-8") as f:
    f.write(content + base_pair_module)

print("OK: TriHelixBasePairCoupling added to common.py")
