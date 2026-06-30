import sys; sys.stdout.reconfigure(encoding="utf-8")

# Fix HeteroMamba to include BasePair coupling
with open("models/common.py", "r", encoding="utf-8") as f:
    c = f.read()

# Add import at top of HeteroMamba init
old_init = '''        # === 时间链: 中状态+中卷积+多尺度 ===
        self.mamba_t = Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2)  # d_conv max=4
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 4]
        ])
        self.fusion_t = nn.Linear(d_model * 3, d_model)
        self.norm_t = nn.LayerNorm(d_model)'''

new_init = '''        # === 时间链: 中状态+中卷积+多尺度 ===
        self.mamba_t = Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2)  # d_conv max=4
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 4]
        ])
        self.fusion_t = nn.Linear(d_model * 3, d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # === v5: 三螺旋碱基对耦合 ===
        self.basepair = TriHelixBasePairCoupling(d_model)'''

if old_init in c:
    c = c.replace(old_init, new_init)
    print("Added basepair to HeteroMamba __init__")
else:
    print("WARN: old_init not found")

# Fix forward to add basepair after time chain processing
old_fwd = '''        # --- 时间链: 多尺度膨胀卷积 ---
        t_out = self.mamba_t(h_t)  # (B, T, d)
        t_conv = t_out.permute(0, 2, 1)  # (B, d, T)
        multi_scale = [conv(t_conv) for conv in self.dilated_convs]  # 3 x (B, d, T)
        t_fused = torch.cat(multi_scale, dim=1).permute(0, 2, 1)  # (B, T, 3d)
        h_t_out = self.fusion_t(t_fused)  # (B, T, d)
        h_t_out = self.norm_t(h_t + h_t_out)  # 残差
        
        return h_s_out, h_c_out, h_t_out'''

new_fwd = '''        # --- 时间链: 多尺度膨胀卷积 ---
        t_out = self.mamba_t(h_t)  # (B, T, d)
        t_conv = t_out.permute(0, 2, 1)  # (B, d, T)
        multi_scale = [conv(t_conv) for conv in self.dilated_convs]  # 3 x (B, d, T)
        t_fused = torch.cat(multi_scale, dim=1).permute(0, 2, 1)  # (B, T, 3d)
        h_t_out = self.fusion_t(t_fused)  # (B, T, d)
        h_t_out = self.norm_t(h_t + h_t_out)  # 残差
        
        # --- v5: 碱基对耦合 (三链相位同步) ---
        h_s_out, h_c_out, h_t_out, bp_info = self.basepair(h_s_out, h_c_out, h_t_out)
        
        return h_s_out, h_c_out, h_t_out, bp_info'''

if old_fwd in c:
    c = c.replace(old_fwd, new_fwd)
    print("Added basepair to HeteroMamba forward")
else:
    print("WARN: old_fwd not found")

# Fix return signature - now returns 4 values
old_ret = "return h_s_out, h_c_out, h_t_out"
if old_ret not in c.split("return"):
    print("Checking return...")
    for line in c.split("\n"):
        if "return h_s" in line:
            print(f"  Found: {line.strip()}")

with open("models/common.py", "w", encoding="utf-8") as f:
    f.write(c)
print("Done")
