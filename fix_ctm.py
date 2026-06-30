import sys; sys.stdout.reconfigure(encoding="utf-8")
with open("models/common.py","r",encoding="utf-8") as f:
    c = f.read()

# Add CTM oscillators in HeteroMamba __init__
old = "        self.basepair = TriHelixBasePairCoupling(d_model)"
new = """        self.basepair = TriHelixBasePairCoupling(d_model)
        
        # v6: CTM多频振荡器(三条链各一个频段)
        self.ctm_s = CTM_Oscillator(64, "delta")   # 空间=慢波
        self.ctm_c = CTM_Oscillator(8, "theta")    # 因果=海马节律
        self.ctm_t = CTM_Oscillator(32, "gamma")   # 时间=快波"""

c = c.replace(old, new)

# Add CTM after each chain's Mamba output in forward
# After spatial Mamba output
old_s = "h_s_out = self.norm_s_in(h_s + h_s_out)"
new_s = """h_s_out = self.norm_s_in(h_s + h_s_out)
        h_s_out = h_s_out + 0.1 * self.ctm_s(h_s_out)  # CTM微调"""

old_c = "h_c_out = self.norm_c_in(h_c + gate * c_out + (1 - gate) * agent_mean)"
new_c = """h_c_out = self.norm_c_in(h_c + gate * c_out + (1 - gate) * agent_mean)
        h_c_out = h_c_out + 0.1 * self.ctm_c(h_c_out)  # CTM微调"""

old_t = "h_t_out = self.norm_t_in(h_t + h_t_out)"
new_t = """h_t_out = self.norm_t_in(h_t + h_t_out)
        h_t_out = h_t_out + 0.1 * self.ctm_t(h_t_out)  # CTM微调"""

for old_line, new_line in [(old_s, new_s), (old_c, new_c), (old_t, new_t)]:
    if old_line in c:
        c = c.replace(old_line, new_line)
        print(f"  Added CTM to chain")
    else:
        print(f"  WARN: not found")

with open("models/common.py","w",encoding="utf-8") as f:
    f.write(c)
print("CTM integrated into HeteroMamba")
