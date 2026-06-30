import sys; sys.stdout.reconfigure(encoding="utf-8")
with open("models/three_chain.py","r",encoding="utf-8") as f:
    t = f.read()

# Fix: handle bp_info from HeteroMamba
old_line = "h_s, h_c, h_t = self.mamba(h_s, h_c, h_t)"
new_line = "h_s, h_c, h_t, bp_info = self.mamba(h_s, h_c, h_t)"
t = t.replace(old_line, new_line)

# Add bp_info to aux dict
old_aux = 'aux = {"space_logits": space_logits, "causal_logits": causal_logits, "time_pred": time_pred}'
new_aux = 'aux = {"space_logits": space_logits, "causal_logits": causal_logits, "time_pred": time_pred, "bp_info": bp_info}'
t = t.replace(old_aux, new_aux)

# Update docstring
t = t.replace("v4: HeteroMamba", "v5: HeteroMamba + BasePair Coupling")

with open("models/three_chain.py","w",encoding="utf-8") as f:
    f.write(t)
print("OK: three_chain.py updated to V5")
