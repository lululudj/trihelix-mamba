import sys; sys.stdout.reconfigure(encoding="utf-8")
with open("models/common.py","r",encoding="utf-8") as f:
    c = f.read()
# Fix d_conv values
c = c.replace("d_conv=7, expand=1)","d_conv=4, expand=1)  # d_conv max=4 per causal_conv1d limit")
c = c.replace("d_conv=5, expand=2)","d_conv=4, expand=2)  # d_conv max=4")
# Fix the second mamba_s_row reference for the col scanner (appears after row)
c = c.replace("self.mamba_s_col = Mamba(d_model=d_model, d_state=64, d_conv=7, expand=1)  # d_conv max=4 per causal_conv1d limit",
              "self.mamba_s_col = Mamba(d_model=d_model, d_state=64, d_conv=4, expand=1)  # col scanner same params")
with open("models/common.py","w",encoding="utf-8") as f:
    f.write(c)
print("Fixed d_conv values")
# Verify
for line in c.split("\n"):
    if "d_conv" in line and "Mamba" in line:
        print(line.strip()[:100])
