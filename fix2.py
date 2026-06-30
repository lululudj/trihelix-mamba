import sys; sys.stdout.reconfigure(encoding="utf-8")
with open("models/common.py","r",encoding="utf-8") as f:
    c = f.read()
c = c.replace('CTM_Oscillator(64, "delta")', 'CTM_Oscillator(d_model, "delta")')
c = c.replace('CTM_Oscillator(8, "theta")', 'CTM_Oscillator(d_model, "theta")')
c = c.replace('CTM_Oscillator(32, "gamma")', 'CTM_Oscillator(d_model, "gamma")')
with open("models/common.py","w",encoding="utf-8") as f:
    f.write(c)
print("Fixed")
