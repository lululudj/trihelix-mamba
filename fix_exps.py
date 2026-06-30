import sys; sys.stdout.reconfigure(encoding="utf-8")
import re

# Fix exp1
with open("killer_experiments/exp1_chain_ablation/run_exp1.py","r",encoding="utf-8") as f:
    c = f.read()

# Fix import path
c = c.replace("three_chain_validation", ".")

# Fix mamba_s -> mamba.mamba_s_row etc
c = c.replace("model.mamba_s", "model.mamba.mamba_s_row")
c = c.replace("model.mamba_c", "model.mamba.mamba_c")
c = c.replace("model.mamba_t", "model.mamba.mamba_t")

with open("killer_experiments/exp1_chain_ablation/run_exp1.py","w",encoding="utf-8") as f:
    f.write(c)
print("Fixed exp1")

# Fix exp2
with open("killer_experiments/exp2_causal_needle/run_exp2.py","r",encoding="utf-8") as f:
    c = f.read()
c = c.replace("three_chain_validation", ".")
with open("killer_experiments/exp2_causal_needle/run_exp2.py","w",encoding="utf-8") as f:
    f.write(c)
print("Fixed exp2")

# Fix exp3
with open("killer_experiments/exp3_concept_drift/run_exp3.py","r",encoding="utf-8") as f:
    c = f.read()
c = c.replace("three_chain_validation", ".")
with open("killer_experiments/exp3_concept_drift/run_exp3.py","w",encoding="utf-8") as f:
    f.write(c)
print("Fixed exp3")

# Fix exp4
with open("killer_experiments/exp4_maze_navigation/run_exp4.py","r",encoding="utf-8") as f:
    c = f.read()
c = c.replace("three_chain_validation", ".")
with open("killer_experiments/exp4_maze_navigation/run_exp4.py","w",encoding="utf-8") as f:
    f.write(c)
print("Fixed exp4")
