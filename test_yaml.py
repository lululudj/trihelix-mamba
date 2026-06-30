import yaml
with open("/mnt/e/three_chain_v3/configs/default.yaml") as f:
    d = yaml.safe_load(f)
print("default OK, n_layers=" + str(d["model"]["n_layers"]))
with open("/mnt/e/three_chain_v3/configs/scaled_single.yaml") as f:
    d = yaml.safe_load(f)
print("scaled_single OK, n_layers=" + str(d["model"]["n_layers"]))
with open("/mnt/e/three_chain_v3/configs/scaled_transformer.yaml") as f:
    d = yaml.safe_load(f)
print("scaled_transformer OK, n_layers=" + str(d["model"]["n_layers"]))
