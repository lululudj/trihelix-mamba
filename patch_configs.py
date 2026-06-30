import yaml
from pathlib import Path

base = Path('/mnt/e/three_chain_v3/configs')

# matched_three: d_model=300
with open(base / 'matched_three.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['model']['d_model'] = 300
cfg['train']['max_steps'] = 2000
cfg['train']['batch_size'] = 8
with open(base / 'matched_three.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('matched_three: d_model=300, max_steps=2000, batch=8')

# matched_single: d_model=256, n_layers=9
with open(base / 'matched_single.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['model']['d_model'] = 256
cfg['model']['n_layers'] = 9
cfg['train']['max_steps'] = 2000
cfg['train']['batch_size'] = 16
with open(base / 'matched_single.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('matched_single: d_model=256, n_layers=9, max_steps=2000')

# matched_transformer: d_model=256, n_layers=5
with open(base / 'matched_transformer.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['model']['d_model'] = 256
cfg['model']['n_layers'] = 5
cfg['train']['max_steps'] = 2000
cfg['train']['batch_size'] = 16
with open(base / 'matched_transformer.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('matched_transformer: d_model=256, n_layers=5, max_steps=2000')

print('Done patching configs')
