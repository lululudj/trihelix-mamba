import sys, numpy as np
from pathlib import Path

sys.path.insert(0, '/mnt/e/three_chain_v3')
from data.gen_grid_world import generate_scenario

# OOD Level 2: T=200 with 20% action noise
OUT2 = Path('/mnt/e/three_chain_v3/data/ood_T200_noise')
OUT2.mkdir(parents=True, exist_ok=True)
N, K, T = 8, 8, 200
scenario_types = ['random', 'goal_directed', 'adversarial']
p_transfers = [0.0, 0.3, 0.6]
n_per = 8
seed_base = 200000

i = 0
for st in scenario_types:
    for p in p_transfers:
        for j in range(n_per):
            seed = seed_base + i
            S_0, actions, trajectory, meta = generate_scenario(N, K, T, p_transfer=p, scenario_type=st, seed=seed)
            # actions: (N, T), add 20% noise to action columns
            rng = np.random.RandomState(seed+88888)
            noise_mask = rng.random(T) < 0.2
            noisy_actions = actions.copy()
            noisy_actions[:, noise_mask] = rng.randint(0, 5, (N, noise_mask.sum()))
            fname = f'scen_{N:02d}_{K}_{T:03d}_{int(p*10)}_{st}_{i:06d}.npz'
            np.savez(OUT2 / fname, S_0=S_0, actions=noisy_actions, S_t=trajectory, N=N, K=K, T=T, p_transfer=p, scenario_type=st)
            i += 1
print(f'OOD T200+noise: {i} samples in {OUT2}')

# OOD Level 3: T=250 with 30% masked states 
OUT3 = Path('/mnt/e/three_chain_v3/data/ood_T250_mask')
OUT3.mkdir(parents=True, exist_ok=True)
T = 250
seed_base = 300000

i = 0
for st in scenario_types:
    for p in p_transfers:
        for j in range(n_per):
            seed = seed_base + i
            S_0, actions, trajectory, meta = generate_scenario(N, K, T, p_transfer=p, scenario_type=st, seed=seed)
            # S_0: (N,N), trajectory: (T+1, N, N). Mask 30% in S_0
            rng = np.random.RandomState(seed+77777)
            mask = rng.random((N, N)) < 0.3
            S_0_masked = S_0.copy()
            S_0_masked[mask] = -1
            fname = f'scen_{N:02d}_{K}_{T:03d}_{int(p*10)}_{st}_{i:06d}.npz'
            np.savez(OUT3 / fname, S_0=S_0_masked, actions=actions, S_t=trajectory, N=N, K=K, T=T, p_transfer=p, scenario_type=st)
            i += 1
print(f'OOD T250+mask: {i} samples in {OUT3}')
