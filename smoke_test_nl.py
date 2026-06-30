import torch, sys, time
sys.path.insert(0, ".")
from models.baselines import SingleChain
from data.dataset import make_loaders

for nl in [2, 10]:
    m = SingleChain(cell_types=16, action_dim=5, d_model=256, n_layers=nl).cuda()
    p = sum(p.numel() for p in m.parameters())
    print(f"n_layers={nl}: {p:,} params")
    
    loaders = make_loaders("data_split", batch_size=16)
    batch = next(iter(loaders["train"]))
    S_0 = batch["S_0"].cuda()
    actions = batch["actions"].cuda()
    S_t = batch["S_t"].cuda()
    
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    t0 = time.time()
    for i in range(20):
        opt.zero_grad()
        logits, info = m(S_0, actions)
        loss, _ = m.loss(logits, S_t)
        loss.backward()
        opt.step()
    t1 = time.time()
    print(f"  20 steps: {t1-t0:.1f}s ({(t1-t0)/20:.3f}s/step)")
    
    del m, opt
    torch.cuda.empty_cache()
    print()
print("Both OK!")
