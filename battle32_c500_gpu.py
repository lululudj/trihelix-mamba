"""C500版battle32_gpu.py补丁: 在训练循环中加入torch.cuda.synchronize()防止rq_qos_wait死锁

策略:
  1. BATCH_SIZE=1 (C500内存安全)
  2. 每步训练后torch.cuda.synchronize()防止kernel队列堆积
  3. 每个模型训练完强制synchronize + sleep(0.5)
  4. 捕获更多异常类型, 单模型失败不中断整体实验
"""
import sys, json, time, logging, os, gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

LOG_PATH = Path(__file__).parent / "battle32_c500_log.txt"
_file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_logger = logging.getLogger("battle32_c500")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _logger.addHandler(_file_handler)

def log(msg):
    print(msg, flush=True)
    _logger.info(msg)

from data.gen_grid_world import generate_scenario
from data.dataset import GridWorldDataset, GroupedByNSampler, collate_fn

# ========== C500配置 ==========
D_MODEL = 64
BATCH_SIZE = 1
MAX_STEPS = 100
LR = 1e-3
AUX_WEIGHT = 0.3
SEEDS = [42, 123, 456]
N_TRAIN, N_VAL, N_OOD = 60, 16, 16

DATA_ROOT = Path(__file__).parent / "data_battle32"
RESULTS_PATH = Path(__file__).parent / "battle32_results.json"

SCENARIOS_32 = [
    ("rand_n6_k4",     6,  4, 100, 200, 0.0,  "random",        "随机:小网格少agent"),
    ("rand_n6_k8",     6,  8, 100, 200, 0.15, "random",        "随机:小网格中agent"),
    ("rand_n8_k4",     8,  4, 100, 200, 0.0,  "random",        "随机:中网格少agent"),
    ("rand_n8_k8",     8,  8, 100, 200, 0.15, "random",        "随机:中网格中agent"),
    ("rand_n8_k12",    8, 12, 100, 200, 0.3,  "random",        "随机:中网格多agent"),
    ("rand_n12_k8",   12,  8, 100, 200, 0.15, "random",        "随机:大网格中agent"),
    ("rand_n12_k12",  12, 12, 100, 200, 0.3,  "random",        "随机:大网格多agent"),
    ("rand_n6_k12",    6, 12, 100, 200, 0.6,  "random",        "随机:高密度强转移"),
    ("goal_n6_k4",     6,  4, 100, 200, 0.0,  "goal_directed", "目标:小网格少agent"),
    ("goal_n6_k8",     6,  8, 100, 200, 0.15, "goal_directed", "目标:小网格中agent"),
    ("goal_n8_k4",     8,  4, 100, 200, 0.05, "goal_directed", "目标:中网格少agent"),
    ("goal_n8_k8",     8,  8, 100, 200, 0.15, "goal_directed", "目标:中网格中agent"),
    ("goal_n8_k12",    8, 12, 100, 200, 0.3,  "goal_directed", "目标:中网格多agent"),
    ("goal_n12_k4",   12,  4, 100, 200, 0.05, "goal_directed", "目标:大网格少agent"),
    ("goal_n12_k8",   12,  8, 100, 200, 0.15, "goal_directed", "目标:大网格中agent"),
    ("goal_n12_k12",  12, 12, 100, 200, 0.3,  "goal_directed", "目标:大网格多agent"),
    ("goal_n6_k4_p6",  6,  4, 100, 200, 0.6,  "goal_directed", "目标:强转移"),
    ("goal_n8_k8_p6",  8,  8, 100, 200, 0.6,  "goal_directed", "目标:中密度强转移"),
    ("goal_n12_k4_l", 12,  4, 100, 200, 0.0,  "goal_directed", "目标:大网格长程"),
    ("goal_n6_k12_l",  6, 12, 100, 200, 0.3,  "goal_directed", "目标:高密度长程"),
    ("adv_n6_k4",      6,  4, 100, 200, 0.0,  "adversarial",   "对抗:小网格少agent"),
    ("adv_n6_k8",      6,  8, 100, 200, 0.15, "adversarial",   "对抗:小网格中agent"),
    ("adv_n8_k4",      8,  4, 100, 200, 0.05, "adversarial",   "对抗:中网格少agent"),
    ("adv_n8_k8",      8,  8, 100, 200, 0.15, "adversarial",   "对抗:中网格中agent"),
    ("adv_n8_k12",     8, 12, 100, 200, 0.3,  "adversarial",   "对抗:中网格多agent"),
    ("adv_n12_k8",    12,  8, 100, 200, 0.15, "adversarial",   "对抗:大网格中agent"),
    ("adv_n12_k12",   12, 12, 100, 200, 0.3,  "adversarial",   "对抗:大网格多agent"),
    ("adv_n6_k12_p6",  6, 12, 100, 200, 0.6,  "adversarial",   "对抗:高密度强转移"),
    ("ext_n6_k4",      6,  4, 100, 200, 0.0,  "goal_directed", "极端:最小最快"),
    ("ext_n12_k12",   12, 12, 100, 200, 0.6,  "adversarial",   "极端:最大最密"),
    ("ext_n8_k8_hp",   8,  8, 100, 200, 0.6,  "random",        "极端:高转移随机"),
    ("ext_n8_k4_drone",8,  4, 100, 200, 0.05, "goal_directed", "极端:无人机轨迹"),
]

MODELS = [
    ("Transformer",    "transformer",         3, {}),
    ("Mamba3单链",      "single_chain_mamba3",  1, {}),
    ("三链Mamba3",      "three_chain_mamba3",   1, {}),
    ("三链Mamba3+BP",   "three_chain_mamba3",   1, {"enable_bp": True}),
]

def gen_scen_files(out_dir, n_samples, N, K, T, p_transfer, scenario_type, seed_base):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        S_0, actions, trajectory, meta = generate_scenario(
            N=N, K=K, T=T, p_transfer=p_transfer,
            scenario_type=scenario_type, seed=seed_base + i,
        )
        np.savez_compressed(
            out_dir / f"scen_{i:05d}.npz",
            S_0=S_0, actions=actions, S_t=trajectory,
            N=np.array(N), K=np.array(K), T=np.array(T),
            p_transfer=np.array(p_transfer),
            scenario_type=np.array(scenario_type),
        )

def prepare_scenario_data(name, N, K, T_train, T_ood, p_transfer, scenario_type):
    scen_dir = DATA_ROOT / name
    for split, n, T, seed_base in [
        ("train", N_TRAIN, T_train, hash(name) % 10000),
        ("val",   N_VAL,   T_train, 100000 + hash(name) % 10000),
        ("ood",   N_OOD,   T_ood,   200000 + hash(name) % 10000),
    ]:
        out = scen_dir / split
        if out.exists() and any(out.glob("scen_*.npz")):
            continue
        log(f"    gen {name}/{split}: {n} samples, T={T}")
        gen_scen_files(out, n, N, K, T, p_transfer, scenario_type, seed_base)
    return scen_dir

def build_model(registry_name, n_layers, max_T, extra_kwargs):
    from models import MODEL_REGISTRY
    cls = MODEL_REGISTRY[registry_name]
    kwargs = {
        "cell_types": 16, "action_dim": 5,
        "d_model": D_MODEL, "n_layers": n_layers, "max_T": max_T,
    }
    kwargs.update(extra_kwargs)
    try:
        model = cls(**kwargs)
    except TypeError:
        kwargs.pop("max_T")
        model = cls(**kwargs)
    return model, sum(p.numel() for p in model.parameters())

def compute_loss(model, logits, S_t, info):
    try:
        loss, li = model.loss(logits, S_t, info, aux_weight=AUX_WEIGHT)
    except TypeError:
        loss, li = model.loss(logits, S_t, aux_weight=AUX_WEIGHT)
    return loss

def quick_eval(model, loader, device):
    model.eval()
    ac, at, cc, ct = 0, 0, 0, 0
    with torch.no_grad():
        for b in loader:
            S_0 = b["S_0"].to(device)
            actions = b["actions"].to(device)
            S_t = b["S_t"].to(device)
            logits, info = model(S_0, actions)
            torch.cuda.synchronize()
            pred = logits.argmax(-1)
            target = S_t[:, 1:]
            ac += (pred == target).sum().item()
            at += target.numel()
            cm = (target != S_0.unsqueeze(1))
            cc += ((pred == target) & cm).sum().item()
            ct += cm.sum().item()
    return {"acc": ac/max(at,1), "ch_acc": cc/max(ct,1) if ct > 0 else 0}

def train_and_eval(display_name, registry_name, n_layers, max_T,
                   train_loader, val_loader, ood_loader, device, seed, extra_kwargs):
    log(f"    --- {display_name} (seed={seed}) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        model, n_params = build_model(registry_name, n_layers, max_T, extra_kwargs)
    except Exception as e:
        import traceback
        log(f"      [FAIL] build: {traceback.format_exc()}")
        return None
    model = model.to(device)
    torch.cuda.synchronize()
    log(f"      params: {n_params:,}, steps: {MAX_STEPS}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    t0 = time.time()
    step = 0
    while step < MAX_STEPS:
        for batch in train_loader:
            if step >= MAX_STEPS:
                break
            S_0 = batch["S_0"].to(device)
            actions = batch["actions"].to(device)
            S_t = batch["S_t"].to(device)
            try:
                logits, info = model(S_0, actions)
                loss = compute_loss(model, logits, S_t, info)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
            except RuntimeError as e:
                emsg = str(e).lower()
                if "out of memory" in emsg:
                    log(f"      [OOM] step {step}, skip batch")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    continue
                if "device-side assert" in emsg or "cuda error" in emsg or "maca" in emsg:
                    log(f"      [GPU ERROR] step {step}: {str(e)[:200]}")
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise
                raise
            if step % 20 == 0:
                bp_str = ""
                if hasattr(model, 'mamba') and hasattr(model.mamba, 'base_pair'):
                    ms = model.mamba.base_pair[0].modulation_strength()
                    bp_str = f" γ_norm={ms['bp3_gamma']:.4f}"
                log(f"      step {step:3d} | loss={loss.item():.4f} | {time.time()-t0:.1f}s{bp_str}")
            step += 1

    torch.cuda.synchronize()
    train_time = time.time() - t0
    val_m = quick_eval(model, val_loader, device)
    torch.cuda.synchronize()
    ood_m = quick_eval(model, ood_loader, device)
    torch.cuda.synchronize()
    decay = (ood_m["ch_acc"] - val_m["ch_acc"]) / max(val_m["ch_acc"], 1e-8) * 100

    result = {
        "model": display_name,
        "seed": seed,
        "params": n_params,
        "steps": MAX_STEPS,
        "train_time_s": round(train_time, 1),
        "val_acc": round(val_m["acc"]*100, 2),
        "val_ch_acc": round(val_m["ch_acc"]*100, 2),
        "ood_acc": round(ood_m["acc"]*100, 2),
        "ood_ch_acc": round(ood_m["ch_acc"]*100, 2),
        "ood_decay_pct": round(decay, 2),
    }
    if hasattr(model, 'mamba') and hasattr(model.mamba, 'base_pair'):
        ms = model.mamba.base_pair[0].modulation_strength()
        result["bp_mod_strength"] = {k: round(v, 4) for k, v in ms.items()}

    log(f"      val_ch={result['val_ch_acc']}%  ood_ch={result['ood_ch_acc']}%  "
        f"decay={result['ood_decay_pct']}%  [{train_time:.1f}s]")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(0.3)
    return result

def make_loaders(scen_dir, seed):
    train_ds = GridWorldDataset(str(scen_dir / "train"))
    val_ds = GridWorldDataset(str(scen_dir / "val"))
    ood_ds = GridWorldDataset(str(scen_dir / "ood"))
    def make_loader(ds, shuffle):
        s = GroupedByNSampler(ds.Ns, batch_size=BATCH_SIZE, shuffle=shuffle,
                              seed=seed if shuffle else 0, Ts=ds.Ts, Ks=ds.Ks)
        return DataLoader(ds, batch_sampler=s, collate_fn=collate_fn)
    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(ood_ds, False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{'='*70}")
    log(f"  C500 32场景 × 4模型 × 3seed (加固版, BATCH_SIZE={BATCH_SIZE})")
    log(f"{'='*70}")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        _props = torch.cuda.get_device_properties(0)
        _vram = getattr(_props, 'total_memory', getattr(_props, 'total_mem', 0))
        log(f"VRAM: {_vram/1024**3:.1f} GB")
    try:
        import mamba_ssm
        log(f"mamba_ssm: {mamba_ssm.__version__}")
    except ImportError:
        log("mamba_ssm: not installed")
    log(f"MAMBA3_FORCE_REF: {os.environ.get('MAMBA3_FORCE_REF', 'not set')}")
    log(f"Config: D_MODEL={D_MODEL}, BATCH_SIZE={BATCH_SIZE}, MAX_STEPS={MAX_STEPS}")
    log(f"防护: 每步cuda.synchronize() + 模型间sleep(0.3s) + BATCH_SIZE=1")

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        log(f"\n已有结果: {len(all_results)} 条, 断点续跑")
    else:
        all_results = {}

    total = len(SCENARIOS_32) * len(MODELS) * len(SEEDS)
    done = 0
    t_start = time.time()
    errors_row = 0

    for scen in SCENARIOS_32:
        scen_name = scen[0]
        N, K, T_train, T_ood = scen[1], scen[2], scen[3], scen[4]
        p_transfer, scenario_type, desc = scen[5], scen[6], scen[7]

        log(f"\n{'='*70}")
        log(f"  场景: {scen_name} ({desc})")
        log(f"{'='*70}")

        try:
            scen_dir = prepare_scenario_data(scen_name, N, K, T_train, T_ood,
                                             p_transfer, scenario_type)
        except Exception as e:
            log(f"  [FAIL] data gen: {e}")
            continue

        for seed in SEEDS:
            try:
                train_loader, val_loader, ood_loader = make_loaders(scen_dir, seed)
            except Exception as e:
                log(f"  [FAIL] make_loaders seed={seed}: {e}")
                continue

            for display_name, registry_name, n_layers, extra_kwargs in MODELS:
                done += 1
                key = f"{scen_name}|seed{seed}|{display_name}"
                if key in all_results and "error" not in all_results[key]:
                    log(f"  [{done}/{total}] [skip] {key}")
                    continue

                log(f"  [{done}/{total}] {key}")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                try:
                    r = train_and_eval(display_name, registry_name, n_layers, T_ood,
                                       train_loader, val_loader, ood_loader,
                                       device, seed, extra_kwargs)
                    if r is not None:
                        all_results[key] = r
                        errors_row = 0
                    else:
                        all_results[key] = {"model": display_name, "seed": seed, "error": "build_failed"}
                except Exception as e:
                    import traceback
                    err_str = traceback.format_exc()
                    log(f"      [ERROR]: {str(e)[:300]}")
                    all_results[key] = {"model": display_name, "seed": seed, "error": str(e)[:500]}
                    errors_row += 1
                    if errors_row >= 5:
                        log("  !!! 连续5个错误, 尝试重置GPU状态 !!!")
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        time.sleep(2)
                        errors_row = 0

                with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

                elapsed = time.time() - t_start
                if done > 0 and all(k in all_results and "error" not in all_results[k] for k in []):
                    n_ok = sum(1 for v in all_results.values() if "error" not in v)
                    eta = elapsed / max(n_ok,1) * (total - n_ok)
                    log(f"      进度: {len(all_results)}/{total}  已用: {elapsed/60:.1f}min  ETA: {eta/60:.1f}min")

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # Stats
    log(f"\n{'='*70}")
    n_ok = sum(1 for v in all_results.values() if "error" not in v)
    n_err = sum(1 for v in all_results.values() if "error" in v)
    log(f"  完成: {n_ok}/{total}, 错误: {n_err}")
    log(f"  总耗时: {(time.time()-t_start)/60:.1f} min")

    try:
        log("运行stats_analysis.py...")
        import subprocess
        subprocess.run([sys.executable, "stats_analysis.py"], timeout=300)
    except Exception as e:
        log(f"stats_analysis failed: {e}")

    log("=== C500 EXPERIMENT COMPLETE ===")

if __name__ == "__main__":
    main()
