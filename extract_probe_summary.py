"""临时: 提取 probe .npz 关键数据点用于假说验证报告"""
import numpy as np
from pathlib import Path

for name in ["probe_100step_m3subset_seed0", "probe_benchmark_seed42"]:
    d = np.load(Path("results_stage3") / f"{name}.npz", allow_pickle=True)
    ns = d["norm_s_last"].item()
    nt = d["norm_t_last"].item()
    nc = d["norm_c_last"].item()
    cst = d["cos_st_last"].item()
    csc = d["cos_sc_last"].item()
    ctc = d["cos_tc_last"].item()
    t_final = max(int(k) for k in ns)
    print(f"\n=== {name} ===")
    print(f"  t=1:   s={ns['1']:.2f}  t={nt['1']:.2f}  c={nc['1']:.2f}  cos_st={cst['1']:.3f}  cos_sc={csc['1']:.3f}  cos_tc={ctc['1']:.3f}")
    print(f"  t=50:  s={ns['50']:.2f}  t={nt['50']:.2f}  c={nc['50']:.2f}  cos_st={cst['50']:.3f}  cos_sc={csc['50']:.3f}  cos_tc={ctc['50']:.3f}")
    print(f"  t=100: s={ns['100']:.2f}  t={nt['100']:.2f}  c={nc['100']:.2f}  cos_st={cst['100']:.3f}  cos_sc={csc['100']:.3f}  cos_tc={ctc['100']:.3f}")
    print(f"  t={t_final}: s={ns[str(t_final)]:.2f}  t={nt[str(t_final)]:.2f}  c={nc[str(t_final)]:.2f}  cos_st={cst[str(t_final)]:.3f}  cos_sc={csc[str(t_final)]:.3f}  cos_tc={ctc[str(t_final)]:.3f}")
    s_d = (ns[str(t_final)] - ns['100']) / ns['100'] * 100
    t_d = (nt[str(t_final)] - nt['100']) / nt['100'] * 100
    c_d = (nc[str(t_final)] - nc['100']) / nc['100'] * 100
    print(f"  OOD decay (t=100→{t_final}): s={s_d:+.1f}%  t={t_d:+.1f}%  c={c_d:+.1f}%")
