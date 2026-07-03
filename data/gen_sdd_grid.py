"""SDD → GridWorld .npz 数据生成器 (阶段 2 真实数据验证)

读 Stanford Drone Dataset 的 annotations.txt (vatic 格式),
离散化到 N×N 网格,输出与 gen_grid_world.py 同 schema 的 .npz。

复用 ThreeChainMamba2 的 (S_0, actions, S_t) 接口,零模型改动。

vatic 格式: trackId xmin ymin xmax ymax frame lost occluded generated label
  - trackId: agent ID (每视频内唯一)
  - xmin/ymin/xmax/ymax: bbox 像素坐标
  - frame: 30fps 帧号 (0-indexed)
  - lost: 1=轨迹丢失 (agent 出视野)
  - occluded: 1=被遮挡
  - generated: 1=插值生成
  - label: Pedestrian/Biker/Car/...

输出 .npz (与 gen_grid_world.py 完全一致):
  S_0: (N, N) int64 — 首帧网格状态
  actions: (K, T) int64 — K agent 的 5 类量化位移
  S_t: (T+1, N, N) int64 — 完整轨迹
  N, K, T, p_transfer, scenario_type: 元数据
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ========== 动作常量 (与 gen_grid_world.py 完全一致) ==========
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
NUM_ACTIONS = 5


# ========== SDD 标注解析 ==========
def parse_sdd_annotations(ann_txt_path):
    """读 vatic 格式 annotations.txt。

    返回: list of dict, 每个 dict 是一行记录
    """
    records = []
    with open(ann_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            rec = {
                "trackId": int(parts[0]),
                "xmin": float(parts[1]),
                "ymin": float(parts[2]),
                "xmax": float(parts[3]),
                "ymax": float(parts[4]),
                "frame": int(parts[5]),
                "lost": int(parts[6]),
                "occluded": int(parts[7]),
                "generated": int(parts[8]),
                "label": " ".join(parts[9:]),  # label 可能含空格
            }
            records.append(rec)
    return records


def group_by_track(records):
    """按 trackId 分组。

    返回: {trackId: [(frame, cx, cy, lost), ...]} 按 frame 升序
    """
    tracks = {}
    for r in records:
        cx = (r["xmin"] + r["xmax"]) / 2
        cy = (r["ymin"] + r["ymax"]) / 2
        tracks.setdefault(r["trackId"], []).append(
            (r["frame"], cx, cy, r["lost"])
        )
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])
    return tracks


def assign_slots(tracks, K=8):
    """按有效轨迹长度 top-K 映射到 slot 1..K。

    slot 0 = 空格, slot 1..K = agent ID (与 GridWorld cell_type 编码一致)
    返回: {trackId: slot_id}
    """
    track_lens = {
        tid: sum(1 for _, _, _, lost in pts if lost == 0)
        for tid, pts in tracks.items()
    }
    sorted_tids = sorted(track_lens.keys(), key=lambda t: -track_lens[t])
    top_k = sorted_tids[:K]
    slot_map = {tid: i + 1 for i, tid in enumerate(top_k)}
    return slot_map


# ========== 坐标转换 ==========
def bbox_to_grid(cx, cy, img_w, img_h, N):
    """bbox 中心 (像素) → 网格坐标 (row, col)"""
    col = int(cx / img_w * N)
    row = int(cy / img_h * N)
    col = max(0, min(N - 1, col))
    row = max(0, min(N - 1, row))
    return row, col


def derive_action(pos_t, pos_tp1):
    """从相邻帧位置量化为 5 类 action。

    pos: (row, col)
    主轴方向 (±row/±col), 死区为 0 (任何位移都算动作)
    返回: UP/DOWN/LEFT/RIGHT/STAY (0/1/2/3/4)
    """
    dr = pos_tp1[0] - pos_t[0]
    dc = pos_tp1[1] - pos_t[1]
    if dr == 0 and dc == 0:
        return STAY
    if abs(dr) >= abs(dc):
        return DOWN if dr > 0 else UP
    return RIGHT if dc > 0 else LEFT


def render_state(positions, slot_map, N):
    """渲染一帧的 (N, N) 网格状态。

    positions: {trackId: (row, col)} 或 {trackId: None} (lost)
    slot_map: {trackId: slot_id}
    冲突规则: 低 slot 优先保留 (与 GridWorld 编号小者保留一致)
    """
    S = np.zeros((N, N), dtype=np.int64)
    # 按 slot 降序填入, 低 slot 后填覆盖高 slot (低 slot 优先保留)
    for tid in sorted(slot_map.keys(), key=lambda t: -slot_map[t]):
        pos = positions.get(tid)
        if pos is None:
            continue
        row, col = pos
        S[row, col] = slot_map[tid]
    return S


# ========== 主场景生成 ==========
def generate_scenario_from_sdd(
    ann_txt_path, N=24, K=8, T=100, fps_stride=6, scene_name="unknown"
):
    """从 SDD annotations 生成多个滑窗场景。

    返回: list of (S_0, actions, S_t, meta), 每个滑窗一个
    """
    records = parse_sdd_annotations(ann_txt_path)
    if not records:
        return []

    tracks = group_by_track(records)
    slot_map = assign_slots(tracks, K)
    if not slot_map:
        return []

    # 推断图像尺寸 (用所有 bbox 的 max, 近似)
    img_w = max(r["xmax"] for r in records)
    img_h = max(r["ymax"] for r in records)

    # 降采样: 30fps → 5fps (stride=6)
    all_frames = sorted(set(r["frame"] for r in records))
    sampled_frames = all_frames[::fps_stride]

    # 构建 frame -> {trackId: (row, col) or None} 索引
    sampled_set = set(sampled_frames)
    frame_data = {}
    for r in records:
        if r["frame"] not in sampled_set:
            continue
        cx = (r["xmin"] + r["xmax"]) / 2
        cy = (r["ymin"] + r["ymax"]) / 2
        row, col = bbox_to_grid(cx, cy, img_w, img_h, N)
        fdata = frame_data.setdefault(r["frame"], {})
        fdata[r["trackId"]] = (row, col) if r["lost"] == 0 else None

    # 滑窗切分 (T+1 帧, stride=T//3 重叠 ~70%)
    num_frames = len(sampled_frames)
    stride = max(1, T // 3)
    scenarios = []

    for start in range(0, num_frames - T, stride):
        window_frames = sampled_frames[start : start + T + 1]
        if len(window_frames) < T + 1:
            break

        # 渲染 S_t (T+1 帧)
        S_t = np.zeros((T + 1, N, N), dtype=np.int64)
        positions_seq = []
        for i, f in enumerate(window_frames):
            positions = frame_data.get(f, {})
            S_t[i] = render_state(positions, slot_map, N)
            positions_seq.append(positions)

        S_0 = S_t[0].copy()

        # derive actions (K, T)
        actions = np.zeros((K, T), dtype=np.int64)
        for t in range(T):
            pos_t = positions_seq[t]
            pos_tp1 = positions_seq[t + 1]
            for tid, slot in slot_map.items():
                k = slot - 1  # 0-indexed
                p_t = pos_t.get(tid)
                p_tp1 = pos_tp1.get(tid)
                if p_t is None or p_tp1 is None:
                    actions[k, t] = STAY  # lost 帧默认 STAY
                else:
                    actions[k, t] = derive_action(p_t, p_tp1)

        meta = {
            "N": N,
            "K": K,
            "T": T,
            "p_transfer": 0.0,
            "scenario_type": scene_name,
        }
        scenarios.append((S_0, actions, S_t, meta))

    return scenarios


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(description="SDD → GridWorld .npz 生成器")
    parser.add_argument(
        "--sdd_root", required=True, help="SDD 数据根目录 (含 annotations/)"
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--scenes", nargs="+", default=["bookstore"], help="场景名列表"
    )
    parser.add_argument("--N", type=int, default=24)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--fps_stride", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for scene in args.scenes:
        scene_dir = Path(args.sdd_root) / "annotations" / scene
        if not scene_dir.exists():
            print(f"[警告] 场景目录不存在: {scene_dir}")
            continue
        # 遍历视频
        video_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir()])
        for video_dir in tqdm(video_dirs, desc=f"{scene}"):
            ann_file = video_dir / "annotations.txt"
            if not ann_file.exists():
                continue
            scenarios = generate_scenario_from_sdd(
                ann_file,
                N=args.N,
                K=args.K,
                T=args.T,
                fps_stride=args.fps_stride,
                scene_name=scene,
            )
            for S_0, actions, S_t, meta in scenarios:
                fname = f"scen_{args.N:02d}_{args.K}_{args.T:02d}_0_{scene}_{idx:06d}.npz"
                np.savez(
                    out_dir / fname,
                    S_0=S_0,
                    actions=actions,
                    S_t=S_t,
                    N=args.N,
                    K=args.K,
                    T=args.T,
                    p_transfer=0.0,
                    scenario_type=scene,
                )
                idx += 1
        print(f"  {scene}: 累计 {idx} 个窗口")

    print(f"完成,共 {idx} 个场景到 {out_dir}")


if __name__ == "__main__":
    main()
