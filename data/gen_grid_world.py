"""网格世界 simulator + 数据生成器

按 spec §1.1 / §4 实现：
- N×N 网格，K 个智能体并行演化 T 步
- 动作集 A = {上, 下, 左, 右, 停留}
- 冲突规则：同格多智能体 → 编号小者保留，其余回退
- 携带物：智能体可携带 token，冲突时按 p_transfer 概率转移

Cell 类型编码（C=16，难度升级 v2）：
    0      空格
    1-12   智能体 ID（最多 K=12）
    13-15  物品 ID（最多 3 个）

3 种场景：
    random         50%  完全随机动作
    goal_directed  30%  每智能体有目标格，贪心选最短路径
    adversarial    20%  两两配对，pursuer 追 escaper
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ========== 动作常量 ==========
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    STAY: (0, 0),
}
NUM_ACTIONS = 5

# ========== 默认配置（spec §4.1，难度升级 v2） ==========
# 小网格 + 多智能体 + 长序列：cell 变化率从 6% → 25%+
ALL_N = [6, 8, 12]
ALL_K = [4, 8, 12]
ALL_T = [30, 60, 100, 200]
ALL_P = [0.0, 0.15, 0.3, 0.6]
NUM_ITEMS = 3
# 物品 ID 起始（避开智能体 ID 段 1..K_max=12）
ITEM_ID_BASE = 13
# 每子配置场景数（按 50/30/20 比例分配场景类型，合计 125）
PER_CONFIG = {"random": 62, "goal_directed": 38, "adversarial": 25}


# ========== Simulator ==========
class GridWorld:
    """网格世界 simulator（单场景）。

    状态 S 是 (N, N) int 数组，编码见模块 docstring。
    """

    def __init__(self, N, K, num_items=NUM_ITEMS, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.N = N
        self.K = K
        self.num_items = num_items

        self.agent_pos = self._sample_unique_positions(K)
        self.item_pos = self._sample_unique_positions(num_items, exclude=self.agent_pos)
        # 智能体携带物 (K,)：0 无携带，9..11 物品 ID
        self.carrying = np.zeros(K, dtype=np.int64)
        self.item_taken = np.zeros(num_items, dtype=bool)

        self.S = self._render()

    def _sample_unique_positions(self, count, exclude=None):
        occupied = set()
        if exclude is not None:
            for r, c in exclude:
                occupied.add((int(r), int(c)))
        positions = []
        while len(positions) < count:
            r = int(self.rng.integers(0, self.N))
            c = int(self.rng.integers(0, self.N))
            if (r, c) in occupied:
                continue
            occupied.add((r, c))
            positions.append([r, c])
        return np.array(positions, dtype=np.int64)

    def _render(self):
        S = np.zeros((self.N, self.N), dtype=np.int64)
        # 物品（未被携带的）
        for i in range(self.num_items):
            if not self.item_taken[i]:
                r, c = self.item_pos[i]
                S[int(r), int(c)] = ITEM_ID_BASE + i
        # 智能体（覆盖物品：若同格则智能体优先显示）
        for k in range(self.K):
            r, c = self.agent_pos[k]
            S[int(r), int(c)] = 1 + k
        return S

    def step(self, actions, p_transfer=0.0):
        """执行 K 个智能体的动作，返回新的 S。"""
        target_pos = self.agent_pos.copy()
        for k in range(self.K):
            dr, dc = ACTION_DELTAS[int(actions[k])]
            nr = int(self.agent_pos[k, 0] + dr)
            nc = int(self.agent_pos[k, 1] + dc)
            if 0 <= nr < self.N and 0 <= nc < self.N:
                target_pos[k] = [nr, nc]

        # 冲突处理：同 cell → 编号小者保留，其余回退
        cell_owner = {}
        conflicts = []  # (loser, winner) 用于携带物转移
        for k in range(self.K):
            cell = (int(target_pos[k, 0]), int(target_pos[k, 1]))
            if cell in cell_owner:
                conflicts.append((k, cell_owner[cell]))
                target_pos[k] = self.agent_pos[k].copy()
            else:
                cell_owner[cell] = k

        self.agent_pos = target_pos

        # 拾取物品（智能体到达物品位置且未携带时）
        for k in range(self.K):
            if self.carrying[k] != 0:
                continue
            for i in range(self.num_items):
                if self.item_taken[i]:
                    continue
                if (self.agent_pos[k] == self.item_pos[i]).all():
                    self.carrying[k] = ITEM_ID_BASE + i
                    self.item_taken[i] = True
                    break

        # 携带物转移：冲突时按 p_transfer 从 loser 转给 winner
        if p_transfer > 0:
            for loser, winner in conflicts:
                if self.carrying[loser] != 0 and self.carrying[winner] == 0:
                    if self.rng.random() < p_transfer:
                        self.carrying[winner] = self.carrying[loser]
                        self.carrying[loser] = 0

        self.S = self._render()
        return self.S


# ========== 场景动作生成 ==========
def _gen_goal_directed(world, goals, T, rng, greedy_prob=0.7):
    """生成 goal_directed 动作序列。

    用"理想位置"贪心决策（不真正 forward world），动作与 trajectory 重放解耦。
    """
    actions = np.zeros((world.K, T), dtype=np.int64)
    cur_pos = world.agent_pos.copy()
    for t in range(T):
        for k in range(world.K):
            if rng.random() < greedy_prob:
                r, c = cur_pos[k]
                gr, gc = goals[k]
                dr, dc = int(gr) - int(r), int(gc) - int(c)
                if dr == 0 and dc == 0:
                    a = STAY
                elif abs(dr) >= abs(dc):
                    a = DOWN if dr > 0 else UP
                else:
                    a = RIGHT if dc > 0 else LEFT
            else:
                a = int(rng.integers(0, NUM_ACTIONS))
            actions[k, t] = a
            dr_a, dc_a = ACTION_DELTAS[a]
            nr, nc = int(cur_pos[k, 0] + dr_a), int(cur_pos[k, 1] + dc_a)
            if 0 <= nr < world.N and 0 <= nc < world.N:
                cur_pos[k] = [nr, nc]
    return actions


def _gen_adversarial(world, K, T, rng, greedy_prob=0.7):
    """生成 adversarial 动作序列：两两配对，pursuer 追 escaper。"""
    pairs = {}
    for i in range(0, K - 1, 2):
        pairs[i] = i + 1
        pairs[i + 1] = i
    if K % 2 == 1:
        pairs[K - 1] = K - 1  # 落单者跟自己（停留）

    actions = np.zeros((K, T), dtype=np.int64)
    cur_pos = world.agent_pos.copy()
    for t in range(T):
        for k in range(K):
            if rng.random() >= greedy_prob:
                a = int(rng.integers(0, NUM_ACTIONS))
            else:
                partner = pairs[k]
                if partner == k:
                    a = STAY
                else:
                    pr, pc = cur_pos[partner]
                    r, c = cur_pos[k]
                    dr, dc = int(pr) - int(r), int(pc) - int(c)
                    is_pursuer = (k % 2 == 1)
                    if is_pursuer:
                        if abs(dr) >= abs(dc):
                            a = DOWN if dr > 0 else UP
                        else:
                            a = RIGHT if dc > 0 else LEFT
                    else:  # escaper 反向跑
                        if abs(dr) >= abs(dc):
                            a = UP if dr > 0 else DOWN
                        else:
                            a = LEFT if dc > 0 else RIGHT
                        dr_a, dc_a = ACTION_DELTAS[a]
                        nr, nc = int(r + dr_a), int(c + dc_a)
                        if not (0 <= nr < world.N and 0 <= nc < world.N):
                            a = STAY
            actions[k, t] = a
            dr_a, dc_a = ACTION_DELTAS[a]
            nr, nc = int(cur_pos[k, 0] + dr_a), int(cur_pos[k, 1] + dc_a)
            if 0 <= nr < world.N and 0 <= nc < world.N:
                cur_pos[k] = [nr, nc]
    return actions


def generate_scenario(N, K, T, p_transfer, scenario_type, seed=None):
    """生成一个完整场景：返回 (S_0, actions, trajectory, meta)。"""
    rng = np.random.default_rng(seed)
    world = GridWorld(N, K, rng=rng)

    if scenario_type == "random":
        actions = rng.integers(0, NUM_ACTIONS, size=(K, T), dtype=np.int64)
    elif scenario_type == "goal_directed":
        goals = world._sample_unique_positions(K)
        actions = _gen_goal_directed(world, goals, T, rng)
    elif scenario_type == "adversarial":
        actions = _gen_adversarial(world, K, T, rng)
    else:
        raise ValueError(f"Unknown scenario_type: {scenario_type}")

    # 重放 actions 得到干净轨迹（用同一个 seed 重建 world，保证 S_0 一致）
    world = GridWorld(N, K, rng=np.random.default_rng(seed))
    S_0 = world.S.copy()
    trajectory = np.zeros((T + 1, N, N), dtype=np.int64)
    trajectory[0] = S_0
    for t in range(T):
        world.step(actions[:, t], p_transfer=p_transfer)
        trajectory[t + 1] = world.S.copy()

    meta = {
        "N": N, "K": K, "T": T,
        "p_transfer": p_transfer, "scenario_type": scenario_type,
    }
    return S_0, actions, trajectory, meta


# ========== 配置生成 ==========
def iter_all_configs():
    """按 spec §4.1 生成 81 子配置 × 125 场景的 (N,K,T,p,type) 列表。"""
    configs = []
    for N in ALL_N:
        for K in ALL_K:
            for T in ALL_T:
                for p in ALL_P:
                    for st, n in PER_CONFIG.items():
                        for _ in range(n):
                            configs.append((N, K, T, p, st))
    return configs  # 81 * 125 = 10125


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(description="网格世界数据生成")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_per_config", type=int, default=None,
                        help="覆盖每子配置场景数；不指定则用默认 PER_CONFIG")
    parser.add_argument("--seed", type=int, default=0)
    # 单配置覆盖（可选）
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--p_transfer", type=float, default=None)
    parser.add_argument("--scenario_type", type=str, default=None,
                        choices=["random", "goal_directed", "adversarial"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构建配置列表
    if args.N is not None:
        n_per = args.num_per_config or 125
        configs = [(args.N, args.K, args.T, args.p_transfer, args.scenario_type)] * n_per
    else:
        if args.num_per_config is not None:
            ratio = args.num_per_config / 125
            scaled = {k: max(1, int(v * ratio)) for k, v in PER_CONFIG.items()}
            configs = []
            for N in ALL_N:
                for K in ALL_K:
                    for T in ALL_T:
                        for p in ALL_P:
                            for st, n in scaled.items():
                                for _ in range(n):
                                    configs.append((N, K, T, p, st))
        else:
            configs = iter_all_configs()

    print(f"生成 {len(configs)} 个场景到 {out_dir}")
    for i, (N, K, T, p, st) in enumerate(tqdm(configs)):
        S_0, actions, trajectory, meta = generate_scenario(
            N, K, T, p, st, seed=args.seed + i
        )
        fname = f"scen_{N:02d}_{K}_{T:02d}_{int(p*10)}_{st}_{i:06d}.npz"
        np.savez(
            out_dir / fname,
            S_0=S_0,
            actions=actions,
            S_t=trajectory,
            N=N, K=K, T=T, p_transfer=p, scenario_type=st,
        )
    print("完成。")


if __name__ == "__main__":
    main()
