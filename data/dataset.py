"""网格世界 Dataset 与 DataLoader

按 N 分桶采样，保证每个 batch 内网格尺寸一致。
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class GridWorldDataset(Dataset):
    """加载 .npz 场景数据集。

    返回 dict:
        S_0:        (N, N) long
        actions:    (K, T) long
        S_t:        (T+1, N, N) long
        N, K, T, p_transfer, scenario_type: 元数据
    """

    def __init__(self, root, max_T=None):
        self.files = sorted(Path(root).glob("scen_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No scen_*.npz under {root}")
        # 预读每个样本的 N, K, T 用于分组采样（保证 batch 内 N、K、T 一致）
        self.Ns = []
        self.Ks = []
        self.Ts = []
        for f in self.files:
            with np.load(f, allow_pickle=True) as d:
                self.Ns.append(int(d["N"]))
                self.Ks.append(int(d["K"]))
                T = int(d["T"])
                if max_T is not None:
                    T = min(T, max_T)
                self.Ts.append(T)
        self.max_T = max_T  # 截断到 max_T 步（用于 OOD 长程泛化测试）

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        S_0 = torch.from_numpy(d["S_0"]).long()
        actions = torch.from_numpy(d["actions"]).long()
        S_t = torch.from_numpy(d["S_t"]).long()
        T = int(d["T"])
        if self.max_T is not None and T > self.max_T:
            actions = actions[:, :self.max_T]
            S_t = S_t[:self.max_T + 1]
            T = self.max_T
        return {
            "S_0": S_0,
            "actions": actions,
            "S_t": S_t,
            "N": int(d["N"]),
            "K": int(d["K"]),
            "T": T,
            "p_transfer": float(d["p_transfer"]),
            "scenario_type": str(d["scenario_type"]),
        }


class GroupedByNSampler(Sampler):
    """按 (N, K, T) 分组的 batch 采样器：每个 batch 内 N、K、T 均相同。

    保证 collate_fn 的 torch.stack 不会因维度不一致而失败。
    """

    def __init__(self, Ns, batch_size, shuffle=True, seed=0, Ts=None, Ks=None):
        self.Ns = Ns
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.groups = defaultdict(list)
        for i, n in enumerate(Ns):
            parts = [n]
            if Ks is not None:
                parts.append(Ks[i])
            if Ts is not None:
                parts.append(Ts[i])
            self.groups[tuple(parts)].append(i)

    def __iter__(self):
        all_batches = []
        for n, indices in self.groups.items():
            indices = list(indices)
            if self.shuffle:
                self.rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                all_batches.append(indices[i:i + self.batch_size])
        if self.shuffle:
            self.rng.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.groups.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def collate_fn(batch):
    """堆叠 batch（同 N 已由 sampler 保证）。"""
    out = {
        "S_0": torch.stack([b["S_0"] for b in batch]),       # (B, N, N)
        "actions": torch.stack([b["actions"] for b in batch]),  # (B, K, T)
        "S_t": torch.stack([b["S_t"] for b in batch]),        # (B, T+1, N, N)
        "N": batch[0]["N"],
        "K": batch[0]["K"],
        "T": batch[0]["T"],
        "p_transfer": batch[0]["p_transfer"],
        "scenario_type": batch[0]["scenario_type"],
    }
    return out


def make_loaders(data_root, batch_size=64, seed=0, max_T=None):
    """构建 train/val/test DataLoader。

    假设数据已分成 train/val/test 三个子目录。
    """
    root = Path(data_root)
    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        ds = GridWorldDataset(split_dir, max_T=max_T)
        sampler = GroupedByNSampler(
            ds.Ns, batch_size=batch_size,
            shuffle=(split == "train"), seed=seed,
            Ts=ds.Ts, Ks=ds.Ks,
        )
        loaders[split] = DataLoader(
            ds, batch_sampler=sampler,
            collate_fn=collate_fn, num_workers=0,
        )
    return loaders


def _link_or_copy(src_file, dst_file):
    """跨平台链接文件：symlink → hardlink → copy 逐级 fallback。"""
    import os
    import shutil
    src_resolved = str(src_file.resolve())
    dst_str = str(dst_file)
    # 1) symlink（Linux/macOS 或 Windows 开发者模式）
    try:
        os.symlink(src_resolved, dst_str)
        return
    except (OSError, NotImplementedError):
        pass
    # 2) hardlink（同卷，Windows 普通用户可用）
    try:
        os.link(src_resolved, dst_str)
        return
    except (OSError, NotImplementedError):
        pass
    # 3) copy 兜底
    shutil.copy2(src_file, dst_file)


def split_dataset(src_dir, dst_root, ratios=(0.8, 0.1, 0.1), seed=0):
    """把 src_dir 下所有 .npz 按 ratios 划分到 dst_root/{train,val,test}。

    跨平台：symlink → hardlink → copy 逐级 fallback。
    """
    src = Path(src_dir)
    dst = Path(dst_root)
    rng = np.random.default_rng(seed)
    files = sorted(src.glob("scen_*.npz"))
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }
    for split, fs in splits.items():
        out_dir = dst / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in fs:
            link = out_dir / f.name
            if link.exists():
                continue
            _link_or_copy(f, link)
    print(f"划分完成: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
