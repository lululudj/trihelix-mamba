"""共享工具：device / seed / LR schedule / τ 退火 / checkpoint / 日志 / config

被 train.py 和 eval.py 复用。
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------- 基础 ----------

def set_seed(seed):
    """设置所有随机种子（python / numpy / torch）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """自动检测 device。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params(model):
    """可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(path):
    """加载 YAML 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(name, cfg):
    """从 MODEL_REGISTRY 构建模型。

    所有模型统一接受 (cell_types, action_dim, d_model, n_layers)。
    three_chain_mamba2 额外接受 enable_m3/enable_jepa 开关（阶段 A）。
    """
    from models import MODEL_REGISTRY  # 延迟 import
    cls = MODEL_REGISTRY[name]
    kwargs = dict(
        cell_types=cfg["task"]["cell_types"],
        action_dim=cfg["task"]["action_dim"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
    )
    # 阶段 A：给 three_chain_mamba2 / three_chain_mamba3 传 .m3/JEPA 开关 + jepa_weight
    # （两模型同接口），其他模型不接这些参数
    if name in ("three_chain_mamba2", "three_chain_mamba2_hta", "three_chain_mamba3"):
        m = cfg.get("model", {})
        kwargs["enable_m3"] = m.get("enable_m3", False)
        kwargs["enable_jepa"] = m.get("enable_jepa", False)
        kwargs["jepa_weight"] = m.get("jepa_weight", 0.3)
        # 极限测试: max_T 可从 config 覆盖 (默认 256, 评估 T>256 时需提升)
        kwargs["max_T"] = m.get("max_T", 256)
        # C 方案: gradient checkpointing (默认关, 1B 模型时开)
        kwargs["use_checkpoint"] = m.get("use_checkpoint", False)
        # 阶段 3.2: 消融开关（置零某链输出，保持参数量不变，严格控制变量）
        kwargs["ablate_s"] = m.get("ablate_s", False)
        kwargs["ablate_t"] = m.get("ablate_t", False)
        kwargs["ablate_c"] = m.get("ablate_c", False)
        # three_chain_mamba2_hta: 白嫖 Mamba3 heavy_tail_activation (A 激活 exp→heavy_tail)
        if name == "three_chain_mamba2_hta":
            kwargs["use_hta"] = True
    return cls(**kwargs)


# ---------- 调度 ----------

def cosine_lr(step, max_steps, lr, warmup):
    """cosine 退火 LR：warmup 线性升 → cosine 衰减。"""
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def eagle_tau_schedule(step, tau_init, tau_target, warmup, anneal):
    """Eagle τ 课程学习退火：warmup 前 tau_init，warmup→warmup+anneal 线性退火到 tau_target。"""
    tau_init = float(tau_init)
    tau_target = float(tau_target)
    warmup = int(warmup)
    anneal = int(anneal)
    if step < warmup:
        return tau_init
    if step >= warmup + anneal:
        return tau_target
    progress = (step - warmup) / max(1, anneal)
    return tau_init + (tau_target - tau_init) * progress


# ---------- checkpoint / 日志 ----------

def save_checkpoint(state, path):
    """保存 checkpoint（原子写入）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    """加载 checkpoint。返回 state。"""
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


class Logger:
    """JSONL 日志写入器（每行一条 JSON 记录）。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")  # 清空旧文件

    def log(self, record):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- 指标计算 ----------

def cell_accuracy(logits, S_t):
    """cell 级准确率。

    logits: (B, T, N, N, C)
    S_t:    (B, T+1, N, N)  完整轨迹（含 S_0）
    返回: (acc_final, acc_mean_over_T)
    """
    pred = logits.argmax(dim=-1)            # (B, T, N, N)
    target = S_t[:, 1:]                      # (B, T, N, N)
    correct = (pred == target).float()
    acc_per_step = correct.mean(dim=(0, 2, 3))  # (T,)
    return float(acc_per_step[-1]), acc_per_step.cpu().numpy()
