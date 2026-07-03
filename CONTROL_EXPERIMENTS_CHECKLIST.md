# 控制实验清单 (Control Experiments Checklist)

## 背景: 当前结论的危险

经过 V1/V2 极限测试后的自我审查, 发现 3 处结论站不住:

1. **单 seed 过度解读**: 1B、300M、700M 等关键节点都只有 1 个 seed, 而 100M 多 seed 实测方差 1.7% 大于相邻规模的 decay 差值
2. **B vs B2 训练步数混淆**: B (n_layers=4) 跑 500 步得 +4% decay, B2 (n_layers=8) 跑 1000 步得 -0.76% decay, 训练步数不同导致无法干净比较深度影响
3. **decay 指标可能测的是噪声**: 所有充分训练的模型 decay 都在 ±1% 附近, 没有验证这个指标是否真的能区分"泛化"和"过拟合到训练分布"

## 能站得住的结论 (无需补实验)

> **ThreeChainMamba2 在 30M 到 1.13B 范围内, 没有出现 Transformer-tiny 那样的全面崩盘 (zero_ratio=1.0)**

这条稳, 因为是二元判断 (崩/不崩), 对照组 Transformer-tiny 完全退化, 所有 SSM 规模 zero_ratio < 0.90, changed_acc > 0.30。

## 需要补的控制实验 (按优先级)

---

### 实验 1: B (n_layers=4) 长训对照 (最高优先级)

**目的**: 消除 B vs B2 的训练步数混淆变量
**针对危险**: #2 (B vs B2 confound)

**做法**: 用 B2 完全相同的设置 (1000 步, lr=3e-4, warmup=500), 只把 n_layers 从 8 改回 4

**命令**:
```bash
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_30m_deep_maxT.yaml \
    --max_steps 1000 --batch_size 2 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_30m_deep_n4_seed0_1000
# eval T150, T300
for T in 150 300; do
    python -u eval_ood.py \
        --checkpoint results/run_30m_deep_n4_seed0_1000/final.pt \
        --config configs/matched_mamba2_30m_deep_maxT.yaml \
        --data_root ./data/ood_T${T} --batch_size 1 --seed 0 \
        --extend_max_T 1024 \
        --out results/run_30m_deep_n4_seed0_1000/ood_A_T${T}.json
done
```

**判定标准**:
- 若 B@1000 仍得 +4% decay → "n_layers=4 有特殊长程优势" 真实, 需深究原因
- 若 B@1000 回归 ±1% decay → B 的 +4% 是欠训练副产物, **删掉"越深越强"叙事**

**时间**: ~12 min (训练 6 min + 2 eval 6 min)

---

### 实验 2: 1B 多 seed (高优先级)

**目的**: 验证 1B "decay=+0.28%" 不是单 seed 抽到的幸运值
**针对危险**: #1 (单 seed 过度解读)

**做法**: 用 C2 完全相同设置, 跑 seed=1, seed=2

**命令**:
```bash
for SEED in 1 2; do
    python -u train.py \
        --model three_chain_mamba2 \
        --config configs/matched_mamba2_1000m_ckpt_long.yaml \
        --max_steps 2000 --batch_size 1 --seed $SEED \
        --data_root ./data_split_m3 \
        --out_dir results/run_1000m_ckpt_long_seed${SEED}_2000
    python -u eval_ood.py \
        --checkpoint results/run_1000m_ckpt_long_seed${SEED}_2000/final.pt \
        --config configs/matched_mamba2_1000m_ckpt_long.yaml \
        --data_root ./data/ood_T150 --batch_size 1 --seed $SEED \
        --out results/run_1000m_ckpt_long_seed${SEED}_2000/ood_metrics.json
    rm -f results/run_1000m_ckpt_long_seed${SEED}_2000/final.pt
    rm -f results/run_1000m_ckpt_long_seed${SEED}_2000/best.pt
done
```

**判定标准**:
- 3 seeds decay 全在 ±2% 内 → "1B 不退化" 成立
- 有 seed decay > 5% 或 < -5% → 1B 训练不稳定, 之前 +0.28% 是幸运
- 方差 > 3% → 1B 不能下定量结论, 只能说"没崩盘"

**时间**: ~95 min (2 × 47 min)

---

### 实验 3: 随机标签 sanity check (高优先级)

**目的**: 验证 decay 指标本身是否有意义 (能否区分泛化 vs 过拟合)
**针对危险**: #3 (decay 测噪声)

**做法**: 训练一个小模型 (30M) 在随机打乱的标签上, 看 decay 是否还接近 0

**需改代码** (data/dataset.py 加一个 `shuffle_labels` 选项):
```python
# 在 dataset 返回前, 把 S_t 在 batch 维度打乱 (S_0 和 actions 不动)
if self.shuffle_labels:
    perm = torch.randperm(S_t.size(0))
    S_t = S_t[perm]  # 每个 sample 的 S_0/actions 配别人的 S_t
```

**命令**:
```bash
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_30m.yaml \
    --max_steps 500 --batch_size 4 --seed 0 \
    --data_root ./data_split_m3_shuffle \
    --out_dir results/run_30m_shufflelabel_seed0_500
python -u eval_ood.py \
    --checkpoint results/run_30m_shufflelabel_seed0_500/final.pt \
    --config configs/matched_mamba2_30m.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
    --out results/run_30m_shufflelabel_seed0_500/ood_metrics.json
```

**判定标准**:
- 随机标签模型 ch_acc ≈ 0 (无法学习随机映射) → 指标有效, 之前的 decay 是真泛化信号
- 随机标签模型 ch_acc > 0.4 且 decay ≈ 0% → **指标坏了**, decay 不能区分泛化/噪声, 所有定量结论作废
- 随机标签模型 ch_acc > 0.4 且 decay 大 (如 -20%) → 指标部分有效, 但 decay 的绝对值解读要谨慎

**时间**: ~8 min (训练 5 min + eval 3 min)

---

### 实验 4: 100M 长训饱和检查 (中优先级)

**目的**: 判断任务是否在 ch_acc=0.59 处饱和 (若是, 则 1B 的 0.579 不是欠训练)
**针对危险**: 1B val ch_acc (0.579) 低于 100M (0.59) 的反常现象

**做法**: 100M 跑 2000 步 (vs 通常 500), 看 val ch_acc 能否超过 0.59

**命令**:
```bash
python -u train.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2_100m.yaml \
    --max_steps 2000 --batch_size 4 --seed 0 \
    --data_root ./data_split_m3 \
    --out_dir results/run_100m_seed0_2000
python -u eval_ood.py \
    --checkpoint results/run_100m_seed0_2000/final.pt \
    --config configs/matched_mamba2_100m.yaml \
    --data_root ./data/ood_T150 --batch_size 1 --seed 0 \
    --out results/run_100m_seed0_2000/ood_metrics.json
```

**判定标准**:
- 100M@2000 val ch_acc > 0.62 → 任务未饱和, 1B 的 0.579 是欠训练, "1B 不退化"结论弱化
- 100M@2000 val ch_acc ≤ 0.60 → 任务在 0.59 饱和, 1B 的 0.579 合理, "1B 不退化"成立

**时间**: ~15 min (训练 10 min + eval 5 min)

---

## 不需要补的 (已稳)

- **gradient checkpointing 工程**: 1B 训练 OOM 突破是工程事实, 不依赖 decay 结论
- **Transformer 对照**: Transformer-tiny zero_ratio=1.0 是已验证的对照
- **Mamba3 算法退化**: 已多实现验证 (3rd-party + 官方 Triton)

## 总时间预算

| 实验 | 时间 | 优先级 |
|------|------|--------|
| 1. B@1000 步对照 | 12 min | 必须 |
| 2. 1B 多 seed (2颗) | 95 min | 必须 |
| 3. 随机标签 sanity | 8 min + 改代码 | 必须 |
| 4. 100M@2000 饱和 | 15 min | 可选 |
| **合计** | **~130 min** | |

下次开机 2-3 小时即可全部跑完。跑完后:
- 实验 1 + 2 通过 → "SSM 缩放不退化" 可对外定量说
- 实验 3 通过 → decay 指标有效, 定量结论有意义
- 任一不通过 → 回到只说"没崩盘"的定性结论, 不勉强

## 决策原则

**实验 3 (随机标签) 是一票否决**: 如果它失败, 所有 decay 数值结论作废, 不管实验 1/2 结果如何。所以实验 3 应该**第一个跑**, 成本最低 (8 min), 但信息量最大。
