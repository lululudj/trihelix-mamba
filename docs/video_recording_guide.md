# Demo 演示视频录制指南

> ThreeChainMamba3 项目演示视频录制脚本与旁白词
> 按照本指南录制可生成完整的演示视频（约15-20分钟）

---

## 录制准备

### 工具推荐
- **OBS Studio**（免费开源）：https://obsproject.com
- **录屏设置**：1920×1080, 30fps, MP4格式
- **麦克风**：建议使用降噪麦克风，语速适中

### 环境准备
```bash
# 确保以下脚本可运行
python _local_grad_test.py          # Demo 2 梯度验证
python _gen_screenshots.py          # 截图生成
# Demo 1 和 Demo 3 需要GPU/C500环境，可用截图替代
```

### 录制前检查
- [ ] 终端字体调大（至少16pt，确保视频清晰）
- [ ] 终端背景设为深色（宇宙深色风 #0a0e27）
- [ ] 关闭通知/弹窗（勿扰模式）
- [ ] 准备好项目仓库页面 https://www.gitlink.org.cn/lulululudj/ThreeChainMamba3

---

## 视频结构（4个章节，约15-20分钟）

| 章节 | 内容 | 时长 | 素材 |
|------|------|------|------|
| 1 | 项目总览与核心创新 | ~4分钟 | README + 架构图 |
| 2 | Demo 2: BP v2.1梯度验证 | ~3分钟 | 实时运行 _local_grad_test.py |
| 3 | C500国产GPU实验展示 | ~5分钟 | 截图 + 实验结果 |
| 4 | 实验结果与对比分析 | ~4分钟 | 图表 + 性能报告 |

---

## 第1章：项目总览与核心创新（~4分钟）

### 画面
1. 打开浏览器，访问 GitLink 仓库主页
2. 滚动浏览 README，停在架构图

### 旁白词

> 大家好，今天演示的是 ThreeChainMamba3——三链 DNA-Mamba3 长程时空外推模型。
>
> 这个项目的核心创新有三点：
>
> **第一，三链异构架构。** 受DNA三螺旋结构启发，我们设计了空间链、时间链、因果链三条Mamba3扫描链，分别从三个视角扫描同一张时空张量。空间链做双向扫描看全局结构，时间链做因果扫描看时序演化，因果链做agent间因果交互。
>
> **第二，Mamba3的dt-RoPE复数状态空间。** 我们使用官方Mamba3内核，它的旋转位置编码是基于时间步dt自适应累积的，不需要预设最大长度，天然支持任意长度外推。同时用梯形离散化替代了Mamba2的ZOH，大步长下误差更小。
>
> **第三，BP v2.1碱基对耦合。** 这是我们最新的修复——三对碱基对在链间做信息交换。之前v2版本有个乘法梯度死锁的bug，v2.1改成加法独立调制后彻底解决了。

### 画面切换
- 展示 `figures/fig2_ood_curve.png`（OOD外推曲线）
- 展示 `models/three_chain_mamba3.py` 代码片段

---

## 第2章：Demo 2 - BP v2.1梯度验证（~3分钟）

### 画面
1. 打开终端，cd到项目目录
2. 运行 `python _local_grad_test.py`

### 旁白词

> 现在演示BP v2.1梯度验证。这个Demo只需要PyTorch，不需要GPU，大约10秒就能跑完。
>
> **运行命令：** `python _local_grad_test.py`
>
> 大家看输出——训练前，所有7个调制网络的权重范数都是0，这是零初始化的设计，保证初始状态等价于baseline。
>
> **关键看梯度：** bp1_gate的梯度范数是19.87，bp2_reset是28.14，都是非零的！这说明梯度成功流过了这两个网络。
>
> 一步Adam更新后，权重从0增长到0.32——验证了v2.1加法修复确实让两个网络都能正常学习。
>
> **对比v2乘法版本：** bp1_gate和bp2_reset的梯度都是0.000000——这就是乘法梯度死锁。两个零初始化网络相乘，梯度链式法则含对方项（等于0），互相阻塞，永久死锁。
>
> v2.1改成加法后，`reset_b * x_t + sgate_b * x_t`，两项独立加到x_t上，各自梯度只依赖自身，不再互相阻塞。

### 画面
- 展示 `figures/screenshots/screenshot_demo2_gradtest.png`
- 展示 `figures/screenshots/screenshot_demo2_v2_deadlock.png`

---

## 第3章：C500国产GPU实验展示（~5分钟）

### 画面
1. 展示C500实验监控截图
2. 展示C500实验结果截图

### 旁白词

> 接下来展示在国产沐曦C500 GPU上的大规模实验。
>
> **硬件环境：** 沐曦MetaX C500，64GB显存，使用Triton-MXMACA编译后端。Mamba3的Triton内核在沐曦MXMACA上零修改运行，不需要改任何Python或Triton源码。
>
> **实验规模：** 32个场景，覆盖随机、目标导向、对抗、极端四类；每个场景跑3个seed；对比4个模型——Transformer、Mamba3单链、三链Mamba3、三链Mamba3+BP。总共384次训练。
>
> **实验过程：** 监控脚本每30分钟汇报进度。大家看截图——从4点19分开始，5点52分完成，192个结果，耗时约93分钟。整个过程0错误。
>
> **工程防护：** C500上有个rq_qos_wait内核死锁风险，我们通过每步`torch.cuda.synchronize()`+BATCH_SIZE=1+模型间sleep解决了。还遇到__pycache__缓存旧代码导致bp1_gate和bp2_reset全为0的问题——部署时强制删除所有__pycache__后修复。

### 画面
- 展示 `figures/screenshots/screenshot_c500_monitor.png`
- 展示 `figures/screenshots/screenshot_c500_results.png`

---

## 第4章：实验结果与对比分析（~4分钟）

### 画面
1. 展示性能对比表
2. 展示OOD外推曲线图
3. 展示BP按场景类型提升

### 旁白词

> 最后看实验结果。
>
> **主结果：** 三链Mamba3的ood_ch_acc是57.14%，加BP后提升到57.86%。同参数Transformer只有30.86%，Mamba3单链30.67%——三链架构相比单链有巨大提升，翻了将近一倍。
>
> **长程外推：** 训练T=100步，评估T=150步，外推50%。三链Mamba3的OOD decay只有0.41%，几乎不退化。而Transformer在OOD区直接塌缩成全猜0。
>
> **BP死锁修复验证：** 94个BP结果中，bp1_gate平均0.3512，bp2_reset平均0.3500，全部非零——v2.1加法修复彻底解决了死锁问题。对比v2版本的0.0000，效果立竿见影。
>
> **BP按场景提升：** BP在对抗场景提升+1.1%，在极端场景提升+1.9%——说明碱基对耦合在复杂多智能体交互场景中价值最大。简单场景提升较小（+0.3-0.4%），因为简单场景三链本身已经足够强。
>
> **总结：** ThreeChainMamba3在O(N)线性复杂度下实现了长程外推不退化，BP v2.1碱基对耦合在难场景有显著提升，全部代码在国产沐曦C500 GPU上零修改运行。感谢大家观看。

### 画面
- 展示 `figures/fig2_ood_curve.png`
- 展示 `figures/fig3_ood_decay.png`
- 展示 GitLink 仓库页面

---

## 录制后处理

### 视频剪辑
1. 每章开头添加标题卡（深色背景 + 章节名）
2. 画面切换时添加淡入淡出效果
3. 代码展示时适当放大字体
4. 添加背景音乐（可选，音量调低）

### 输出参数
- 格式：MP4 (H.264)
- 分辨率：1920×1080
- 帧率：30fps
- 码率：5-8 Mbps
- 时长：约15-20分钟

### 上传
1. 上传到B站/YouTube
2. 获取视频链接
3. 在README和GitLink发行版中填写视频链接

---

## 截图素材清单

以下截图已由 `_gen_screenshots.py` 自动生成到 `figures/screenshots/`：

| 截图文件 | 内容 | 用途 |
|----------|------|------|
| `screenshot_demo2_gradtest.png` | BP v2.1梯度验证终端输出 | Demo 2展示 |
| `screenshot_demo2_v2_deadlock.png` | v2乘法死锁对照 | Demo 2对比 |
| `screenshot_c500_results.png` | C500 384次实验结果表 | 实验结果展示 |
| `screenshot_c500_monitor.png` | C500实验监控进度 | 实验过程展示 |
| `screenshot_training.png` | 三链Mamba3训练+OOD评估 | Demo 1展示 |

## 图表素材清单

以下图表由 `regen_figures.py` 生成到 `figures/`：

| 图表文件 | 内容 | 用途 |
|----------|------|------|
| `fig1_training_dynamics.png` | 训练动态曲线 | 训练过程展示 |
| `fig2_ood_curve.png` | OOD长程外推主图 | 核心结果展示 |
| `fig3_ood_decay.png` | OOD衰减柱状图 | 衰减对比 |
| `fig4_param_efficiency.png` | 参数效率散点图 | 参数量对比 |
| `fig5_collapse_diagnosis.png` | 退化诊断 | 退化分析 |
| `fig6_seed_variance.png` | Seed方差箱线图 | 稳定性验证 |
| `fig7_ablation.png` | 消融实验 | BP消融对比 |
