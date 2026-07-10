# 闲鱼赛博数据库 - 大模型数据训练中心文件库设计文档

> 日期：2026-07-10
> 状态：设计阶段

## 1. 项目概述

### 1.1 目标

构建一个本地桌面应用（exe），让个人用户能够：
- 从 GitHub、HuggingFace（含国内镜像）和网络搜索采集训练数据
- 清洗、格式化成标准训练数据集（图像/音频/结构化数据）
- 自动搜索本地模型文件，选择模型进行训练
- 打包数据集为 ZIP 文件，在闲鱼平台贩卖
- 内置 Qwen 小模型作为 AI 助手，可交互对话、自主上网搜索、每日自动更新数据集

### 1.2 核心用户场景

1. 用户双击 exe -> 浏览器自动打开管理界面
2. 通过 AI 助手对话："帮我找猫狗分类图像数据" -> AI 上网搜索 -> 推荐数据集 -> 一键采集
3. 采集完成后 -> 数据自动清洗去重 -> 生成质量报告
4. 选择本地模型 -> 配置训练参数 -> 开始训练 -> 实时查看进度和 loss 曲线
5. 训练完成 -> 打包数据集为 ZIP -> 自动生成闲鱼文案和定价建议 -> 复制粘贴到闲鱼售卖
6. 每日 AI 自动上网搜索新数据集 -> 推送到仪表盘"今日发现"

### 1.3 非目标

- 不做在线交易平台（交付方式为文件打包发送）
- 不做云端服务（纯本地运行）
- 不做多用户协作（单用户桌面应用）

## 2. 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 后端 | Python 3.10+ / FastAPI | 异步 API，自带文档 |
| 前端 | Vue 3 + Vite + Element Plus | 单页应用，构建为静态文件 |
| 数据库 | SQLite | 元数据、任务记录、训练日志 |
| 文件存储 | 本地文件系统 | 数据集按标准目录结构组织 |
| AI 助手 | Ollama + Qwen 小模型 | 本地 LLM，支持 Function Calling |
| 打包 | PyInstaller | 打包为单个 exe，双击即用 |
| 训练 | PyTorch | 模型训练引擎 |
| 数据采集 | httpx / aiohttp | 异步 HTTP 请求 |

## 3. 系统架构

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                    Web 界面 (Vue.js)                      │
│  仪表盘 | 数据集库 | 采集任务 | 数据处理 | 模型管理        │
│  训练中心 | 打包导出 | 设置 | AI 助手聊天窗口              │
├──────────────────────────────────────────────────────────┤
│                    API 层 (FastAPI)                       │
│              REST API + WebSocket (AI 对话)               │
├───────┬───────┬───────┬───────┬───────┬─────────────────┤
│ 采集  │ 处理  │ 模型  │ 训练  │ 打包  │   AI Agent       │
│ 模块  │ 模块  │ 模块  │ 模块  │ 模块  │   (Qwen+工具)    │
├───────┴───────┴───────┴───────┴───────┴─────────────────┤
│  SQLite (元数据) + 本地文件系统 (数据/模型)                │
└──────────────────────────────────────────────────────────┘
```

### 3.2 启动流程

```
双击 xianyu_dataset_hub.exe
    -> 后台线程启动 FastAPI 服务器 (localhost:8000)
    -> 检测 Ollama 是否安装/运行
        -> 未安装：提示用户安装（首次引导）
        -> 已安装：加载 Qwen 小模型
    -> 自动打开默认浏览器到 localhost:8000
    -> 主线程保活，关闭进程时清理资源
```

### 3.3 项目目录结构（开发时）

```
xianyu_dataset_hub/
├── backend/
│   ├── api/                # API 路由
│   │   ├── datasets.py     # 数据集 CRUD
│   │   ├── collect.py      # 采集任务 API
│   │   ├── process.py      # 数据处理 API
│   │   ├── models.py       # 模型管理 API
│   │   ├── train.py        # 训练 API
│   │   ├── export.py       # 打包导出 API
│   │   └── chat.py         # AI 助手 WebSocket
│   ├── collectors/         # 数据采集器
│   │   ├── base.py         # 采集器基类
│   │   ├── github.py       # GitHub 采集器（含镜像）
│   │   ├── huggingface.py  # HuggingFace 采集器（含镜像）
│   │   └── web_search.py   # 网络搜索采集器
│   ├── processors/         # 数据处理
│   │   ├── dedup.py        # 去重
│   │   ├── validate.py     # 格式校验
│   │   ├── standardize.py  # 标准化
│   │   └── quality.py      # 质量评分
│   ├── model_manager/      # 模型管理
│   │   ├── scanner.py      # 本地模型扫描
│   │   ├── analyzer.py     # 模型分析
│   │   └── registry.py     # 模型注册表
│   ├── trainer/            # 训练引擎
│   │   ├── engine.py       # 训练引擎
│   │   ├── callbacks.py    # 训练回调（进度、日志）
│   │   └── schemas.py      # 训练配置模型
│   ├── packager/           # 打包导出
│   │   ├── export.py       # ZIP 打包
│   │   └── listing.py      # 闲鱼文案生成
│   ├── ai_agent/           # AI 助手
│   │   ├── agent.py        # Qwen Agent 主逻辑
│   │   ├── tools.py        # Function Calling 工具
│   │   ├── web_search.py   # AI 自主上网搜索
│   │   └── daily_update.py # 每日自动更新
│   ├── db/                 # 数据库
│   │   ├── models.py       # SQLAlchemy 模型
│   │   └── database.py     # 数据库连接
│   ├── config.py           # 配置管理
│   └── main.py             # FastAPI 应用入口
├── frontend/               # Vue.js 前端
│   ├── src/
│   │   ├── views/          # 页面组件
│   │   ├── components/     # 通用组件
│   │   ├── api/            # API 调用
│   │   └── stores/         # Pinia 状态管理
│   └── dist/               # 构建产物（打包进 exe）
├── datasets/               # 数据集存储（运行时生成）
├── exports/                # 导出 ZIP（运行时生成）
├── build.py                # PyInstaller 打包脚本
├── run.py                  # 开发启动脚本
└── requirements.txt
```

### 3.4 打包后用户看到的

```
xianyu_dataset_hub.exe      # 单个可执行文件
datasets/                   # 数据集存储（自动创建）
exports/                    # 导出文件（自动创建）
config.db                   # SQLite 数据库（自动创建）
```

## 4. 模块详细设计

### 4.1 数据采集模块

#### 4.1.1 采集器基类

```python
class BaseCollector:
    """采集器基类，定义统一接口"""
    
    async def search(self, query: str, data_type: str, limit: int) -> list[DatasetItem]:
        """搜索数据集"""
        pass
    
    async def download(self, item: DatasetItem, dest_path: str, mirror: str = None) -> bool:
        """下载数据集"""
        pass
    
    def get_metadata(self, item: DatasetItem) -> dict:
        """获取元数据"""
        pass
```

#### 4.1.2 国内镜像配置

| 数据源 | 官方 | 国内镜像 |
|--------|------|----------|
| GitHub | github.com | gitclone.com、kkgithub.com |
| GitHub 下载代理 | - | ghproxy.com |
| HuggingFace | huggingface.co | hf-mirror.com（设置 `HF_ENDPOINT`） |

采集流程带镜像自动回退：官方源 -> 镜像1 -> 镜像2 -> 标记失败。

#### 4.1.3 各采集器

- **GitHub 采集器**：GitHub API 搜索仓库，下载 release 附件、数据文件，通过镜像站加速
- **HuggingFace 采集器**：HF API / `datasets` 库搜索和下载，通过 hf-mirror.com 镜像
- **网络搜索采集器**：搜索引擎 + 网页爬取，采集图像/音频/表格数据

#### 4.1.4 采集任务管理

- Web 界面创建任务（选数据源、镜像、关键词、数据类型、数量）
- 后台异步执行，WebSocket 实时推送进度
- 采集结果自动入库，含来源 URL、许可证信息

### 4.2 数据处理模块

#### 4.2.1 处理流水线

```
原始数据 -> 去重 -> 格式校验 -> 标准化 -> 质量评分 -> 入库
```

- **去重**：文件哈希（SHA256）精确去重 + 图像 pHash 近似去重
- **格式校验**：图像完整性、音频可解码性、结构化数据格式正确性
- **标准化**：
  - 图像：统一格式（JPG/PNG），生成 labels.json
  - 音频：统一采样率/格式（WAV/MP3），生成 transcripts.json
  - 结构化：统一 CSV/Parquet，生成 schema.json
- **质量评分**：分辨率、清晰度、完整性，生成 quality_report.json

#### 4.2.2 数据集标准目录结构

```
datasets/{dataset_id}/
├── data/
│   ├── images/
│   ├── audio/
│   └── structured/
├── metadata.json        # 来源、格式、规模、许可证
├── labels.json          # 标签/标注
├── schema.json          # 数据模式描述
├── quality_report.json  # 质量报告
└── README.md            # 数据集说明
```

### 4.3 打包导出模块

#### 4.3.1 打包流程

```
选择数据集 -> 可选子集/筛选 -> 生成说明文档 -> 打包 ZIP -> 导出到 exports/
```

#### 4.3.2 导出包内容

```
dataset_name_v1.0.zip
├── data/
├── metadata.json
├── labels.json
├── schema.json
├── quality_report.json
├── README.md            # 人类可读说明（含使用示例）
├── LICENSE
└── train_sample.py      # PyTorch DataLoader 训练示例代码
```

#### 4.3.3 闲鱼售卖辅助

- 自动生成标题：`[N]条[数据类型]训练数据集 - [主题]`
- 自动生成商品描述（数据量、格式、用途、示例）
- 自动生成定价建议（基于数据量和质量评分）
- 一键复制文案到剪贴板

### 4.4 模型搜索与管理模块

#### 4.4.1 智能模型搜索

```
扫描本地文件 -> AI 分析文件名/路径/大小 -> 推断模型架构和用途
    -> 匹配模型注册表（ResNet, ViT, BERT, Qwen, LLaMA...）
    -> 生成模型描述
    -> 推荐适合的训练数据类型
```

- 扫描格式：.pt, .pth, .onnx, .safetensors, .bin, .ckpt, .h5
- 扫描路径：Downloads, Desktop, Documents, 用户自定义路径
- AI 辅助识别模型架构和用途

#### 4.4.2 模型管理

- 模型列表（文件名、路径、大小、格式、推断架构、描述）
- 手动添加模型路径
- 模型详情查看

### 4.5 训练模块

#### 4.5.1 训练流程

```
选择数据集 -> 选择模型 -> 配置训练参数 -> 开始训练 -> 实时监控 -> 保存模型
```

#### 4.5.2 训练参数

- 训练轮数 (epochs)
- 批次大小 (batch_size)
- 学习率 (learning_rate)
- 验证集比例
- GPU/CPU 选择
- 输出目录

#### 4.5.3 训练进度可视化

- 总进度条（Epoch 13/20）
- Batch 级进度条（450/800）
- 实时 Loss 值和趋势
- 实时准确率/指标
- Loss/准确率实时曲线图
- 预计剩余时间
- 暂停/停止按钮

#### 4.5.4 训练任务管理

- 训练任务列表（数据集、模型、状态、进度）
- 历史训练记录
- 训练结果对比

### 4.6 AI 助手模块

#### 4.6.1 AI 助手界面

- 右侧常驻聊天窗口
- 流式输出对话
- 可调用系统功能（Function Calling）

#### 4.6.2 AI Agent 工具集

```python
AI 可调用的工具：
- web_search(query)                  # 网络搜索
- github_search(query)               # GitHub 搜索
- hf_search(query)                   # HuggingFace 搜索
- create_collect_task(item)           # 创建采集任务
- analyze_dataset(path)              # 分析数据集
- recommend_params(model, dataset)    # 推荐训练参数
- generate_listing(dataset)          # 生成闲鱼文案
- scan_local_models(path)            # 扫描本地模型
```

#### 4.6.3 每日自动更新

```
每日定时任务（如 9:00）：
    -> AI 自动上网搜索热门/新发布的数据集
    -> 按用户兴趣偏好筛选
    -> 自动采集小样本预览
    -> 生成"今日发现"报告
    -> 推送到仪表盘，等用户确认入库
```

#### 4.6.4 技术实现

- Ollama 本地运行 Qwen 小模型（如 `qwen2.5:3b` 或 `qwen3:4b`）
- 首次启动检测并引导安装 Ollama + 拉取模型
- FastAPI WebSocket 实现实时对话流式输出
- Function Calling 实现 AI 调用系统操作

### 4.7 Web 界面模块

#### 4.7.1 界面布局

左侧导航栏 + 右侧主内容区 + 最右侧 AI 助手聊天窗口。

#### 4.7.2 页面清单

| 页面 | 功能 |
|------|------|
| 仪表盘 | 数据集总数、总数据量、最近任务、"今日发现" |
| 数据集库 | 网格/列表浏览，按类型/来源/质量筛选，详情预览 |
| 采集任务 | 新建任务、任务列表、进度监控 |
| 数据处理 | 选择数据集 -> 清洗/去重/标准化 -> 质量报告 |
| 模型管理 | 本地模型列表、扫描、添加、详情 |
| 训练中心 | 选择数据集+模型 -> 配置参数 -> 训练进度 -> 结果 |
| 打包导出 | 选择数据集 -> 打包 -> 闲鱼文案生成 -> 一键复制 |
| 设置 | 镜像配置、存储路径、超时/重试、Ollama 配置 |

#### 4.7.3 数据预览

- 图像：缩略图网格
- 音频：波形图 + 播放按钮
- 结构化：表格预览（前 100 行）

## 5. 数据流

### 5.1 数据采集流

```
用户/AI 发起采集
    -> 选择数据源 + 镜像
    -> 搜索 -> 下载（镜像回退）-> 原始数据暂存
    -> 去重 -> 校验 -> 标准化 -> 质量评分
    -> 写入 datasets/{id}/ + SQLite 元数据
    -> WebSocket 推送完成通知
```

### 5.2 模型训练流

```
用户选择数据集 + 模型
    -> 配置训练参数
    -> 加载数据集 -> 加载模型
    -> 训练循环（每个 batch/epoch 通过 WebSocket 推送进度）
    -> 保存训练后的模型
    -> 记录训练日志到 SQLite
```

### 5.3 打包导出流

```
用户选择数据集
    -> 可选筛选条件
    -> 生成 README/train_sample.py
    -> 打包 ZIP
    -> 生成闲鱼文案
    -> 导出到 exports/
```

## 6. 错误处理

| 场景 | 处理方式 |
|------|----------|
| 镜像站全部不可达 | 标记任务失败，提示用户检查网络或手动配置 |
| 下载中断 | 支持断点续传，记录已下载部分 |
| 数据格式损坏 | 跳过损坏文件，记录到质量报告 |
| Ollama 未安装 | 首次启动引导安装，AI 助手显示"未就绪"状态 |
| 模型加载失败 | 提示错误原因，不影响其他功能 |
| 训练 GPU 不可用 | 自动回退到 CPU，提示用户 |
| 端口 8000 被占用 | 自动尝试下一个可用端口（8001, 8002...） |
| SQLite 并发写入 | 使用 WAL 模式，`check_same_thread=False` |

## 7. 测试策略

- **单元测试**：各采集器、处理器、打包器的核心逻辑
- **集成测试**：采集 -> 处理 -> 打包完整流程
- **API 测试**：FastAPI 接口测试（pytest + httpx）
- **前端测试**：Vue 组件测试（Vitest）

## 8. 打包构建

### 8.1 构建步骤

1. 构建前端：`cd frontend && npm run build` -> 生成 `dist/`
2. PyInstaller 打包后端 + 前端静态文件：
   ```
   pyinstaller --onefile --windowed \
     --add-data "frontend/dist;frontend/dist" \
     --hidden-import uvicorn.logging \
     --hidden-import starlette.routing \
     --hidden-import starlette.exceptions \
     --hidden-import uvicorn.config \
     --hidden-import uvicorn.server \
     --hidden-import fastapi.utils \
     main.py
   ```

### 8.2 资源路径处理

```python
def resource_path(relative_path):
    """兼容 PyInstaller 打包后的资源路径"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
```

## 9. 依赖清单

```
# 后端
fastapi
uvicorn
sqlalchemy
httpx
aiohttp
pydantic
PyInstaller

# 数据处理
Pillow
pydub
pandas
pyarrow
imagehash

# AI
ollama
websockets

# 训练
torch
torchvision

# 前端
vue@3
vite
element-plus
axios
pinia
```
