
# RINE-Opt: Deepfake Detection Framework

<div align="center">
  
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/yourusername/rine-opt)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

RINE-Opt 是一个基于先进视觉 Transformer 架构的深度伪造检测框架，具有多分支特征提取和注意力机制。

## 📋 目录

- [功能特性](#-功能特性)
- [安装](#-安装)
- [数据集准备](#-数据集准备)
- [训练](#-训练)
- [评估](#-评估)
- [项目结构](#-项目结构)
- [配置](#-配置)
- [贡献](#-贡献)
- [许可证](#-许可证)

## 🌟 功能特性

- **先进架构**：结合 CLIP 视觉 Transformer 和自定义 SSCA（空间和通道注意力）模块
- **多分支处理**：同时处理高层语义特征和中层空间特征
- **层选择**：灵活选择用于特征提取的 Transformer 层
- **注意力可视化**：内置注意力图可视化工具
- **全面日志**：TensorBoard 集成用于监控训练进度
- **可配置流水线**：基于 YAML 的实验配置系统

## 🛠️ 安装

### 先决条件

- Python 3.10+
- CUDA 兼容 GPU（推荐）
- platform: Linux

### 安装依赖

> 使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

或者手动安装依赖：

```bash
pip install torch torchvision transformers ruamel.yaml loguru pillow matplotlib seaborn scikit-learn pandas numpy
```
> 使用uv安装依赖(推荐)
```bash
uv sync
```
或者

```bash
uv venv env --python 3.10
source env/bin/activate
uv pip install -r requirements.txt
```


## 📁 数据集准备

### 数据集结构

预期的数据集结构如下：

```
dataset/
├── Celeb-DF/
│   ├── 00000/
│   │   ├── 00000_frame000000.jpg
│   │   ├── 00000_frame000010.jpg
│   │   └── ...
│   ├── 00001/
│   │   ├── 00001_frame000000.jpg
│   │   ├── 00001_frame000010.jpg
│   │   └── ...
│   └── labels.csv
└── ...
```

每个数据集文件夹包含视频的子文件夹，每个子文件夹包含从相应视频中提取的帧。`labels.csv` 文件包含每帧的标签。

### 使用 video2image.py 构建数据集

使用提供的 `video2image.py` 工具构建数据集：

```bash
python tools/video2image.py --input_dir /path/to/videos --output_dir dataset/Celeb-DF --frame_rate 10
```

该脚本将：
1. 以指定帧率从视频中提取帧
2. 按视频将帧组织到子目录中
3. 生成包含路径和标签的 labels.csv 文件

示例 labels.csv 格式：
```csv
path,label,video,frame
dataset/Celeb-DF/00000/00000_frame000000.jpg,0,00000,0
dataset/Celeb-DF/00000/00000_frame000010.jpg,0,00000,10
...
```

### 数据集格式

- 图像应为 JPG 格式
- 推荐尺寸：224x224 像素
- 标签格式：0 表示真实，1 表示伪造
- 帧命名约定：`{video_id}_frame{frame_number:06d}.jpg`

## 🏃 训练

要训练模型，运行：

```bash
python train.py
```

### 配置

训练前，在 `config/config.yaml` 中配置实验：

```yaml
dataset_root: "./dataset"
dataset_path: ["Celeb-DF"]
model_save_path: "checkpoint"

transform:
  resize: [224, 224]
  normalize: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
```

在 `config/train_config.yaml` 中配置训练参数：

```yaml
batch_size: 16
learning_rate: 0.001
epochs: 50
dataset_split: 0.8
optimizer: "adam"
loss_function: "bce_with_logits"
```

### 使用 TensorBoard 监控

训练期间，日志保存在 `runs/` 目录中。使用 TensorBoard 监控训练进度：

```bash
tensorboard --logdir runs
```

您可以查看：
- 训练和验证损失曲线
- 准确率指标
- 模型架构图
- 配置参数

## 🧪 评估

使用以下命令评估训练好的模型：

```bash
python test.py
```

这将输出：
- 准确率分数
- 精确率、召回率和 F1 分数
- 混淆矩阵
- ROC 曲线和 AUC 分数

## 🗂️ 项目结构

```
rine-opt/
├── config/
│   ├── config.yaml          # 基础配置
│   └── train_config.yaml    # 训练配置
├── model/
│   ├── FullModel.py         # 主模型架构
│   ├── SSCA.py              # 频谱-空间协同注意力模块
│   ├── DCTConvBlock.py      # DCT 卷积块
│   ├── GateMLP.py           # 门控 MLP 模块
│   └── CrossAttention.py    # 交叉注意力组合模块
├── tools/
│   ├── dataset_loader.py    # 数据集加载工具
│   ├── image_preprocess.py  # 图像预处理函数
│   ├── trainer.py           # 训练循环实现
│   ├── validation.py        # 验证工具
│   ├── utils.py             # 通用工具函数
│   ├── heatmap.py           # 注意力可视化工具
│   └── video2image.py       # 视频到图像数据集构建器
├── checkpoint/              # 模型检查点
├── runs/                    # TensorBoard 日志
├── dataset/                 # 数据集目录
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── requirements.txt         # Python 依赖
└── README.md               # 本文件
```

## ⚙️ 配置

### 基础配置 (`config/config.yaml`)

| 参数 | 描述 | 默认值 |
|------|------|--------|
| [dataset_root](file:///home/liangshuqiao/hong/deepfake/rine_opt/test.py#L22-L22) | 数据集根目录 | `"./dataset"` |
| `dataset_path` | 要使用的数据集名称列表 | `["Celeb-DF"]` |
| `model_save_path` | 保存模型检查点的目录 | `"checkpoint"` |

### 训练配置 (`config/train_config.yaml`)

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `batch_size` | 每批样本数 | `16` |
| `learning_rate` | 优化器学习率 | `0.001` |
| `epochs` | 训练轮数 | `50` |
| `dataset_split` | 训练/验证分割比例 | `0.8` |
| `optimizer` | 优化器类型 | `"adam"` |
| `loss_function` | 损失函数 | `"bce_with_logits"` |

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 仓库
2. 创建新分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建新的 Pull Request

## 📄 许可证

该项目基于 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。
