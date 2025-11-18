from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional  # 确保导入必要的类型提示模块
from PIL import Image
from rich.console import Console
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from typing import Optional, Dict, Any
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,roc_curve
from typing import Any
from ruamel.yaml import YAML
import io

yaml = YAML()

# 将 CommentedMap 转换为字符串
def yaml_to_string(data):
    stream = io.StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()

def crop_face_with_mtcnn(img:Image.Image) -> Image.Image:
    """数据与处理 检测人脸

    Args:
        img (Image.Image): 输入图片

    Returns:
        Image: 输出处理后的图片
    """
    # Detect face with MTCNN
    mtcnn = MTCNN(keep_all=True)
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        # 如果检测不到人脸，返回中心裁剪后的图像
        width, height = img.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        cropped = img.crop((left, top, right, bottom))
    else:
        # 假设每张图像只有一个主要人脸
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        size = int(max(w, h) * 1.3 / 2)
        cropped = img.crop((cx - size, cy - size, cx + size, cy + size))

    return cropped.resize((224, 224))

def l2_normalize(x, dim=1, eps=1e-8):
    """对张量按指定维度进行 L2 归一化"""
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)



def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8
) -> torch.Tensor:
    """监督对比损失函数（Supervised Contrastive Loss）

    计算基于标签信息的对比损失，用于模型训练中的特征区分任务。
    通过拉近相同标签样本的特征距离、推远不同标签样本的特征距离，
    使模型学习到更具判别力的特征表示。

    核心逻辑：
    1. 对输入特征进行 L2 归一化，消除尺度差异影响
    2. 计算特征间的相似度矩阵（除以温度系数调节分布平滑度）
    3. 基于标签构建正负样本掩码（排除自身对比）
    4. 数值稳定性处理（减去每行最大值）
    5. 计算负样本的归一化概率，最终得到对比损失


    Args:
        features: 输入特征张量，形状为 [N, D]，其中 N 为样本数，D 为特征维度
            （无需提前归一化，函数内部会处理）
        labels: 样本标签张量，形状为 [N]，数据类型为长整数（long），
            相同标签表示同类样本
        temperature: 温度系数，用于调节相似度分布的平滑程度，默认值为 0.07
            较小值会使分布更尖锐（区分度更强），较大值会使分布更平缓
        eps: 数值稳定性参数，用于避免除以零或对数运算中出现零输入，默认值为 1e-8

    Returns:
        torch.Tensor: 批次样本的平均监督对比损失，标量张量


        1. 输入特征会被自动进行 L2 归一化，无需外部预处理
        2. 掩码构建时会排除样本自身与自身的对比（self-contrast）
        3. 温度系数的选择会影响损失的梯度特性，建议根据任务调整（常见范围 0.01-0.5）
        4. 支持 CUDA 张量计算，设备会自动与输入特征保持一致
    """
    # 获取输入特征所在设备（CPU/GPU），确保后续计算在同一设备上进行
    device = features.device
    
    # 对特征进行 L2 归一化（按特征维度 D 归一化），消除尺度差异
    features = l2_normalize(features, dim=1)
    
    # 计算特征相似度矩阵：[N, D] @ [D, N] = [N, N]，除以温度系数调节平滑度
    logits = torch.div(torch.matmul(features, features.t()), temperature)
    
    # 调整标签形状为 [N, 1]，便于后续广播计算标签匹配掩码
    labels = labels.contiguous().view(-1, 1)
    
    # $$y=kx$$
    # 构建同类样本掩码：相同标签位置为 1，不同标签为 0（包含自身）
    mask = torch.eq(labels, labels.t()).float().to(device)
    
    # 构建自对比排除掩码：对角线位置（自身）设为 0，其他位置为 1
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
    
    # 更新同类样本掩码：排除自身与自身的对比（将对角线位置的 1 置为 0）
    mask = mask * logits_mask
    
    # 数值稳定性处理：减去每行最大值，避免指数运算时出现数值溢出
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()  # detach 避免梯度传播到 max 操作
    
    # 计算指数化相似度（仅保留非自身样本的贡献）
    exp_logits = torch.exp(logits) * logits_mask
    
    # 计算负样本的归一化对数概率：log(分子 / 分母) = log(分子) - log(分母)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)
    
    # 计算每个样本在同类样本上的平均对数似然（除以同类样本数量，加 eps 避免除零）
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)
    
    # 对比损失为负的平均对数似然（使同类样本的对数似然最大化）
    loss = -mean_log_prob_pos
    
    # 计算批次样本的平均损失（标量输出）
    loss = loss.mean()
    
    return loss

def calculate_metrics(y_true, y_scores):
    """
    计算多种评估指标，包括AUC、准确率等
    
    Args:
        y_true: 真实标签
        y_scores: 预测概率分数
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 计算AUC
    auc = roc_auc_score(y_true, y_scores)
    
    # 计算最佳阈值下的准确率
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 使用最优阈值进行预测
    y_pred = (y_scores >= optimal_threshold).astype(int)
    
    # 计算准确率
    accuracy = (y_pred == y_true).mean() * 100
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'optimal_threshold': optimal_threshold
    }

def plot_roc_curve(y_true, y_scores, epoch=None):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_scores: 预测概率分数
        epoch: 当前epoch数（可选）
    
    Returns:
        matplotlib.figure.Figure: ROC曲线图
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic (Epoch {epoch})' if epoch is not None else 'ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    return fig

def print_config(config: dict,text:Optional[str]=''):
    """
    使用rich美化输出嵌套字典格式的配置
    
    Args:
        config: 待输出的配置字典，支持嵌套结构和多种数据类型
        text: 需要补充的标题 ，后面会接上配置信息
    """
    console = Console()
    # 创建根节点（配置标题）
    root = Tree(Text(f"{text}配置信息", style="bold magenta"))
    
    def add_nodes(parent: Tree, data: Any, key: str = ""):
        """
        递归添加节点到树结构中，支持多种数据类型
        
        Args:
            parent: 父节点
            data: 待添加的数据（支持dict、list、str、int、float、bool、None）
            key: 数据对应的键名
        """
        # 根据数据类型处理不同的显示逻辑
        if isinstance(data, dict):
            # 嵌套字典：创建折叠节点
            if key:
                node = parent.add(Text(f"🔑 {key}", style="bold blue"))
            else:
                node = parent  # 根节点本身是字典时直接使用
            
            # 递归处理字典中的每个键值对
            for sub_key, sub_data in data.items():
                add_nodes(node, sub_data, sub_key)
        
        elif isinstance(data, list):
            # 列表：显示索引和元素
            node = parent.add(Text(f"🔑 {key}", style="bold blue") + Text(f" (列表, 长度: {len(data)})", style="italic yellow"))
            for idx, item in enumerate(data):
                # 列表元素显示索引
                add_nodes(node, item, f"[{idx}]")
        
        else:
            # 基础数据类型：根据类型设置不同颜色
            if isinstance(data, str):
                value_text = Text(f": {data!r}", style="green")  # 字符串：绿色，带引号
            elif isinstance(data, (int, float)):
                value_text = Text(f": {data}", style="cyan")  # 数字：青色
            elif isinstance(data, bool):
                # 布尔值：True绿色，False红色
                value_style = "bold green" if data else "bold red"
                value_text = Text(f": {str(data).upper()}", style=value_style)
            elif data is None:
                value_text = Text(": None", style="italic gray")  # None：灰色斜体
            else:
                # 其他类型：默认显示
                value_text = Text(f": {str(data)}", style="white")
            
            # 添加叶子节点（键值对）
            parent.add(Text(f"🔑 {key}", style="bold blue") + value_text)
    
    # 递归构建树结构
    add_nodes(root, config)
    
    # 使用面板包裹整个配置，提升视觉效果
    panel = Panel(
        root,
        title=Text("Config Printer", style="bold white on blue"),
        border_style="blue",
        expand=False  # 不自动扩展宽度，适应内容
    )
    
    # 输出到终端
    console.print(panel)


def print_model_summary(
    model: nn.Module,
    input_shape: Optional[tuple] = None,
    device: Optional[torch.device] = None,
    max_depth: int = 3,
    show_trainable: bool = True,
    show_output_shape: bool = True
) -> None:
    """
    使用 rich 在终端美观输出 PyTorch 模型结构 summary

    Args:
        model: 待输出的 PyTorch 模型实例（nn.Module）
        input_shape: 输入张量形状（不含 batch 维度），例如 (3, 224, 224)，用于计算输出形状
        device: 计算输出形状时使用的设备（默认自动检测 CPU/GPU）
        max_depth: 最大显示层级（避免嵌套过深导致输出冗长）
        show_trainable: 是否显示层的可训练状态
        show_output_shape: 是否显示层的输出形状（需提供 input_shape）
    """
    # 初始化 rich 控制台
    console = Console(width=120)
    # 简洁的标题分隔符（修复标签格式）
    console.print(f"\n[bold blue]=== Model Summary: {model.__class__.__name__} ===")

    # 设备自动检测
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 记录每层信息的列表
    layers_info = []
    total_params = 0
    trainable_params = 0

    # 递归遍历模型层（只统计叶子节点的参数量，避免重复计算）
    def _recursive_parse(
        module: nn.Module,
        name: str,
        depth: int = 0,
        parent_output_shape: Optional[tuple] = None
    ) -> None:
        nonlocal total_params, trainable_params

        # 超过最大深度则跳过
        if depth > max_depth:
            return

        # 判断是否为叶子节点（没有子模块的层），只统计叶子节点参数量避免重复
        has_children = len(list(module.named_children())) > 0
        params = sum(p.numel() for p in module.parameters()) if not has_children else 0
        trainable = any(p.requires_grad for p in module.parameters()) if not has_children else False
        
        # 只累加叶子节点的参数量
        if not has_children:
            total_params += params
            if trainable:
                trainable_params += params

        # 计算输出形状（如果提供了输入形状）
        output_shape = None
        if show_output_shape and input_shape is not None:
            try:
                # 构造虚拟输入（batch_size=1）
                dummy_input = torch.randn(1, *input_shape).to(device)
                module.to(device)
                
                # 临时设置模型为eval模式，避免影响训练状态
                is_training = module.training
                module.eval()
                
                with torch.no_grad():
                    # 处理Sequential等容器的输出形状传递
                    if parent_output_shape is None:
                        output = module(dummy_input)
                    else:
                        dummy_input = torch.randn(1, *parent_output_shape).to(device)
                        output = module(dummy_input)
                    
                    # 提取输出形状（去除batch维度）
                    if isinstance(output, (tuple, list)):
                        # 如果是模型整体输出（包含logits和repr），只取第一个输出的形状
                        if len(output) >= 1 and isinstance(output[0], torch.Tensor):
                            output_shape = tuple(output[0].shape[1:])
                        else:
                            output_shape = "N/A"
                    elif isinstance(output, torch.Tensor):
                        output_shape = tuple(output.shape[1:])
                    else:
                        output_shape = "N/A"
                
                # 恢复模型训练状态
                if is_training:
                    module.train()
            except Exception:
                output_shape = "N/A"

        # 添加当前层信息
        layers_info.append({
            "depth": depth,
            "name": name,
            "type": module.__class__.__name__,
            "params": params,
            "trainable": trainable,
            "output_shape": output_shape
        })

        # 递归处理子模块
        for child_name, child_module in module.named_children():
            # 构造子模块名称（层级分隔符用"."）
            child_full_name = f"{name}.{child_name}" if name else child_name
            # 传递当前层输出形状给子层
            _recursive_parse(child_module, child_full_name, depth + 1, output_shape)

    # 开始递归解析模型
    _recursive_parse(model, name="", parent_output_shape=None if input_shape else None)

    # 创建 rich 表格
    table = Table(show_header=True, header_style="bold blue", row_styles=["", "dim"], box=None)
    table.add_column("Layer (Depth)", width=30)
    table.add_column("Type", width=25)
    if show_output_shape:
        table.add_column("Output Shape", width=20)
    table.add_column("Params", width=15, justify="right")
    if show_trainable:
        table.add_column("Trainable", width=10)

    # 填充表格数据
    for info in layers_info:
        # 层名称（根据深度添加缩进）
        indent = "  " * info["depth"]
        layer_name = Text(f"{indent}{info['name']}" if info['name'] else f"{indent}[Root]", style="green")
        
        # 层类型
        layer_type = Text(info["type"], style="yellow")
        
        # 参数量（格式化显示，如 1.2M、3.4K）
        def format_params(num: int) -> str:
            if num >= 1e6:
                return f"{num / 1e6:.2f}M"
            elif num >= 1e3:
                return f"{num / 1e3:.2f}K"
            return str(num) if num != 0 else "0"
        
        params_text = Text(format_params(info["params"]), style="cyan")
        
        # 可训练状态
        trainable_text = Text("✅" if info["trainable"] else "❌", 
                           style="green" if info["trainable"] else "red") if info["params"] > 0 else Text("-", style="gray")
        
        # 输出形状
        output_shape_text = Text(str(info["output_shape"]), style="purple") if show_output_shape else None

        # 添加行到表格
        row = [layer_name, layer_type]
        if show_output_shape:
            row.append(output_shape_text)
        row.append(params_text)
        if show_trainable:
            row.append(trainable_text)
        table.add_row(*row)

    # 打印表格
    console.print(table)

    # 打印模型统计信息（修复标签格式：闭合所有样式标签）
    console.print(f"\n[bold blue]=== Model Statistics ===")
    console.print(f"[bold]Total Parameters:[/bold] {format_params(total_params)}")
    console.print(f"[bold]Trainable Parameters:[/bold] {format_params(trainable_params)}")
    console.print(f"[bold]Non-trainable Parameters:[/bold] {format_params(total_params - trainable_params)}")
    console.print(f"[bold]Trainable Ratio:[/bold] {trainable_params / total_params:.2%}" if total_params > 0 else "0.00%")
    # 修复：使用正确的标签闭合（同时移除多余的分隔符，保持简洁）
    console.print("\n")
