import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import warnings
import matplotlib

warnings.filterwarnings("ignore")

class ModelValidator:
    def __init__(self, dataloader: DataLoader, 
                 class_names: List[str], 
                 chechpoint_info:dict,
                 model=None, 
                 model_path: Optional[str] = None, 
                 log_dir: str = 'runs/validation',
                 ):
        """
        初始化模型验证器
        
        Args:
            dataloader: 验证数据的 DataLoader
            class_names: 类别名称列表
            model: 已经初始化的模型实例（可选）
            model_path: 模型文件路径（可选，如果提供了model则不需要）
            log_dir: tensorboard日志保存路径
        """
        self.checkpoint_info = chechpoint_info
        self.name = self.checkpoint_info.get('name')
        data_name_value = self.checkpoint_info.get('dataset')
        if isinstance(data_name_value, list):
            self.data_name = " | ".join(str(item) for item in data_name_value)
        elif data_name_value is None:
            self.data_name = "Unknown Dataset"
        else:
            self.data_name = str(data_name_value)

        self.dataloader = dataloader    
        self.class_names = class_names
        self.model_path = model_path
        self.log_dir = log_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 设置模型
        if model is not None:
            self.model = model
            self.model.to(self.device)
            self.model.eval()
        elif model_path is not None:
            self.load_model()
        else:
            raise ValueError("必须提供model实例或model_path")
        
    def load_model(self):
        """加载训练好的模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理不同类型的checkpoint
        if isinstance(checkpoint, dict):
            # 尝试多种可能的键名来获取模型状态字典
            state_dict = None
            model_keys = ['model_state_dict', 'state_dict', 'model', 'net']
            
            for key in model_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            
            if state_dict is not None:
                # 如果已经有模型实例，直接加载权重
                if hasattr(self, 'model') and self.model is not None:
                    self.model.load_state_dict(state_dict)
                else:
                    # 如果没有模型实例，创建一个占位符模型
                    self.model = torch.nn.Module()  # 这只是一个占位符
                    try:
                        self.model.load_state_dict(state_dict)
                    except Exception as e:
                        logger.warning(f"无法直接加载状态字典: {e}")
                        # 在这种情况下，我们只能使用状态字典本身
                        self.model = state_dict
            else:
                # 如果checkpoint中没有明显的状态字典，尝试直接使用模型
                self.model = checkpoint.get('model', checkpoint)
        else:
            # 直接加载模型对象
            self.model = checkpoint
        
        # 确保模型在正确的设备上并处于评估模式
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()
            
    def validate(self) -> Dict[str, float]:
        """
        执行模型验证并计算各项指标
        
        Returns:
            包含各项指标的字典
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        # 添加进度条
        progress_bar = tqdm(self.dataloader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                outputs = outputs[0]
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()  # 使用0.5作为阈值

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # 更新进度条描述，显示当前批次的一些信息（可选）
                progress_bar.set_postfix({
                    'Batch Size': inputs.size(0)
                })
        
        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算各项指标
        metrics = {}
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['precision'] = precision_score(all_labels, all_preds, average='weighted')
        metrics['recall'] = recall_score(all_labels, all_preds, average='weighted')
        metrics['f1_score'] = f1_score(all_labels, all_preds, average='weighted')
        
        # 对于二分类任务计算AUC
        if len(self.class_names) == 2:
            # 修复：对于二分类，如果只有一个概率值（正类概率），直接使用
            if all_probs.ndim == 1:
                metrics['auc'] = roc_auc_score(all_labels, all_probs)
            else:
                metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # 多分类AUC
            try:
                metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            except:
                metrics['auc'] = 0.0
        
        # 计算错误率
        metrics['error_rate'] = 1 - metrics['accuracy']
        
        return metrics, all_labels, all_preds, all_probs # type:ignore
    
    def plot_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> plt.Figure: # type:ignore
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix \nmodel:{self.name}\ndataset:{self.data_name}',
                     pad=15)    # BUG 可能出现title显示不正常
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, labels: np.ndarray, probs: np.ndarray) -> plt.Figure: # type:ignore
        """绘制ROC曲线"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 处理二分类情况
        if len(self.class_names) == 2:
            if probs.ndim == 1:
                # 如果probs是一维的，直接使用
                fpr, tpr, _ = roc_curve(labels, probs)
                auc_score = roc_auc_score(labels, probs)
            else:
                # 如果probs是二维的，使用第二列（正类）
                fpr, tpr, _ = roc_curve(labels, probs[:, 1])
                auc_score = roc_auc_score(labels, probs[:, 1])
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {auc_score:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Receiver Operating Characteristic (ROC) Curve\nmodel:{self.name}\ndataset:{self.data_name}',
                         pad=15,
                         )  # TODO  这个也添加元信息
            ax.legend(loc="lower right")
            ax.grid(True)
        else:
            # 多分类ROC曲线
            from sklearn.preprocessing import label_binarize
            from sklearn.multiclass import OneVsRestClassifier
            from itertools import cycle
            
            # Binarize the output
            y_bin = label_binarize(labels, classes=range(len(self.class_names)))
            n_classes = y_bin.shape[1]
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curves
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=2,
                       label=f'ROC curve of class {self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Multi-class ROC Curves')
            ax.legend(loc="lower right")
            ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def visualize_samples(self, num_samples: int = 16):
        """可视化示例图片"""
        # 获取一批数据用于可视化
        data_iter = iter(self.dataloader)
        images, labels = next(data_iter)
        
        # 将标准化的图像还原（这里假设使用了标准的ImageNet归一化）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(std * img + mean, 0, 1)  # 反标准化
            
            axes[i].imshow(img)
            axes[i].set_title(f'True: {self.class_names[labels[i]]}')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.title(f"sample images\n{self.name}\n{self.data_name}")
        return fig
    
    def log_to_tensorboard(self, metrics: Dict[str, float], 
                          labels: np.ndarray, preds: np.ndarray, 
                          probs: np.ndarray):
        """将结果记录到tensorboard"""
        # 记录标量指标
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'Validation/{metric_name}', value, 0)
        
        # 记录混淆矩阵
        cm_fig = self.plot_confusion_matrix(labels, preds)
        self.writer.add_figure('Validation/Confusion_Matrix', cm_fig, 0)
        
        # 记录ROC曲线
        roc_fig = self.plot_roc_curve(labels, probs)
        self.writer.add_figure('Validation/ROC_Curve', roc_fig, 0)
        
        # 记录示例图片
        sample_fig = self.visualize_samples()
        self.writer.add_figure('Validation/Sample_Images', sample_fig, 0)
        
        # 创建指标表格
        metric_table = f"#### model:{self.name}\n#### dataset:{self.data_name}\n"
        metric_table += "| Metric | Value |\n|--------|-------|\n"       # TODO 添加模型名称
        for name, value in metrics.items():
            metric_table += f"| {name} | {value:.4f} |\n"
        
        # 记录文本格式的指标
        metric_text = "Metrics logged to TensorBoard:\n"
        for name, value in metrics.items():
            metric_text += f"{name}: {value:.4f}\n"
            print(f"{name}: {value:.4f}")
        
        self.writer.add_text('Validation/Metrics_Table', metric_table, 0)
        self.writer.add_text('Validation/Metrics_Text', metric_text, 0)
    
    def run_validation(self):
        """运行完整的验证流程"""
        logger.info("Running validation...")
        metrics, labels, preds, probs = self.validate()
        
        logger.info("Logging to TensorBoard...")
        self.log_to_tensorboard(metrics, labels, preds, probs)
        
        logger.info(f"Validation completed. Results saved to {self.log_dir}")
        self.writer.close()

        logger.success("validation success")