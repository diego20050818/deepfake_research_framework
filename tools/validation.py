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
                    # 如果没有模型实例，直接使用状态字典初始化模型
                    raise ValueError("未提供模型实例，请在初始化ModelValidator时提供model参数")
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
        修复了 batch 不整除导致的 numpy 报错，并增加了详细的概率对比日志
        """
        # 使用列表收集每个 batch 的结果，最后再拼接
        # 避免直接 append 到一个大 list 然后转 numpy 导致的维度错误
        batch_preds_list = []
        batch_labels_list = []
        batch_probs_list = []
        
        # 添加进度条
        progress_bar = tqdm(self.dataloader, desc="Validating", leave=False)
        
        self.model.eval() # 确保是 eval 模式
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # --- 1. 获取模型输出 ---
                try:
                    # 你的模型返回 (logits, repr)
                    outputs_tuple = self.model(inputs)
                    if isinstance(outputs_tuple, tuple):
                        logits = outputs_tuple[0] # 取第一个元素：logits
                    else:
                        logits = outputs_tuple
                except Exception as e:
                    print(f"Model output error: {e}")
                    raise e
                
                # --- 2. 转换为正样本概率 ---
                # Logits -> Sigmoid -> Probability (0.0 ~ 1.0)
              
                probs = torch.sigmoid(logits) 
                
                # --- 3. 生成硬预测 (0 或 1) ---
                preds = (probs > 0.5).float()

                # --- 4. 收集数据 (保持在 CPU 上) ---
                # 注意：这里直接存 numpy 数组，而不是 extend 列表
                batch_probs_list.append(probs.cpu().numpy())
                batch_labels_list.append(labels.cpu().numpy())
                batch_preds_list.append(preds.cpu().numpy())
                
                # 更新进度条
                progress_bar.set_postfix({'Batch': inputs.size(0)})
        
        # --- 5. 安全拼接 (Fix: 解决 inhomogeneity 报错) ---
        # 使用 concatenate 处理最后一个 batch 大小不一致的问题
        all_probs = np.concatenate(batch_probs_list, axis=0)
        all_labels = np.concatenate(batch_labels_list, axis=0)
        all_preds = np.concatenate(batch_preds_list, axis=0)

        # all_probs = all_preds.squeeze(1) 
        
        # --- 6. 打印直观对比 ---
        print("\n" + "="*40)
        print("🔍 Probability vs Label Check (Top 10 samples)")
        print(f"{'Probability (Positive)':<25} | {'Label':<10} | {'Correct?'}")
        print("-" * 50)
        for i in range(min(10, len(all_labels))):
            p = all_probs[i]
            l = all_labels[i]
            # 判断预测是否正确
            is_correct = "✅" if (p > 0.5) == (l == 1) else "❌"
            print(f"{p:.4f} ({(p*100):.1f}%) {'':<12} | {int(l):<10} | {is_correct}")
        print("="*40 + "\n")

        # --- 7. 计算指标 (保持原有逻辑) ---
        metrics = {}
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['precision'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 二分类特有指标
        if len(self.class_names) == 2:
            try:
                metrics['auc'] = roc_auc_score(all_labels, all_probs)
                
                # 计算最佳阈值
                fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                metrics['optimal_threshold'] = optimal_threshold
                
                optimal_preds = (all_probs >= optimal_threshold).astype(int)
                metrics['optimal_accuracy'] = accuracy_score(all_labels, optimal_preds)
            except Exception as e:
                print(f"Warning: Could not calculate ROC/AUC: {e}")
                metrics['auc'] = 0.0
        
        metrics['error_rate'] = 1 - metrics['accuracy']
        
        return metrics, all_labels, all_preds, all_probs
    
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
                     pad=15)   
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, labels: np.ndarray, probs: np.ndarray) -> plt.Figure: # type:ignore
        """绘制ROC曲线"""
        fig, ax = plt.subplots(figsize=(8, 6))
        

        if probs.ndim == 1:
            # 如果probs是一维的，直接使用
            fpr, tpr, _ = roc_curve(labels, probs)
            auc_score = roc_auc_score(labels, probs)
        else:
            # 如果probs是二维的，使用第二列（正类）
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])# BUG
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
                        )  
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
        fig.suptitle(f"Sample images\n{self.name}\n{self.data_name}", fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.85)
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
        metric_table += "| Metric | Value |\n|--------|-------|\n"       
        for name, value in metrics.items():
            metric_table += f"| {name} | {value:.4f} |\n"
        
        # 记录文本格式的指标
        metric_text = "Metrics logged to TensorBoard:\n"
        for name, value in metrics.items():
            metric_text += f"{name}: {value:.4f}\n"
            print(f"{name}: {value:.4f}")
        
        self.writer.add_text('Validation/Metrics_Table', metric_table, 0)
        self.writer.add_text('Validation/Metrics_Text', metric_text, 0)
    
    # @logger.catch()
    def run_validation(self):
        """运行完整的验证流程"""
        logger.info("Running validation...")
        metrics, labels, preds, probs = self.validate()
        
        logger.info("Logging to TensorBoard...")
        self.log_to_tensorboard(metrics, labels, preds, probs)
        
        logger.info(f"Validation completed. Results saved to {self.log_dir}")
        self.writer.close()

        logger.success("validation success")