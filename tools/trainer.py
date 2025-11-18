import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tools.utils import calculate_metrics, plot_roc_curve

# -----------------
# 损失函数和优化器
# -----------------
def setup_training_components(
    model: nn.Module,
    train_config: Dict[str, Any]
) -> Tuple[nn.Module, Optimizer, _LRScheduler]:
    """设置训练所需的损失函数、优化器和学习率调度器。

    为二分类任务配置 BCEWithLogitsLoss 损失函数、AdamW 优化器和
    CosineAnnealingLR 学习率调度器，参数通过 train_config 配置。

    Args:
        model: 待训练的PyTorch模型
        train_config: 训练配置字典，支持的键包括：
            - learning_rate: 学习率（默认1e-4）
            - weight_decay: 权重衰减（默认1e-4）
            - epochs: 训练轮数（用于调度器T_max，默认50）

    Returns:
        Tuple[nn.Module, Optimizer, _LRScheduler]:
            criterion: BCEWithLogitsLoss损失函数实例
            optimizer: AdamW优化器实例
            scheduler: CosineAnnealingLR学习率调度器实例

    Examples:
        >>> model = nn.Linear(10, 1)
        >>> config = {'learning_rate': 5e-5, 'weight_decay': 1e-5, 'epochs': 100}
        >>> criterion, optimizer, scheduler = setup_training_components(model, config)
    """
    criterion = nn.BCEWithLogitsLoss()  # 适用于二分类任务 # NOTE 损失函数

    optimizer = torch.optim.AdamW(  # NOTE 优化器
        model.parameters(),
        lr=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 1e-4)
    )
    
    # 学习率调度器：按余弦曲线衰减学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(     # NOTE 学习调度器
        optimizer,
        T_max=train_config.get('epochs', 50)  
    )
    
    return criterion, optimizer, scheduler #type:ignore

# -----------------
# 训练函数（支持batch级进度条和TensorBoard更新）
# -----------------

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int
) -> Tuple[float, float, int]:
    """训练模型一个epoch。

    执行单轮训练流程：前向传播计算损失、反向传播更新参数、统计训练损失和准确率。
    支持batch级进度条和TensorBoard实时更新。

    Args:
        model: 待训练的PyTorch模型
        train_loader: 训练数据加载器，每个batch包含(data, target)
        criterion: 损失函数实例
        optimizer: 优化器实例
        device: 训练设备（CPU/GPU）
        epoch: 当前训练轮次
        writer: TensorBoard日志写入器（用于batch级更新）
        global_step: 全局步数（累计所有epoch的batch数，保证TensorBoard索引连续）

    Returns:
            tuple[float, float, int]:
                avg_loss: 该epoch的平均训练损失
                accuracy: 该epoch的训练准确率（百分比）
                global_step: 更新后的全局步数
    """
    model.train()  # 设置模型为训练模式
    total_loss = 0.0
    correct = 0
    total = 0
    epoch_steps = len(train_loader)
    
    # 用于AUC计算的存储
    all_targets = []
    all_scores = []
    
    # 内层进度条：每个epoch的batch迭代
    batch_pbar = tqdm(
        enumerate(train_loader),
        total=epoch_steps,
        desc=f"Epoch {epoch+1} (Batch)",
        unit="batch",
        leave=False
    )
    
    for batch_idx, (data, target) in batch_pbar:
        # 过滤空batch（来自之前的collate_fn处理）
        if data is None or target is None:
            continue
        
        # 数据移至目标设备，target转换为float类型（适配BCEWithLogitsLoss）
        data, target = data.to(device), target.to(device).float()
        
        # 前向传播
        optimizer.zero_grad()  # 清空梯度
        outputs, _ = model(data)  # 返回(logits, repr_for_contrast)
        
        # 计算损失
        loss = criterion(outputs, target)
        
        # 反向传播与参数更新
        loss.backward()
        optimizer.step()
        
        # 统计训练指标
        batch_loss = loss.item()
        total_loss += batch_loss
        
        # 计算batch准确率
        predicted_probs = torch.sigmoid(outputs)
        predicted = (predicted_probs > 0.5).float()
        batch_correct = (predicted == target).sum().item()
        batch_total = target.size(0)
        batch_acc = 100.0 * batch_correct / batch_total
        
        correct += batch_correct
        total += batch_total
        
        # 收集预测结果用于AUC计算
        all_targets.extend(target.cpu().numpy())
        all_scores.extend(predicted_probs.detach().cpu().numpy())
        
        # ----------------------
        # 每10个batch更新TensorBoard
        # ----------------------
        if global_step % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/Train_Batch', batch_loss, global_step)
            writer.add_scalar('Accuracy/Train_Batch', batch_acc, global_step)
            writer.add_scalar('Learning Rate', current_lr, global_step)

            tqdm.write(f">>> [{global_step}] Batch Loss:{batch_loss:.4f}\t|  Batch Acc:{batch_acc:.2f}\t|  lr:{current_lr:.6f}")
        
        # 更新进度条显示
        batch_pbar.set_postfix({
            "Global Step": global_step
        })
        
        # 更新全局步数
        global_step += 1
    
    batch_pbar.close()
    
    # 计算epoch级平均指标
    avg_loss = total_loss / epoch_steps if epoch_steps > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    # 记录epoch级指标（与batch级互补）
    writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train_Epoch', accuracy, epoch)
    
    # 计算并记录训练AUC
    if all_targets and all_scores:
        try:
            train_auc = roc_auc_score(np.array(all_targets), np.array(all_scores))
            writer.add_scalar('AUC/Train_Epoch', train_auc, epoch)
            tqdm.write(f'>>>   Train AUC: {train_auc:.4f}')
        except Exception as e:
            logger.warning(f"计算训练AUC时出错: {e}")
    
    return avg_loss, accuracy, global_step

# -----------------
# 验证函数
# -----------------

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> Tuple[float, float]:
    """验证模型性能。

    在验证集上评估模型的损失和准确率，禁用梯度计算以提高效率。
    支持TensorBoard记录验证指标。

    Args:
        model: 待验证的PyTorch模型
        val_loader: 验证数据加载器，每个batch包含(data, target)
        criterion: 损失函数实例
        device: 验证设备（CPU/GPU）
        epoch: 当前训练轮次（用于TensorBoard索引）
        writer: TensorBoard日志写入器

    Returns:
        Tuple[float, float]:
            avg_loss: 验证集平均损失
            accuracy: 验证集准确率（百分比）
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 用于AUC计算的存储
    all_targets = []
    all_scores = []
    
    # 验证进度条
    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1} (Validation)",
        unit="batch",
        leave=False
    )
    
    with torch.no_grad():  # 禁用梯度计算
        for data, target in val_pbar:
            if data is None or target is None:
                continue
            
            # 数据移至目标设备
            data, target = data.to(device), target.to(device).float()
            outputs, _ = model(data)
            loss = criterion(outputs, target)
            
            # 统计验证指标
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 收集预测结果用于AUC计算
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(torch.sigmoid(outputs).cpu().numpy())
            
            # 更新验证进度条
            val_pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
    
    val_pbar.close()
    
    # 计算验证集平均指标
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    # 计算AUC和其他指标
    if all_targets and all_scores:
        metrics = calculate_metrics(np.array(all_targets), np.array(all_scores))
        auc = metrics['auc']
        optimal_accuracy = metrics['accuracy']
        
        # 记录验证指标到TensorBoard（epoch级）
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        writer.add_scalar('AUC/Validation', auc, epoch)
        writer.add_scalar('Optimal_Accuracy/Validation', optimal_accuracy, epoch)
        
        # 绘制并记录ROC曲线
        try:
            fig = plot_roc_curve(np.array(all_targets), np.array(all_scores), epoch)
            writer.add_figure('ROC_Curve/Validation', fig, epoch)
            plt.close(fig)  # 关闭图形以释放内存
        except Exception as e:
            logger.warning(f"绘制ROC曲线时出错: {e}")
        
        # 输出详细验证结果
        tqdm.write(f'>>>   Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.2f}% | AUC: {auc:.4f}')
        tqdm.write(f'>>>   Optimal Acc (based on ROC): {optimal_accuracy:.2f}% | Threshold: {metrics["optimal_threshold"]:.4f}')
        
        return avg_loss, accuracy
    else:
        # 记录验证指标到TensorBoard（epoch级）
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        
        # 输出验证结果
        tqdm.write(f'>>>   Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.2f}%')
        
        return avg_loss, accuracy

# -----------------
# 主训练循环（双层进度条+batch级TensorBoard）
# -----------------
@logger.catch()
def train_model(
    model: nn.Module,
    train_dataset: DataLoader,
    test_dataset: DataLoader,
    train_config: Dict[str, Any],
    device: torch.device,
    writer: SummaryWriter,
    checkpoint_path:Path
) -> float:
    """模型训练主循环。

    整合训练和验证流程，实现：
    1. 双层进度条（外层epoch，内层batch）
    2. 每个batch更新TensorBoard指标
    3. 学习率调度、最佳模型保存和定期检查点保存

    Args:
        model: 待训练的PyTorch模型
        train_dataset: 训练数据加载器
        test_dataset: 验证数据加载器（命名为test_dataset，实际用于验证）
        train_config: 训练配置字典，支持的键包括：
            - epochs: 总训练轮数（默认50）
            - save_interval: 检查点保存间隔（默认10轮）
            - 其他支持setup_training_components的配置键
        device: 训练设备（CPU/GPU）
        writer: TensorBoard日志写入器实例

    Returns:
        float: 训练过程中的最佳验证准确率（百分比）
    """
    # 初始化训练组件（损失函数、优化器、调度器）
    criterion, optimizer, scheduler = setup_training_components(model, train_config)
    
    best_val_acc = 0.0  # 记录最佳验证准确率
    num_epochs = train_config.get('epochs', 5)  # 总训练轮数
    global_step = 0  # 全局步数（累计所有batch，用于TensorBoard统一索引）
    
    # 外层进度条：epoch迭代
    epoch_pbar = tqdm(
        range(num_epochs),
        desc="Total Training",
        unit="epoch",
        unit_scale=True,
        leave=True  # 训练结束后保留进度条
    )
    
    
    for epoch in epoch_pbar:
        # 训练阶段：单轮训练（带batch进度条和TensorBoard batch级更新）
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_dataset, criterion, optimizer, device, epoch, writer, global_step
        )
        
        # 验证阶段：评估模型性能（带验证进度条）
        val_loss, val_acc = validate(model, test_dataset, criterion, device, epoch, writer)
        
        # 更新学习率
        scheduler.step()
        
        # 更新外层进度条显示（展示当前epoch的关键指标）
        epoch_pbar.set_postfix({
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc:.2f}%",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc:.2f}%",
            "Best Val Acc": f"{best_val_acc:.2f}%"
        })
        
        # 输出轮次训练日志
        tqdm.write(f'\n>>> Epoch [{epoch+1}/{num_epochs}]')
        tqdm.write(f'>>>   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        tqdm.write(f'>>>   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        tqdm.write(f'>>>   Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型（基于验证准确率）
        best_checkpoint_path = checkpoint_path / 'best_model.pth'

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'global_step': global_step,
            }, best_checkpoint_path)
            tqdm.write(f'>>>    保存最佳模型，验证准确率: {val_acc:.2f}%')
        
        # 定期保存训练检查点
        save_interval = train_config.get('save_interval', 10)
        
        if (epoch + 1) % save_interval == 0:
            single_checkpoint_path = checkpoint_path / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'global_step': global_step,
            }, single_checkpoint_path)
            tqdm.write(f'>>>    保存检查点 {str(single_checkpoint_path)}')
    
    epoch_pbar.close()
    
    # 训练完成
    tqdm.write(f"\n>>>  训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    writer.close()  # 关闭TensorBoard写入器
    return best_val_acc