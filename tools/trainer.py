import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler

# 导入自定义的工具函数
from tools.utils import calculate_metrics, plot_roc_curve ,check_data_distribution


# -----------------
# 核心组件设置
# -----------------
def setup_training_components(
    model: nn.Module,
    train_config: Dict[str, Any]
) -> Tuple[nn.Module, Optimizer, _LRScheduler]:
    # 损失函数：BCEWithLogitsLoss 内部集成了 Sigmoid，数值更稳定
    criterion = nn.BCEWithLogitsLoss()

    # 优化器：CLIP 模型通常需要较小的学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 1e-4)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.get('epochs', 50),
        eta_min=train_config.get('min_lr', 1e-6) # 设置一个最小学习率下限
    )
    
    return criterion, optimizer, scheduler

# -----------------
# 训练函数
# -----------------
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    gradient_accumulation_steps: int = 1,
    use_mixed_precision: bool = False
) -> Tuple[float, float, int]:
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    epoch_steps = len(train_loader)
    
    all_targets = []
    all_scores = []
    
    scaler = GradScaler() if use_mixed_precision else None
    
    batch_pbar = tqdm(enumerate(train_loader), total=epoch_steps, 
                      desc=f"Ep {epoch+1} Train", unit="bt", leave=False)
    
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in batch_pbar:
        if data is None or target is None: continue
        
        data, target = data.to(device), target.to(device).float()
        
        # --- 前向传播 ---
        # 兼容混合精度
        context = autocast() if (use_mixed_precision and scaler) else torch.no_grad() # no_grad是占位，实际不用
        if not (use_mixed_precision and scaler): context = torch.enable_grad() # 修正上面占位的逻辑
        
        with (autocast() if use_mixed_precision else torch.enable_grad()):
            # 稳健的模型输出获取
            raw_output = model(data)
            if isinstance(raw_output, tuple):
                outputs = raw_output[0] # 假设第一个是 logits
            else:
                outputs = raw_output
            
            # 确保 outputs 形状和 target 一致 [B] vs [B]
            if outputs.shape != target.shape:
                outputs = outputs.view_as(target)

            loss = criterion(outputs, target) / gradient_accumulation_steps

        # --- 反向传播 ---
        if use_mixed_precision and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # --- 记录与更新 ---
        batch_loss = loss.item() * gradient_accumulation_steps
        total_loss += batch_loss
        
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            batch_correct = (preds == target).sum().item()
            batch_total = target.size(0)
            
            correct += batch_correct
            total += batch_total
            
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

        # --- 梯度更新 ---
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == epoch_steps:
            if use_mixed_precision and scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # TensorBoard Log (每 10 step)
        if global_step % 10 == 0:
            batch_acc = 100 * batch_correct / batch_total
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/Train_Batch', batch_loss, global_step)
            writer.add_scalar('Accuracy/Train_Batch', batch_acc, global_step)
            writer.add_scalar('LR', current_lr, global_step)
            
            tqdm.write(f">>> [{global_step}] Batch Loss:{batch_loss:.4f}\t|  Batch Acc:{batch_acc:.2f}\t|  lr:{current_lr:.6f}")
        global_step += 1
    
    # Epoch 结束统计
    avg_loss = total_loss / epoch_steps
    accuracy = 100.0 * correct / total
    
    # 计算 AUC (这里使用 sklearn 保持原样，也可以改为 calculate_metrics)
    try:
        train_auc = roc_auc_score(all_targets, all_scores)
    except:
        train_auc = 0.5
        
    writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train_Epoch', accuracy, epoch)
    writer.add_scalar('AUC/Train_Epoch', train_auc, epoch)
    
    return avg_loss, accuracy, global_step

# -----------------
# 验证函数 (增强版)
# -----------------
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    use_mixed_precision: bool = False
) -> Tuple[float, float]:
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_scores = []
    all_preds = []
    
    val_pbar = tqdm(val_loader, desc=f"Ep {epoch+1} Val", unit="bt", leave=False)
    
    with torch.no_grad():
        for data, target in val_pbar:
            if data is None or target is None: continue
            
            data, target = data.to(device), target.to(device).float()
            
            with (autocast() if use_mixed_precision else torch.no_grad()):
                raw_output = model(data)
                if isinstance(raw_output, tuple):
                    outputs = raw_output[0]
                else:
                    outputs = raw_output
                
                if outputs.shape != target.shape:
                    outputs = outputs.view_as(target)
                    
                loss = criterion(outputs, target)
            
            # 累加 Loss
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            total += target.size(0)
            correct += (preds == target).sum().item()
            
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            val_pbar.set_postfix({"Loss": f"{loss.item():.3f}"})
            
    # 计算平均指标
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    
    # === 使用 tools.utils.calculate_metrics 计算高级指标 (已修改) ===
    try:
        # 将 list 转换为 numpy array
        np_targets = np.array(all_targets)
        np_scores = np.array(all_scores)

        # 调用 calculate_metrics 来获取所有指标
        metrics = calculate_metrics(np_targets, np_scores)
        
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        auc = metrics.get('auc', 0.5)

        # 调用 plot_roc_curve 绘制 ROC 曲线并记录到 TensorBoard
        roc_fig = plot_roc_curve(np_targets, np_scores, epoch)
        writer.add_figure('ROC_Curve/Validation', roc_fig, epoch)
        
    except Exception as e:
        logger.warning(f"Metrics calculation using tools.utils failed: {e}")
        # 失败时设置默认值
        precision, recall, f1, auc = 0, 0, 0, 0.5
        # 即使失败，仍然尝试记录一个空的图
        try:
             roc_fig = plot_roc_curve(np.array([0, 1]), np.array([0.5, 0.5]), epoch)
             writer.add_figure('ROC_Curve/Validation', roc_fig, epoch)
        except:
             pass

    # 输出详细日志 (这能帮你看出是不是只预测某一个类别)
    num_pos_pred = sum(all_preds)
    tqdm.write(f">>> Val Stats: Pred_Pos: {int(num_pos_pred)}/{total} | Precision: {precision:.3f} | Recall: {recall:.3f}")

    # TensorBoard
    writer.add_scalar('Loss/Validation', avg_loss, epoch)
    writer.add_scalar('Accuracy/Validation', accuracy, epoch)
    writer.add_scalar('AUC/Validation', auc, epoch)
    writer.add_scalar('F1/Validation', f1, epoch)
    
    tqdm.write(f'>>>   Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.2f}% | AUC: {auc:.4f}')
    
    return avg_loss, accuracy

# -----------------
# 主循环
# -----------------
# @logger.catch()
def train_model(
    model: nn.Module,
    train_dataset: DataLoader,
    test_dataset: DataLoader,
    train_config: Dict[str, Any],
    device: torch.device,
    writer: SummaryWriter,
    checkpoint_path: Path
) -> dict:
    
    # 1. 检查数据分布 (这是最关键的一步 debugging)
    check_data_distribution(train_dataset, "Train Set")
    check_data_distribution(test_dataset, "Val Set")
    
    criterion, optimizer, scheduler = setup_training_components(model, train_config)
    
    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
    use_mixed_precision = train_config.get('use_mixed_precision', False)
    
    best_val_acc = 0.0
    num_epochs = train_config.get('epochs', 5)
    global_step = 0
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="ep")
    
    for epoch in epoch_pbar:
        # Train
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_dataset, criterion, optimizer, device, epoch, writer, global_step,
            gradient_accumulation_steps, use_mixed_precision
        )
        
        # Val
        val_loss, val_acc = validate(
            model, test_dataset, criterion, device, epoch, writer, use_mixed_precision
        )
        
        scheduler.step()
        
        epoch_pbar.set_postfix({"T_Acc": f"{train_acc:.1f}%", "V_Acc": f"{val_acc:.1f}%"})
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc
            }, checkpoint_path / 'best_model.pth')
            tqdm.write(f">>> New Best Model Saved! ({val_acc:.2f}%)")
            
        # 定期保存
        if (epoch + 1) % train_config.get('save_interval', 10) == 0:
            torch.save(model.state_dict(), checkpoint_path / f'epoch_{epoch+1}.pth')

    writer.close()
    return {'best_checkpoint':Path(checkpoint_path / 'best_model.pth')}