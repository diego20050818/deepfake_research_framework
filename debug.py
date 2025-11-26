#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Lightweight] 使用合成数据和未预训练模型快速测试验证代码
无需下载任何数据集或权重。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
import tempfile
import os
import numpy as np

# 导入您的验证代码
# 确保 tools.validation 在 python path 下，或者根据实际情况调整路径
from tools.validation import ModelValidator

# --- 1. 更加轻量化的模型设置 ---
class BinaryResNet(nn.Module):
    def __init__(self):
        super(BinaryResNet, self).__init__()
        # 关键修改：weights=None (旧版 pytorch 使用 pretrained=False)
        # 避免下载预训练权重，反正我们要用合成数据训练它
        self.resnet = resnet18(weights=None) 
        
        # 替换全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.resnet(x)

# --- 2. 极速合成数据生成器 (替代 CIFAR10) ---
def create_binary_dataset():
    """
    在内存中直接生成合成数据。
    Class 0: 标准正态分布噪声 (较暗)
    Class 1: 偏移后的噪声 (较亮) -> 保证模型能迅速学会区分
    """
    print("Generating synthetic data in memory (No download needed)...")
    
    # 少量样本即可验证流程 (Train: 100, Test: 50)
    n_train = 128
    n_test = 64
    
    # 模拟 ResNet 输入: (Batch, Channel, Height, Width)
    # 使用 224x224 保证和真实场景一致
    train_x = torch.randn(n_train, 3, 224, 224)
    train_y = torch.randint(0, 2, (n_train,)).float()
    
    test_x = torch.randn(n_test, 3, 224, 224)
    test_y = torch.randint(0, 2, (n_test,)).float()
    
    # 让 Class 1 的数据特征明显不同 (加上偏移量)，确保模型能训练收敛
    train_x[train_y == 1] += 2.0
    test_x[test_y == 1] += 2.0
    
    # 包装成 TensorDataset
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    # 兼容接口，添加 target 属性供部分验证脚本读取
    test_dataset.targets = test_y.tolist() 
    
    # 创建 DataLoader
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return trainloader, testloader

def train_model(trainloader, epochs=2):
    """快速训练"""
    model = BinaryResNet()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化所有层，因为没有预训练
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 每个 Epoch 打印一次即可
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}: Loss {running_loss:.3f}, Accuracy {accuracy:.2f}%')
            
    print(f'Finished Training. Final accuracy: {100 * correct / total:.2f}%')
    return model

def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': 'BinaryResNet_Synthetic'
    }, path)
    print(f"Model saved to {path}")

def main():
    # 1. 获取数据
    trainloader, testloader = create_binary_dataset()
    
    # 2. 训练模型 (因为数据简单，2个epoch就能达到很高准确率)
    print("\n--- Starting Quick Training ---")
    model = train_model(trainloader, epochs=2)
    
    # 3. 保存临时模型
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        checkpoint_path = tmp_file.name
    save_model(model, checkpoint_path)
    
    # 4. 准备验证信息
    checkpoint_info = {
        'name': 'BinaryResNet_Synthetic',
        'dataset': 'Synthetic_Noise'
    }
    
    # try:
    print("\n--- Starting Validation Check ---")
    # 创建验证器
    validator = ModelValidator(
        dataloader=testloader,
        class_names=['Class_Low', 'Class_High'], # 任意名称
        chechpoint_info=checkpoint_info,
        model=model, 
        model_path=checkpoint_path,
        log_dir='./runs/validation_test_synthetic' # 独立的测试目录
    )
    
    # 运行验证
    validator.run_validation()
    print("\n✅ Validation script executed successfully!")
        
    # except Exception as e:
    #     print(f"\n❌ Validation script failed with error:\n{e}")
    #     import traceback
    #     traceback.print_exc()
    # finally:
    #     # 清理
    #     if os.path.exists(checkpoint_path):
    #         os.unlink(checkpoint_path)

if __name__ == "__main__":
    main()