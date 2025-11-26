import torch
import torch.nn as nn

class GlobalSemanticBranch(nn.Module):
    """
    全局语义分支 - 提取图像的整体语义信息
    通过全局平均池化获取每个特征图的全局表示
    """
    def __init__(self, dim):
        super().__init__()
        # 全局平均池化：将任意尺寸的特征图池化为1x1
        self.attention_pool = nn.AdaptiveAvgPool2d(1)
        # 线性投影层：调整特征维度
        self.proj = nn.Linear(dim, dim)

    def forward(self, features_list):
        """
        前向传播
        Args:
            features_list: 多尺度特征图列表，每个元素形状为 [B, C, H, W]
        Returns:
            fused: 融合后的全局语义特征，形状为 [B, dim]
        """
        pooled = []
        for feat in features_list:
            # feat: [B, C, H, W] -> 全局池化 -> [B, C, 1, 1] -> 展平 -> [B, C]
            p = self.attention_pool(feat).view(feat.size(0), -1)
            # 线性投影调整特征
            pooled.append(self.proj(p))
        # 堆叠所有层特征：[B, num_layers, dim] -> 沿层维度平均 -> [B, dim]
        fused = torch.stack(pooled, dim=1).mean(dim=1)
        return fused