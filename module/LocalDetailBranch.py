import torch
import torch.nn as nn
import torch.nn.functional as F
from module import SpatialAttention


class LocalDetailBranch(nn.Module):
    """
    局部细节分支 - 空间注意力机制
    增加输入通道自适配：当传入特征通道数与期望不同时，使用 1x1 卷积投影到期望维度。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.spatial_attention = SpatialAttention(dim)
        # 使用 ModuleDict 缓存不同输入通道对应的对齐层
        self.align_layers = nn.ModuleDict()

    def _get_align(self, in_ch):
        key = str(in_ch)
        if key not in self.align_layers:
            # 创建并注册 1x1 卷积以对齐通道
            self.align_layers[key] = nn.Conv2d(in_ch, self.dim, kernel_size=1, bias=False)
        return self.align_layers[key]

    def forward(self, features_list):
        attended_features = []
        for feat in features_list:
            # feat 期望形状 [B, C, H, W]
            if not isinstance(feat, torch.Tensor):
                # 如果不是张量则跳过
                continue
            B, C, H, W = feat.shape
            if C != self.dim:
                # 使用或创建 1x1 conv 将通道对齐到 self.dim
                align = self._get_align(C).to(feat.device)
                feat = align(feat)
            attended = self.spatial_attention(feat)
            pooled = F.adaptive_avg_pool2d(attended, 1).view(attended.size(0), -1)
            attended_features.append(pooled)

        if len(attended_features) == 0:
            # 返回零向量以保证上层不报错
            return torch.zeros((0, self.dim), device=next(self.parameters()).device)

        fused = torch.stack(attended_features, dim=1).mean(dim=1)
        return fused
