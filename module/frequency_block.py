
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from module import SimpleDCTLayer,DCTModule

class FrequencyDomainBranch(nn.Module):
    """
    频域分支 - 在频域中分析图像特征
    修复说明：
    - 之前使用固定维度的 self.freq_proj 导致输入扁平化尺寸与线性层不匹配。
    - 修复后采用 nn.ModuleList 存储投影层，并在第一次 forward 时，
      为 features_list 中的每一个不同尺度的特征图，分别延迟创建和存储一个线性投影层。
    """
    def __init__(self, dim):
        super().__init__()
        # 频域变换层
        self.dct = SimpleDCTLayer()
        # 存储每个尺度特征图对应的线性投影层
        self.freq_projs = nn.ModuleList() 
        # 存储已初始化的输入维度列表，用于判断是否需要新建投影层
        self.in_features_list = []
        self.target_dim = dim

    def _get_proj(self, in_features, device, dtype):
        """获取或创建匹配当前输入维度的线性投影层"""
        
        # 1. 检查是否已经为当前 in_features 尺寸创建了投影层
        try:
            # 查找 in_features 在已初始化维度列表中的索引
            idx = self.in_features_list.index(in_features)
            # 如果找到，则返回对应的投影层
            return self.freq_projs[idx]
        except ValueError:
            # 2. 如果是新的 in_features 尺寸，则创建新的投影层并添加到列表中
            print(f"Initializing new projection layer for in_features={in_features}")
            new_proj = nn.Linear(in_features, self.target_dim).to(device=device, dtype=dtype)
            self.freq_projs.append(new_proj)
            self.in_features_list.append(in_features)
            return new_proj

    def forward(self, features_list):
        """
        前向传播
        Args:
            features_list: 多尺度特征图列表，每个元素形状为 [B, C, H, W]
        Returns:
            fused: 融合后的频域特征，形状为 [B, dim]
        """
        outs = []
        for feat in features_list:
            # 1. 转换到频域
            # 注意：DCT操作必须是相同的SimpleDCTLayer，以保证参数共享和DCT的一致性
            d = self.dct(feat)  # 可能形状为 [B, C, H, W_reduced]
            
            # 2. 展平频域特征
            d_flat = d.view(d.size(0), -1)  # [B, in_features]
            in_features = d_flat.shape[1] # 获取当前的扁平化维度
            
            # 3. 获取或创建匹配当前扁平化维度的线性层
            freq_proj = self._get_proj(in_features, d_flat.device, d_flat.dtype)
            
            # 4. 线性投影调整特征维度
            outs.append(freq_proj(d_flat)) # [B, target_dim]

        # 堆叠所有层特征并平均融合
        fused = torch.stack(outs, dim=1).mean(dim=1)
        return fused

# FrequencyBranch 保持原实现（不变）
class FrequencyBranch(nn.Module):
    """频域特征提取分支：DCT变换+Patch嵌入+特征增强"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, drop_rate=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 与空域ViT的Patch数量一致
        
        # 1. DCT变换（捕捉低频轮廓+高频噪声，适合鉴伪）
        self.dct = DCTModule()
        
        # 2. 频域特征投影：将3通道频域图→Patch嵌入（与空域embed_dim一致）
        # 输入：频域图 [B, 3, H, W] → 输出：Patch嵌入 [B, num_patches, embed_dim]
        self.freq_proj = nn.Conv2d(
            in_channels=3,  # 频域图仍保持3通道（对应RGB原始通道的频域）
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        
        # 3. 频域特征增强（提升频域特征表达）
        self.freq_norm = nn.LayerNorm(embed_dim)
        self.freq_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 4. 频域分支的Class Token（与空域对齐，用于融合后分类）
        self.freq_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, x):
        # x: 原始图像 [B, 3, H, W]
        B = x.shape[0]
        
        # 步骤1：频域变换（DCT）→ [B, 3, H, W]（频域特征图）
        freq_map = self.dct(x)
        
        # 步骤2：频域图→Patch嵌入（Conv2d实现分块+投影）
        freq_patch_embed = self.freq_proj(freq_map)  # [B, embed_dim, h, w]
        freq_patch_embed = rearrange(freq_patch_embed, 'b d h w -> b (h w) d')  # [B, num_patches, embed_dim]
        
        # 步骤3：频域特征增强
        freq_patch_embed = self.freq_norm(freq_patch_embed)  # [B, N, D]
        freq_patch_embed = freq_patch_embed + self.freq_mlp(freq_patch_embed)  # 残差连接
        
        # 步骤4：添加频域Class Token
        freq_cls_token = self.freq_cls_token.expand(B, -1, -1)  # [B, 1, D]
        freq_features = torch.cat([freq_cls_token, freq_patch_embed], dim=1)  # [B, N+1, D]（与空域特征格式一致）
        
        return freq_features
