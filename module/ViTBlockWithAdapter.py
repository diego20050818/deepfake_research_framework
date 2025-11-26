import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from timm.layers.drop import DropPath

from module.Adapter import Adapter



class ViTBlockWithAdapter(nn.Module):
    """带适配器的ViT块"""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, 
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 use_adapter=True):
        super().__init__()
        self.use_adapter = use_adapter
        
        # 注意力层
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
        # MLP层
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop_rate)
        )
        
        # 适配器
        if use_adapter:
            self.adapter = Adapter(d_model=dim, bottleneck=64, dropout=0.1)

    def forward(self, x):
        # 注意力部分
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        
        # MLP部分 + 适配器
        residual = x
        x = self.norm2(x)
        x = self.drop_path(self.mlp(x))
        
        if self.use_adapter:
            adapt_x = self.adapter(residual, add_residual=False)
            x = x + adapt_x
        
        x = residual + x
        return x
