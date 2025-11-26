import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
from timm.models.vision_transformer import VisionTransformer
# import os
# import sys
# sys.path.append(os.getcwd())
from module.CrossAttention import CrossAttentionCombination
from module.DCTConvBlock import DCTConvBlock
from module.GateMLP import GatedMLP
from module.SSCA import SSCA
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://mirrors.aliyun.com/hugging-face-models'

from module import ViTBlockWithAdapter,FrequencyBranch,Multipole_Attention2D,LocalAttention2D



class FaceAntiSpoofingViT(nn.Module):
    """基于局部窗口注意力的人脸鉴伪模型 - 双分支版本（空域+频域）"""
    def __init__(self, num_classes=1, img_size=224, patch_size=16, 
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 use_adapter=True):
        super().__init__()
        
        from timm.models.vision_transformer import VisionTransformer
        
        # 基础ViT
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,  # 不使用分类头
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # 替换为带适配器的块
        if use_adapter:
            self.vit.blocks = nn.Sequential(*[
                ViTBlockWithAdapter(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate * i / (depth - 1) if depth > 1 else 0,
                    use_adapter=True
                ) for i in range(depth)
            ])

        # 频域分支
        self.frequency_branch = FrequencyBranch(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            drop_rate=drop_rate
        )
        
        # 特征融合模块 - 将空域和频域特征进行融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.LayerNorm(embed_dim // 2)
        )
        
        # 分类头 - 对融合后的特征进行分类
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: 输入图像 [B, 3, H, W]
        
        # 空域特征提取
        spatial_features = self.vit.forward_features(x)
        spatial_cls_token = spatial_features[:, 0]  # [B, embed_dim] - 获取分类token
        
        # 频域特征提取
        frequency_features = self.frequency_branch(x)  # [B, seq_len, embed_dim]
        frequency_cls_token = frequency_features[:, 0]  # [B, embed_dim] - 频域分类token
        
        # 特征融合 - 拼接空域和频域的分类token并投影
        fused_features = torch.cat([spatial_cls_token, frequency_cls_token], dim=1)  # [B, embed_dim*2]
        fused_features = self.feature_fusion(fused_features)  # [B, embed_dim//2]
        
        # 分类
        output = self.classifier(fused_features)  # [B, num_classes]
        output = output.squeeze(1)  # 如果是二分类，将其压缩为一维
        return output

