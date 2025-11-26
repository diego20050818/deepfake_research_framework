import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

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

from module import GlobalSemanticBranch,LocalDetailBranch,FrequencyDomainBranch,CrossAttentionFusion
from module import SwinTransformerBlock,SwinTransformerEncoder

class MultiScaleHierarchicalTransformer(nn.Module):
    """
    多尺度层次化Transformer架构
    
    设计理念：
    - 多尺度输入：处理不同分辨率的图像
    - 层次化特征：从局部到全局的特征提取
    - 多分支融合：结合语义、细节和频域信息
    """
    
    def __init__(self, num_classes=2, img_size=224, embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, use_scales=[0.5, 0.25]):
        super().__init__()
        
        self.img_size = img_size
        self.use_scales = use_scales  # 多尺度比例 [0.5, 0.25]
        
        # ==================== 多尺度输入处理 ====================
        self.scale_encoders = nn.ModuleList()
        for scale in use_scales:
            encoder = SwinTransformerEncoder(
                img_size=int(img_size * scale),
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size
            )
            self.scale_encoders.append(encoder)
        
        # 原始尺度编码器
        self.original_encoder = SwinTransformerEncoder(
            img_size=img_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size
        )

        
        # ==================== 多分支特征提取 ====================
        self.global_branch = GlobalSemanticBranch(embed_dim * 8)  # 阶段4输出维度
        self.local_branch = LocalDetailBranch(embed_dim * 4)     # 阶段3输出维度
        self.frequency_branch = FrequencyDomainBranch(embed_dim * 2)  # 阶段2输出维度
        
        # ==================== 交叉注意力融合 ====================
        self.cross_attention_fusion = CrossAttentionFusion(
            global_dim=embed_dim * 8,
            local_dim=embed_dim * 4,
            freq_dim=embed_dim * 2,
            out_dim=embed_dim * 8
        )
        
        # ==================== 输出头 ====================
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 8),
            nn.Linear(embed_dim * 8, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        self.feature_head = nn.Linear(embed_dim * 8, 512)  # 用于对比学习的特征表示

    def _upsample_features(self, features, target_size):
        """
        上采样特征列表到目标空间尺寸 target_size=(H, W)。
        features: list of tensors or other objects; 对于 4D 张量执行双线性上采样，其它保持不变。
        返回与输入同结构的列表。
        """
        upsampled = []
        for f in features:
            if isinstance(f, torch.Tensor) and f.dim() == 4:
                up = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            else:
                up = f
            upsampled.append(up)
        return upsampled
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 [batch_size, 3, H, W]
        
        返回:
            logits: 分类logits [batch_size, num_classes]
            features: 特征表示 [batch_size, 512]
        """
        batch_size = x.shape[0]
        
        # ==================== 多尺度特征提取 ====================
        multi_scale_features = []
        
        # 原始尺度
        orig_features = self.original_encoder(x)
        multi_scale_features.append(orig_features)
        
        # 多尺度处理
        for i, scale in enumerate(self.use_scales):
            scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            scale_features = self.scale_encoders[i](scaled_x)
            scale_features = self._upsample_features(scale_features, orig_features[-1].shape[-2:])
            multi_scale_features.append(scale_features)
        
        # ==================== 多分支特征提取 ====================
        stage4_features = [feat[-1] for feat in multi_scale_features]  # 阶段4特征
        stage3_features = [feat[-2] for feat in multi_scale_features]  # 阶段3特征
        stage2_features = [feat[-3] for feat in multi_scale_features]  # 阶段2特征
        
        global_features = self.global_branch(stage4_features)
        local_features = self.local_branch(stage3_features)
        freq_features = self.frequency_branch(stage2_features)
        
        # ==================== 交叉注意力融合 ====================
        fused_features = self.cross_attention_fusion(
            global_features, local_features, freq_features
        )
        
        # ==================== 输出 ====================
        logits = self.classifier(fused_features)
        features = self.feature_head(fused_features)
        
        # return logits, features
        return logits