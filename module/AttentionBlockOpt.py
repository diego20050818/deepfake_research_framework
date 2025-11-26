# ------------
# 经优化后的attention blockes
# ------------

import torch
import torch.nn as nn
from typing import Literal
from einops import rearrange
from einops.layers.torch import Rearrange

class SpatialAttention(nn.Module):
    """
    空间注意力模块 - 学习特征图中不同空间位置的重要性权重
    用于增强重要区域的特征，抑制不重要区域
    """
    def __init__(self, dim):
        super().__init__()
        # 3x3卷积学习空间注意力图
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        # Sigmoid激活函数，将权重限制在[0,1]范围内
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图，形状为 [B, C, H, W]
        Returns:
            加权后的特征图，形状为 [B, C, H, W]
        """
        # 计算注意力权重图：[B, C, H, W] -> [B, C, H, W] (值在0-1之间)
        att = self.sigmoid(self.conv(x))
        # 特征图与注意力权重逐元素相乘
        return x * att
    
class AttentionBlock(nn.Module):
    """全局多头自注意力模块"""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LocalAttention2D(nn.Module):
    """2D网格的局部窗口注意力模块"""
    
    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        self.attention = AttentionBlock(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        B, H, W, C = x.shape

        # 检查特征图尺寸是否足够容纳窗口
        if H < self.kernel_size or W < self.kernel_size:
            return x  # 若尺寸不足，直接返回输入（避免报错）

        x = rearrange(x, "B H W C -> B C H W")
        patches = self.unfold(x)
        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        patches = self.norm(patches)
        out = self.attention(patches)

        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        fold = nn.Fold(
            output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride
        )
        out = fold(out)

        # 重叠区域归一化
        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))
        norm = fold(norm)
        out = out / norm

        out = rearrange(out, "B C H W -> B H W C")
        return out


class Multipole_Attention2D(nn.Module):
    """2D多极注意力模块"""
    
    def __init__(
        self,
        image_size,
        in_channels,
        local_attention_kernel_size,
        local_attention_stride,
        downsampling: Literal["avg_pool", "conv"],
        upsampling: Literal["avg_pool", "conv"],
        sampling_rate,
        heads,
        dim_head,
        dropout,
        channel_scale,
    ):
        super().__init__()
        self.kernel_size = local_attention_kernel_size
        
        # 动态计算最大可能的层级数（确保下采样后特征图不小于窗口大小）
        self.levels = 0
        current_size = image_size
        while True:
            current_size = current_size // sampling_rate
            if current_size >= self.kernel_size:
                self.levels += 1
            else:
                break
        
        # 至少保留1个层级（避免层级数为0）
        self.levels = max(1, self.levels)

        channels_conv = [in_channels * (channel_scale ** i) for i in range(self.levels)]

        self.attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=int(channels_conv[0]),
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # 下采样模块
        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )
        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

        # 上采样模块
        if upsampling == "avg_pool":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )
        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        # 调整输入维度以匹配模型要求 [B, H, W, C]
        x = rearrange(x, "B C H W -> B H W C")

        x_in = x
        x_out = []

        # 计算各层级的局部注意力
        x_out.append(self.attention(x_in))
        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.attention(x_in)
            x_out.append(x_out_down)

        # 多尺度特征融合
        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        # 恢复输出维度为 [B, C, H, W]
        return rearrange(res, "B H W C -> B C H W")
    
class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    
    def __init__(self, global_dim, local_dim, freq_dim, out_dim):
        super().__init__()
        
        self.global_proj = nn.Linear(global_dim, out_dim)
        self.local_proj = nn.Linear(local_dim, out_dim)
        self.freq_proj = nn.Linear(freq_dim, out_dim)
        
        self.cross_attn = nn.MultiheadAttention(out_dim, num_heads=8, batch_first=True)
        
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, global_feat, local_feat, freq_feat):
        q = self.global_proj(global_feat).unsqueeze(1)  # [B, 1, D]
        k = self.local_proj(local_feat).unsqueeze(1)    # [B, 1, D]
        v = self.freq_proj(freq_feat).unsqueeze(1)      # [B, 1, D]
        
        attended, _ = self.cross_attn(q, k, v)
        attended = attended.squeeze(1)
        
        concat_features = torch.cat([global_feat, local_feat, freq_feat], dim=1)
        gate_weights = self.gate(concat_features)
        
        fused = gate_weights * attended + (1 - gate_weights) * global_feat
        
        return fused