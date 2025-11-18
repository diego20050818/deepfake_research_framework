# -----------------------------
# DCTConvBlock (approx): we implement depthwise convs as in your graph
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DCTConvBlock(nn.Module):
    """DCT 卷积块（DCTConvBlock）

    该模块是一个基于深度可分离卷积的特征提取单元，用于对输入特征图进行多尺度空间特征提取与通道维度调整。
    核心由深度可分离卷积（7x7 + 3x3）、逐点卷积、批归一化和全局平均池化组成，最终输出固定为 1x1 尺寸的特征，
    便于后续的通道级或全局特征融合操作。

    模块结构：
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │ 输入特征 [B, in_ch, H, W]                                    │
    ├─────────────────────────────────────────────────────────────┤
    │ 深度可分离卷积 7x7（分组卷积，groups=in_ch）→ GELU 激活        │
    ├─────────────────────────────────────────────────────────────┤
    │ 深度可分离卷积 3x3（分组卷积，groups=in_ch）→ 逐点卷积 1x1     │
    ├─────────────────────────────────────────────────────────────┤
    │ 批归一化（BatchNorm2d）→ 全局自适应平均池化（AdaptiveAvgPool2d）│
    └─────────────────────────────────────────────────────────────┘
    │ 输出特征 [B, hidden_ch, 1, 1]                                │
    └─────────────────────────────────────────────────────────────┘
    ```
    """

    def __init__(self, in_ch: int, hidden_ch: Optional[int] = None):
        """初始化 DCT 卷积块的层结构和参数

        Args:
            in_ch: 输入特征图的通道数（对应输入张量 shape [B, in_ch, H, W] 中的 in_ch）。
            hidden_ch: 输出特征图的通道数，若为 None 则默认与输入通道数 in_ch 相同，
                用于调整特征的通道维度（如特征压缩或扩张）。

        Attributes:
            block: 由多个层组成的序列容器（nn.Sequential），包含以下子层：
                - nn.Conv2d（depthwise 7x7）：深度可分离卷积，分组数等于输入通道数，
                  用于捕获大尺度空间特征，padding=3 保证输入输出空间维度一致；
                - nn.GELU：高斯误差线性单元激活函数，引入非线性并增强梯度流动；
                - nn.Conv2d（depthwise 3x3）：深度可分离卷积，进一步提取细粒度空间特征；
                - nn.Conv2d（pointwise 1x1）：逐点卷积，调整通道维度到 hidden_ch；
                - nn.BatchNorm2d：批归一化层，加速训练收敛并提升数值稳定性；
                - nn.AdaptiveAvgPool2d(1)：全局自适应平均池化，将特征图压缩为 1x1 尺寸。
        """
        super().__init__()
        if hidden_ch is None:
            hidden_ch = in_ch
        self.block = nn.Sequential(
            # 深度可分离卷积 7x7：分组卷积，groups=in_ch，stride=1，padding=3 保证 H/W 不变
            nn.Conv2d(in_ch, in_ch, kernel_size=7, stride=1, padding=3, groups=in_ch),
            nn.GELU(),
            # 深度可分离卷积 3x3：分组卷积，groups=in_ch，padding=1 保证 H/W 不变
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            # 逐点卷积 1x1：调整通道数到 hidden_ch
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1),
            nn.BatchNorm2d(hidden_ch),
            # 全局自适应平均池化：输出尺寸固定为 1x1
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """模块前向传播逻辑，执行特征提取与维度压缩

        Args:
            x: 输入特征图张量，shape 为 [B, in_ch, H, W]，其中：
                - B: batch size（批次大小）
                - in_ch: 输入通道数（需与 __init__ 中的 in_ch 参数一致）
                - H: height（特征图高度）
                - W: width（特征图宽度）
                数据类型需为 torch.Tensor

        Returns:
            torch.Tensor: 经过特征提取和池化后的输出特征图，shape 为 [B, hidden_ch, 1, 1]，
                数据类型与输入保持一致，空间维度被压缩为 1x1，便于后续全局特征操作。

        Notes:
            1. 深度可分离卷积的计算复杂度远低于普通卷积，适合对计算资源敏感的场景；
            2. GELU 激活函数在 Transformer 和现代 CNN 中表现优异，兼具非线性和平滑性；
            3. 全局平均池化后输出固定为 1x1 尺寸，可作为“特征向量”直接输入全连接层或注意力模块；
            4. 设备兼容性：模块会自动适配输入张量 x 所在的设备（CPU/GPU）
        """
        return self.block(x)
