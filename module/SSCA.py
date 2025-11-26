
# -----------------------------
# SSCA module: Spectrum-Spatial Collaborative Attention
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSCA(nn.Module):
    """谱-空协同注意力模块（Spectrum-Spatial Collaborative Attention, SSCA）

    该模块通过并行的空间注意力分支和频率注意力分支，对输入特征图进行双通道增强：
    1. 空间分支：基于特征图的全局空间信息学习通道权重，突出空间维度上的关键特征；
    2. 频率分支：通过快速傅里叶变换（FFT）将特征转换到频率域，基于频率幅度信息学习通道权重，
       增强频率域中的有效特征成分；
    3. 最终通过元素相加融合两个分支的增强结果，同时捕获空间和频率维度的关键信息，提升特征判别力。

    核心设计理念：
    - 空间域注意力聚焦于“哪里有重要特征”，频率域注意力聚焦于“哪些频率成分是关键的”；
    - 双分支协同工作，既保留原始特征的空间结构信息，又利用频率域的全局相关性，适用于需要同时
      建模空间局部特征和全局频率特征的任务（如图像分类、目标检测、语义分割等）。

    模块结构：
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │ 输入特征 [B, C, H, W]                                       │
    ├───────────────────────┬─────────────────────────────────────┤
    │ 空间注意力分支         │ 频率注意力分支                      │
    │ 1. 全局空间平均池化    │ 1. 2D 实数快速傅里叶变换（rfft2）   │
    │ 2. MLP 学习通道权重    │ 2. 计算频率幅度（abs）             │
    │ 3. Sigmoid 激活       │ 3. 全局频率幅度平均池化            │
    │ 4. 特征加权（element-wise mul） │ 4. MLP 学习通道权重      │
    └───────────┬───────────┴───────────┬─────────────────────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                    双分支特征融合（element-wise add）
                            ▼
                    输出增强特征 [B, C, H, W]
    └─────────────────────────────────────────────────────────────┘
    ```
    """

    def __init__(self, channels: int, reduction: int = 16):
        """初始化 SSCA 模块的层结构和参数

        Args:
            channels: 输入特征图的通道数（对应输入张量 shape [B, C, H, W] 中的 C），
                即特征图的通道维度大小，决定了 MLP 层的输入输出维度。
            reduction: MLP 瓶颈层的通道压缩系数，默认值为 16。
                用于减少 MLP 层的参数量和计算量，具体为：
                - MLP 第一层将通道数从 C 压缩到 C//reduction；
                - 第二层再将通道数恢复到 C；
                建议根据输入通道数调整（通道数较大时可增大 reduction，如 32；较小时可减小，如 8）。

        Attributes:
            spatial_mlp: 空间分支的 MLP 网络，用于学习空间维度的通道注意力权重。
                由两层全连接层（Linear）和激活函数组成：
                - 第一层：Linear(channels, channels//reduction, bias=False) → 通道压缩
                - 第二层：ReLU(inplace=True) → 非线性激活
                - 第三层：Linear(channels//reduction, channels, bias=False) → 通道恢复
                - 第四层：Sigmoid() → 权重归一化到 [0,1]
            freq_mlp: 频率分支的 MLP 网络，用于学习频率维度的通道注意力权重。
                结构与 spatial_mlp 完全一致，参数独立初始化。
        """
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.freq_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """模块前向传播逻辑，实现空间-频率协同注意力增强

        具体步骤：
        1. 空间分支处理：
           a. 对输入特征图进行全局空间平均池化，压缩空间维度 [H, W] 到 1x1；
           b. 通过 spatial_mlp 学习通道权重，并reshape为与输入通道匹配的维度；
           c. 输入特征图与空间权重进行元素-wise 乘法，得到空间增强特征。

        2. 频率分支处理：
           a. 对输入特征图的空间维度执行 2D 实数快速傅里叶变换（rfft2），转换到频率域；
           b. 计算频率域特征的幅度（绝对值），保留频率强度信息；
           c. 对频率幅度进行全局平均池化，压缩频率-空间维度到通道维度；
           d. 通过 freq_mlp 学习频率域的通道权重，并reshape为匹配维度；
           e. 频率域特征与频率权重进行元素-wise 乘法（广播到全维度）；
           f. 执行逆快速傅里叶变换（irfft2），将加权后的频率域特征转换回空间域。

        3. 特征融合与输出：
           a. 空间增强特征与频率增强特征进行元素-wise 加法融合；
           b. 返回最终的增强特征图。

        Args:
            x: 输入特征图张量，shape 为 [B, C, H, W]，其中：
                - B: batch size（批次大小）
                - C: channels（通道数，需与 __init__ 中的 channels 参数一致）
                - H: height（特征图高度）
                - W: width（特征图宽度）
                数据类型需为 torch.Tensor（支持 CPU/GPU 设备）。

        Returns:
            torch.Tensor: 经过空间-频率协同注意力增强后的特征图，shape 与输入完全一致 [B, C, H, W]，
                数据类型与输入保持一致，保留原始特征的空间维度和通道维度。


            1. 傅里叶变换相关参数：
               - 使用 torch.fft.rfft2（实数输入的 FFT），仅输出非冗余的频率成分（节省计算量）；
               - 逆变换使用 torch.fft.irfft2，通过 s=(H, W) 指定输出空间维度，确保与输入一致；
               - norm='ortho' 启用正交归一化，保证 FFT 变换前后特征的能量守恒。
            2. 设备兼容性：模块会自动适配输入张量 x 所在的设备（CPU/GPU）。
            3. 数值稳定性：ReLU 激活使用 inplace=True 节省内存，Sigmoid 确保权重在 [0,1] 区间，避免梯度爆炸。
            4. 适用场景：适用于需要增强特征全局相关性的计算机视觉任务，如高分辨率图像识别、医学图像分割等，
               尤其对受噪声干扰或细节模糊的特征图有显著增强效果。
            5. 性能提示：频率域变换的计算复杂度为 O(H*W*log(H*W))，相比纯空间注意力略高，
               可通过调整 H/W（如特征图下采样）平衡性能与效果。
        """
        B, C, H, W = x.shape
        # Spatial branch: channel-wise global pooling -> MLP -> sigmoid -> channel weights
        spat = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)  # [B, C]：全局空间池化，压缩空间维度
        spat_w = self.spatial_mlp(spat).view(B, C, 1, 1)   # [B, C, 1, 1]：学习空间通道权重，reshape 适配广播
        x_spatial = x * spat_w  # 空间特征加权：突出空间关键通道

        # Spectral branch: rfft2 -> magnitude -> channel-wise pooling -> MLP -> sigmoid
        # Compute rfft2 on last two dims; result shape: [B, C, H, Wfreq]（Wfreq = W//2 + 1，非冗余频率成分）
        xf = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')  # 转换到频率域，输出复数张量
        xf_abs = torch.abs(xf)  # [B, C, H, Wfreq]：计算频率幅度（保留频率强度信息）
        # Global pooling on magnitude -> [B, C]：全局频率幅度池化，压缩频率-空间维度
        xf_pool = xf_abs.mean(dim=(-2, -1))  # 对 H（频率高度）和 Wfreq（频率宽度）求平均
        freq_w = self.freq_mlp(xf_pool).view(B, C, 1, 1)  # [B, C, 1, 1]：学习频率通道权重
        # Apply frequency weight in freq domain: multiply complex XF by scalar weight per channel
        xf_weighted = xf * freq_w.view(B, C, 1, 1)  # 频率域加权：广播权重到复数张量的每个元素
        # inverse transform：转换回空间域，确保输出维度与输入一致
        x_freq_att = torch.fft.irfft2(xf_weighted, s=(H, W), dim=(-2, -1), norm='ortho')
        # combine：双分支特征融合（元素相加，保留双方有效信息）
        out = x_spatial + x_freq_att
        return out