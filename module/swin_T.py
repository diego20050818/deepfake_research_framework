import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

class Mlp(nn.Module):
    """多层感知机模块"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    """窗口注意力机制"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # 生成相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
            
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        # 兼容输入 token 数与记录的 input_resolution 不一致的情况（例如多尺度输入）
        if H * W != L:
            side = int(L ** 0.5)
            H = W = side

        H_orig, W_orig = H, W  # 原始尺寸（用于最终 reshape/crop）

        shortcut = x
        x = self.norm1(x)

        # reshape -> [B, H, W, C]
        x = x.view(B, H, W, C)

        # 对 H/W 按 window_size 做 padding，保证能整除 window_size
        pad_h = (self.window_size - (H % self.window_size)) % self.window_size
        pad_w = (self.window_size - (W % self.window_size)) % self.window_size
        if pad_h != 0 or pad_w != 0:
            x_nc = x.permute(0, 3, 1, 2).contiguous()  # B,C,H,W
            x_nc = F.pad(x_nc, (0, pad_w, 0, pad_h))
            Hp = H + pad_h
            Wp = W + pad_w
            x = x_nc.permute(0, 2, 3, 1).contiguous()  # B,Hp,Wp,C
        else:
            Hp, Wp = H, W

        # 移位窗口（基于 pad 后的尺寸）
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口划分 -> 窗口注意力
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, ws*ws, C)

        attn_windows = self.attn(x_windows)

        # 窗口合并
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B, Hp, Wp, C

        # 移位还原
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 去掉 padding 区域（若有）
        if pad_h != 0 or pad_w != 0:
            x = x[:, :H_orig, :W_orig, :].contiguous()

        # 回到 [B, H*W, C]
        x = x.view(B, H_orig * W_orig, C)

        # 残差与 MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# ...existing code...
class PatchMerging(nn.Module):
    """Patch合并层 - 下采样（支持奇偶尺寸自动 pad 修复）"""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: [B, L, C], L = H*W (可能与 self.input_resolution 不完全匹配)
        返回: [B, L//4 (rounded), 2*C]
        实现要点：
        - 若推断得到的 H/W 与实际 L 不符，则以 sqrt(L) 回退
        - 当 H 或 W 为奇数时，对右/下侧进行 padding，保证四个子块尺寸一致，避免 concat 时尺寸不匹配
        """
        B, L, C = x.shape

        # 尝试使用给定 input_resolution 作为 H,W
        H, W = self.input_resolution
        if H * W != L:
            # fallback: 根据 L 推断方形尺寸
            side = int(L ** 0.5)
            H = W = side

        # reshape -> [B, H, W, C]
        x = x.view(B, H, W, C)

        # 如果 H 或 W 为奇数，进行右/下方向 pad（在通道维度转换后使用 F.pad）
        pad_h = H % 2
        pad_w = W % 2
        if pad_h != 0 or pad_w != 0:
            # 转为 [B, C, H, W] 以便使用 F.pad
            x_nc = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            # pad 格式 (pad_left, pad_right, pad_top, pad_bottom)
            x_nc = F.pad(x_nc, (0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
            x = x_nc.permute(0, 2, 3, 1).contiguous()  # B, H, W, C

        # 抽取 2x2 子块（现在 H,W 一定为偶数）
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C

        # concat channels -> [B, H/2, W/2, 4*C]
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1)
        new_H = x_cat.shape[1]
        new_W = x_cat.shape[2]

        # flatten -> [B, N', 4*C]
        x_flat = x_cat.view(B, new_H * new_W, 4 * C)
        x_norm = self.norm(x_flat)
        x_reduced = self.reduction(x_norm)  # [B, N', 2*C]
        return x_reduced
# ...existing code...

class BasicLayer(nn.Module):
    """基础层 - 包含多个Swin块"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 构建Swin块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformer(nn.Module):
    """完整的Swin-T模型"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # 分块嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置编码
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建层次结构
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                 patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        # 分类头
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class SwinTransformerEncoder(nn.Module):
    """
    基于仓内 SwinTransformer 的编码器封装。
    forward 返回分阶段的特征列表：[stage1, stage2, stage3, stage4]
    每个 stage 的输出形状为 [B, C, H, W]
    """
    def __init__(self, img_size=224, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, patch_size=4, in_chans=3, **kwargs):
        super().__init__()
        # 使用 SwinTransformer 构建基础网络
        self.model = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # 不使用分类头
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            **kwargs
        )
        # 便利引用
        self.patch_embed = self.model.patch_embed
        self.layers = self.model.layers
        # 初始patch分辨率 (H', W')
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = (int(patches_resolution[0]), int(patches_resolution[1]))
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: [B, 3, H, W]
        返回: list of 4 tensors, 每个 [B, C, H_l, W_l]
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, C]
        # 不使用 pos_drop / classification head，这里直接走各层并记录中间输出
        features = []
        cur = x
        for i, layer in enumerate(self.layers):
            cur = layer(cur)  # BasicLayer 的输出仍为 [B, N_i, C_i]
            # 计算当前层对应的空间分辨率
            H_l = self.patches_resolution[0] // (2 ** i)
            W_l = self.patches_resolution[1] // (2 ** i)
            # N_i = H_l * W_l
            N_i = cur.shape[1]
            # 有时 PatchMerging 会调整 token 数及通道，这里尝试按 H_l*W_l 重塑：
            C_i = cur.shape[-1]
            if N_i != H_l * W_l:
                # 若 token 数与计算不符，尽量通过推断 H_l/W_l 进行 reshape
                # fallback: 设置 H_l = W_l = int(sqrt(N_i))
                side = int(N_i ** 0.5)
                H_l = W_l = side
            feat = cur.view(B, H_l, W_l, C_i).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            features.append(feat)
        # 如果层数不足 4，则通过复制末层填充到 4
        while len(features) < 4:
            features.append(features[-1])
        return features

# 辅助函数 - PatchEmbed 补全
class PatchEmbed(nn.Module):
    """图像分块嵌入"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # 确保img_size和patch_size是元组
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        # 计算patch数量和分辨率
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x的形状: (B, C, H, W)
        B, C, H, W = x.shape
        # 使用conv将图像分块并嵌入到embed_dim维度
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x

# 辅助函数 - window_partition
def window_partition(x, window_size):
    """将特征图划分为窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

# 辅助函数 - window_reverse
def window_reverse(windows, window_size, H, W):
    """将窗口合并回特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# 辅助函数 - DropPath
class DropPath(nn.Module):
    """随机路径丢弃 (Stochastic Depth)"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 为每个样本生成一个随机数，决定是否丢弃
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0或1
        output = x.div(keep_prob) * random_tensor
        return output

# 预定义的Swin-T配置
def swin_tiny(**kwargs):
    """Swin-T微小版本 (Swin-T)"""
    model = SwinTransformer(
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4., **kwargs)
    return model

def swin_small(**kwargs):
    """Swin-S小版本 (Swin-S)"""
    model = SwinTransformer(
        embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4., **kwargs)
    return model

def swin_base(**kwargs):
    """Swin-B基础版本 (Swin-B)"""
    model = SwinTransformer(
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=7, mlp_ratio=4., **kwargs)
    return model

# 示例：如何使用
if __name__ == '__main__':
    # 创建一个Swin-T模型实例
    model = swin_tiny(img_size=224, num_classes=1000)
    print("Swin-T模型结构：")
    # print(model) # 如果想看完整模型结构可以取消注释
    
    # 随机生成一个输入图像
    input_tensor = torch.randn(1, 3, 224, 224) 
    
    # 前向传播
    output = model(input_tensor)
    print(f"\n输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}") 
    
    # 测试一下Swin-B
    model_b = swin_base(img_size=224, num_classes=10)
    output_b = model_b(input_tensor)
    print(f"\nSwin-B输出形状: {output_b.shape}")