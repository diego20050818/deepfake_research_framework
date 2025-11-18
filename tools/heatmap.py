import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, List
from PIL import Image
import torchvision.transforms as T


class AttentionVisualizer:
    """
    用于可视化模型在指定层对输入图像的空间注意力（关注区域）。
    支持一键前向 + 钩子注册 + 可视化 + 自动清理。
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        初始化可视化器。

        Args:
            model (nn.Module): 要分析的模型
            device (torch.device, optional): 运行设备，默认自动检测
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

    def _prepare_input(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """将输入统一转换为 [1, C, H, W] 的 tensor"""
        if isinstance(image, Image.Image):
            transform = T.Compose([T.ToTensor()])
            tensor = transform(image)
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            raise TypeError("Input image must be a PIL.Image or torch.Tensor")

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # 添加 batch 维度
        return tensor.to(self.device)

    def _register_hook_and_forward(
        self, 
        input_tensor: torch.Tensor, 
        target_layer: str
    ) -> torch.Tensor:
        """注册钩子、执行前向、返回目标层输出"""
        features = {}

        def hook_fn(module, input, output):
            features['output'] = output.detach()

        # 获取目标模块
        named_modules = dict(self.model.named_modules())
        if target_layer not in named_modules:
            available = list(named_modules.keys())
            raise ValueError(
                f"Layer '{target_layer}' not found. "
                f"Available layers (first 10): {available[:10]}"
            )

        target_module = named_modules[target_layer]
        hook = target_module.register_forward_hook(hook_fn)

        # 前向传播
        with torch.no_grad():
            _ = self.model(input_tensor)

        # 清理钩子
        hook.remove()

        if 'output' not in features:
            raise RuntimeError("Hook failed to capture output.")
        return features['output']  # [B, C, H, W]

    def visualize_spatial_attention(
        self,
        image: Union[Image.Image, torch.Tensor],
        target_layer: str,
        title: str = "Spatial Attention",
        save_path: Optional[str] = None,
        show_input: bool = True
    ):
        """
        可视化模型在指定层的空间注意力（通道平均），反映“模型关注哪里”。

        Args:
            image: 输入图像（PIL 或 tensor）
            target_layer: 要监控的层名（如 'layer4.0.conv1'）
            title: 图标题
            save_path: 保存路径（None 则显示）
            show_input: 是否同时显示原始图像（左右对比）
        """
        input_tensor = self._prepare_input(image)
        feature_map = self._register_hook_and_forward(input_tensor, target_layer)

        attention = feature_map.mean(dim=1)[0].cpu().numpy()  # [H, W]

        if show_input:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # 显示原图
            if isinstance(image, Image.Image):
                axes[0].imshow(image)
            else:
                img_to_show = image if image.dim() == 3 else image[0]
                axes[0].imshow(img_to_show.permute(1, 2, 0).cpu().numpy())
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            # 显示注意力图
            sns.heatmap(attention, cmap='jet', cbar=True, square=True, ax=axes[1])
            axes[1].set_title(title)
            axes[1].axis("off")
        else:
            plt.figure(figsize=(6, 5))
            sns.heatmap(attention, cmap='jet', cbar=True, square=True)
            plt.title(title)
            plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def visualize_feature_maps(
        self,
        image: Union[Image.Image, torch.Tensor],
        target_layer: str,
        title: str = "Feature Maps",
        save_path: Optional[str] = None,
        max_channels: int = 64
    ):
        """
        可视化指定层的所有（或部分）通道特征图。
        """
        input_tensor = self._prepare_input(image)
        feature_map = self._register_hook_and_forward(input_tensor, target_layer)

        self._plot_feature_maps(feature_map, title, save_path, max_channels)

    @staticmethod
    def _plot_feature_maps(
        feature_map: torch.Tensor,
        title: str = "",
        save_path: Optional[str] = None,
        max_channels: int = 64
    ):
        """静态方法：绘制特征图网格"""
        if feature_map.dim() == 4:
            feature_map = feature_map[0]  # [C, H, W]
        if feature_map.size(0) > max_channels:
            feature_map = feature_map[:max_channels]
        n_channels = feature_map.size(0)

        n_cols = min(8, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8))
        if n_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(n_channels):
            sns.heatmap(
                feature_map[i].cpu().numpy(),
                ax=axes[i],
                cmap='viridis',
                cbar=False,
                xticklabels=False,
                yticklabels=False
            )
            axes[i].set_title(f'Ch{i}', fontsize=8)

        for i in range(n_channels, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def list_layers(self, keyword: Optional[str] = None) -> List[str]:
        """
        列出模型中所有层名称（可选按关键词过滤），方便查找 target_layer。
        """
        layers = list(dict(self.model.named_modules()).keys())
        if keyword:
            layers = [name for name in layers if keyword in name]
        return layers
    
if __name__ == '__main__':
    from torchvision.models import resnet18
    from PIL import Image
    # 1. 加载模型和图像
    model = resnet18(pretrained=True)
    img_path = 'dataset/Celeb-DF/00000/00000_frame000000.jpg'
    img = Image.open(img_path).convert("RGB")

    # 2. 创建可视化器
    vis = AttentionVisualizer(model)

    # （可选）查看有哪些层可用
    print(vis.list_layers("layer4"))  # 输出: ['layer4', 'layer4.0', 'layer4.0.conv1', ...]

    # 3. 可视化空间注意力（带原图对比）
    vis.visualize_spatial_attention(
        image=img,
        target_layer="layer4.1.bn2",
        title="Where does ResNet look at?",
        show_input=True,
        save_path='VSA'
    )

    # 4. 也可单独看特征图
    vis.visualize_feature_maps(
        image=img,
        target_layer="layer4.1.bn2",
        max_channels=16,
        save_path='VFM'
    )