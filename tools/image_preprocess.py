import torch
from torchvision import transforms
from tools.utils import crop_face_with_mtcnn


transforms_train = transforms.Compose([     # NOTE 训练图像增强方法
    transforms.Lambda(lambda img: crop_face_with_mtcnn(img)),  # MTCNN 裁剪
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(degrees=10),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomPerspective(distortion_scale=0.1, p=0.2),  # 随机透视变换
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机仿射变换
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])

# 验证集和测试集：MTCNN 裁剪 + 中心裁剪
transforms_val = transforms.Compose([       # NOTE 验证和测试集图像增强
    transforms.Lambda(lambda img: crop_face_with_mtcnn(img)),  # MTCNN 裁剪
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])





