"""
完善图片批量读取 批量图片处理和构建数据集的方法

"""

import sys
import os
sys.path.append(os.getcwd())

import torch 
from torch.utils.data import DataLoader,Dataset,dataloader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from loguru import logger
from PIL import Image
from pathlib import Path
from typing import Optional,Tuple,List
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import random

from tools.image_preprocess import transforms_train,transforms_val

# logger.level('INFO')


class CustomImageDataset(Dataset):
    def __init__(self, dataset_root:Optional[Path|str],
                 dataset_names:list[str],
                 labels_file_name:str='labels.csv', 
                 transform=None):
        """_summary_

        Args:
            dataset_root (Path): 数据集根目录
            dataset_names (list[str]): 数据集名称
            labels_file_name (str, optional): 存储数据集路(path)和标签字段(label)的文件 Defaults to 'labels.csv'.
            transform (_type_, optional): 数据集变换策略. Defaults to None.

        Raises:
            FileNotFoundError
        """
        if isinstance(dataset_root,Path):
            self.dataset_root = dataset_root
        else:
            self.dataset_root = Path(data_root)

        self.dataset_names = dataset_names
        self.labels_file_name = labels_file_name
        self.transform = transform

        if not self.dataset_root.exists(): 
            raise FileNotFoundError('dataset root not found:',self.dataset_names)
        
        self.all_img_paths = list()
        self.all_img_labels = list()
        
        for dataset_name in tqdm(self.dataset_names,desc='loading dataset'):
            tqdm.write(f"current dataset:{dataset_name}")
            dataset_path = self.dataset_root / dataset_name
            datset_label_file = dataset_path / self.labels_file_name

            if not dataset_path.exists():
                raise FileNotFoundError('dataset not found:',dataset_name)
            if not datset_label_file.is_file():
                raise FileNotFoundError('dataset labels file not found:',datset_label_file)
            
            with open(datset_label_file,'r',encoding='utf-8') as f:
                dataset_info = pd.read_csv(f,header=0)

            img_paths = dataset_info['path']
            img_labels = dataset_info['label'].astype(np.int16)


            self.all_img_paths.extend(img_paths)
            self.all_img_labels.extend(img_labels)

        # 统计总样本数和负样本数
        total_samples = len(self.all_img_labels)
        negative_count = self.all_img_labels.count(0)

        # 计算占比
        negative_ratio = (negative_count / total_samples) * 100 if total_samples > 0 else 0.0
        positive_ratio = 100.0 - negative_ratio  # 正样本占比 = 100% - 负样本占比

        logger.info(
            f"there are {negative_ratio:.2f}% negative samples and {positive_ratio:.2f}% positive samples"
        )
        self.all_img_labels = torch.tensor(self.all_img_labels)
        
        logger.info(f"successfully loaded {len(self.all_img_paths)} samples (*^▽^*)")

            
    def __len__(self):
        """获取数据集长度

        Returns:
            _type_: _description_
        """
        return len(self.all_img_paths)
    
    def __getitem__(self, idx:int):
        """指定索引访问数据的方法

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        img_path:Optional[str|Path] = self.all_img_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error("failed to covert img to RGB:",e)
            raise IOError(f"{img_path}")
        
        label = self.all_img_labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
@logger.catch()
def get_dataloader(
    dataset_root: Path,
    dataset_names: List[str],
    batch_size: int,
    transform: transforms.Compose,
    split: float = 0.8, # 训练集占总数据集的比例 (Train: 0.8, Test: 0.2)
    validation: bool = False, # 是否进行三方分割 (Train/Val/Test)
    val_ratio_of_remainder: float = 0.5, # 如果 validation=True, 剩余部分中分给验证集的比例
    random_seed: int = 42,
    num_works: int = 4,
    labels_file_name: str = 'labels.csv'
) -> Tuple[DataLoader, ...]:
    """
    创建完整数据集并分割为训练集、(验证集) 和测试集，返回对应的 DataLoader。

    Args:
        dataset_root (Path): 数据集根目录
        dataset_names (List[str]): 数据集根目录下的数据集名称
        batch_size (int): 批次大小
        transform (transforms.Compose): 图像变换方法
        split (float, optional): 训练集占总数据集的比例. Defaults to 0.8.
        validation (bool, optional): 是否进行三方分割 (Train/Val/Test). Defaults to False.
        val_ratio_of_remainder (float, optional): 如果 validation=True, 剩余部分中分给验证集的比例. Defaults to 0.5.
        random_seed (int, optional): 随机种子（保证分割可复现）. Defaults to 42.
        num_works (int, optional): 并行数量. Defaults to 4.
        labels_file_name (str, optional): 数据集目录下的标签文件名称. Defaults to 'labels.csv'.

    Returns:
        Tuple[DataLoader, ...]: 
            如果 validation=False: (train_dataloader, test_dataloader)
            如果 validation=True: (train_dataloader, val_dataloader, test_dataloader)
    """
    # 验证分割比例有效性
    if not (0 < split < 1):
        raise ValueError(f"split 必须在 (0, 1) 范围内，当前值: {split}")
    if validation and not (0 < val_ratio_of_remainder < 1):
        raise ValueError(f"val_ratio_of_remainder 必须在 (0, 1) 范围内，当前值: {val_ratio_of_remainder}")
    
    start_time = time.time()
    
    # 1. 创建完整数据集
    full_dataset = CustomImageDataset(
        dataset_root=dataset_root,
        dataset_names=dataset_names,
        labels_file_name=labels_file_name,
        transform=transform
    )
    dataset_size = len(full_dataset)
    logger.info(f"完整数据集大小: {dataset_size}")
    
    # 设置随机种子保证可复现性
    random.seed(random_seed)
    torch.manual_seed(random_seed) 
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    # 2. 数据集分割
    
    # A. 训练集大小
    train_size = int(dataset_size * split)
    train_indices = indices[:train_size]
    
    # B. 剩余部分
    remainder_indices = indices[train_size:]
    remainder_size = len(remainder_indices)
    
    val_dataset = None
    val_dataloader = None
    
    if validation:
        # 三方分割：Train / Val / Test
        
        # 剩余部分按 val_ratio_of_remainder 分割给 Val 和 Test
        val_size = int(remainder_size * val_ratio_of_remainder)
        
        val_indices = remainder_indices[:val_size]
        test_indices = remainder_indices[val_size:]
        
        # 创建验证集
        val_dataset = Subset(full_dataset, val_indices)
        logger.info(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}, 测试集大小: {len(test_indices)}")
    else:
        # 二方分割：Train / Test
        test_indices = remainder_indices # 剩余部分全部作为测试集
        logger.info(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")

    # 创建训练集和测试集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 3. 定义 collate_fn 处理损坏文件
    def collate_fn(batch):
        # 过滤掉 None/损坏的样本
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            # 返回 None, None 让上层调用者可以跳过这个批次
            return None, None 
        return default_collate(batch)
    
    # 4. 构建 DataLoader
    
    # 训练集 DataLoader（打乱）
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 验证集 DataLoader（不打乱）
    if validation:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_works,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # 测试集 DataLoader（不打乱）
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    end_time = time.time()
    logger.info(f"数据集加载与分割完成，总耗时: {end_time - start_time:.2f}s")
    
    if validation:
        return (train_dataloader, val_dataloader, test_dataloader)
    else:
        return (train_dataloader, test_dataloader)

@logger.catch()
def get_validation_dataloader(
        dataset_root: Path,
        dataset_names: List[str],
        batch_size: int,
        transform: transforms.Compose,
        num_works: int = 4,
        labels_file_name: str = 'labels.csv'
) -> DataLoader:
    """
    创建完整数据集作为验证集，返回对应的 DataLoader

    Args:
        dataset_root (Path): 数据集根目录
        dataset_names (List[str]): 数据集根目录下的数据集名称
        batch_size (int): 批次大小
        transform (transforms.Compose): 图像变换方法
        num_works (int, optional): 并行数量. Defaults to 4.
        labels_file_name (str, optional): 数据集目录下的标签文件名称. Defaults to 'labels.csv'.

    Returns:
        DataLoader: 验证集 DataLoader
    """
    
    start_time = time.time()
    
    # 1. 创建完整数据集
    full_dataset = CustomImageDataset(
        dataset_root=dataset_root,
        dataset_names=dataset_names,
        labels_file_name=labels_file_name,
        transform=transform
    )
    logger.info(f"完整验证集大小: {len(full_dataset)}")
    

    def collate_fn(batch):
        # 过滤掉 None/损坏的样本
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None, None
        return default_collate(batch)
    

    val_dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    
    end_time = time.time()
    logger.info(f"验证集加载完成，总耗时: {end_time - start_time:.2f}s")
    
    return val_dataloader

from itertools import islice

class LimitedDataLoader:
    def __init__(self, dataloader, max_batches):
        """限制数据集 对数据集进行切片

        Args:
            dataloader (dataloader): 数据集实例
            max_batches (int): 前n个批次
        """
        self.dataloader = dataloader
        self.max_batches = max_batches
    
    def __iter__(self):
        return islice(self.dataloader, self.max_batches)
    
    def __len__(self):
        return min(self.max_batches, len(self.dataloader))


if __name__ == '__main__':
    from tools.image_preprocess import transforms_train
    data_root = Path('dataset')
    data_name = ['Celeb-DF','Celeb-DF-v2']

    train_dataset,test_dataset = get_dataloader(
        dataset_names=data_name,
        dataset_root=data_root,
        batch_size=16,
        transform=transforms_train,
    )
    logger.debug(train_dataset)


    




    
    


