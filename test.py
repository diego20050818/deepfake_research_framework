import torch
import torch.nn as nn
from loguru import logger
from ruamel.yaml import YAML
from pathlib import Path
from datetime import datetime

from tools.validation import ModelValidator
from tools.dataset_loader import get_validation_dataloader,LimitedDataLoader
from tools.image_preprocess import transforms_val
from model import RINEPlusSSCA, FaceAntiSpoofingViT


config_path = 'config/val_config.yaml'

yaml = YAML()
with open(config_path,'r',encoding='utf-8') as f:
    config = yaml.load(f)

# --------
# 加载数据
# --------
dataset_root = Path(config.get('dataset_root','./dataset'))

val_dataset = get_validation_dataloader(
    dataset_root=dataset_root,
    dataset_names=config.get('dataset_path',['FF++']),
    batch_size=config.get('batch_size',16),
    transform=transforms_val,
)
limited_dataset = LimitedDataLoader(dataloader=val_dataset,max_batches=10)

# --------
# 加载checkpoint和model
# --------

checkpoint_path = config.get('checkpoint_path')
model = RINEPlusSSCA()
# model = FaceAntiSpoofingViT()
time_now = datetime.now().strftime("%Y年%m月%d日 > %H:%M")
log_dir = f'./runs/validation > {time_now}'

checkpoint_info = {
    'name': checkpoint_path,
    'dataset':config.get('dataset_path')
}

validator = ModelValidator(
    model_path=checkpoint_path,
    model=model,
    chechpoint_info=checkpoint_info,
    dataloader=limited_dataset, # type:ignore
    class_names=['fake', 'real'],  # 根据实际情况设置类别名称
    log_dir=log_dir
)

# 运行验证
validator.run_validation()
