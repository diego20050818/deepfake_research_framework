import torch
import os

from ruamel.yaml import YAML
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from loguru import logger
from pprint import pprint
from pathlib import Path
from datetime import datetime

from tools.utils import print_config,print_model_summary,yaml_to_string
from tools.dataset_loader import get_dataloader,LimitedDataLoader
from tools.image_preprocess import transforms_train
from tools.trainer import train_model
from tools.validation import ModelValidator

from model import RINEPlusSSCA,FaceAntiSpoofingViT,MultiScaleHierarchicalTransformer

base_config_path = 'config/config.yaml'
train_config_path = 'config/train_config.yaml'

yaml = YAML()

with open(base_config_path,'r',encoding='utf-8') as f:
    base_config:dict = yaml.load(f)

with open(train_config_path,'r',encoding='utf-8') as f:
    train_config:dict = yaml.load(f)

logger.info("load base config and train config successfully")

print_config(base_config,'基础')
print_config(train_config,'训练')

# 定义设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------
# 数据集加载
# -----------------
dataset_root = Path(base_config.get('dataset_root','./dataset'))

if base_config['validation']:
    train_dataset, test_dataset,val_dataset = get_dataloader(
        validation=True,
        dataset_root=dataset_root,
        dataset_names=base_config.get('dataset_path',['Celeb-DF']),
        batch_size=train_config.get('batch_size',16),
        transform=transforms_train,
        split=train_config.get('dataset_split',0.8)
    )
    # val_dataset = LimitedDataLoader(val_dataset,10)
else:
        train_dataset, test_dataset = get_dataloader(
        dataset_root=dataset_root,
        dataset_names=base_config.get('dataset_path',['Celeb-DF']),
        batch_size=train_config.get('batch_size',16),
        transform=transforms_train,
        split=train_config.get('dataset_split',0.8)
        )


# train_dataset = LimitedDataLoader(train_dataset,50)
# test_dataset  = LimitedDataLoader(test_dataset,10)


# -----------------
# 模型定义
# -----------------
# model = RINEPlusSSCA().to(device)
# model = FaceAntiSpoofingViT().to(device)
model = MultiScaleHierarchicalTransformer().to(device)


logger.info("load model successfully")
# pprint(model)

time_now = datetime.now().strftime("%Y年%m月%d日 > %H:%M")
log_dir = f'./runs/experiment > {time_now}'
base_config_str = yaml_to_string(base_config)
train_config_str = yaml_to_string(train_config)

writer = SummaryWriter(log_dir=log_dir)
writer.add_text('base_config', base_config_str)
writer.add_text('train_config', train_config_str)

# 在 train_model 调用之前添加以下代码
dummy_input = torch.randn(1, 3, 224, 224).to(device) 
try:
    writer.add_graph(model, dummy_input)
except Exception as e:
    logger.warning(f"Failed to add graph to TensorBoard: {e}")

chechpoint_path = Path(base_config.get('model_save_path','checkpoint')) / time_now
chechpoint_path.mkdir(parents=True,exist_ok=True)
logger.info("begin training")

# -----------------
# 开始训练
# -----------------
result = train_model(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    train_config=train_config,
    device=device, # type: ignore
    writer=writer,
    checkpoint_path=chechpoint_path
)

if base_config['validation']:
    time_now = datetime.now().strftime("%Y年%m月%d日 > %H:%M")
    log_dir = f'./runs/validation > {time_now}'
    checkpoint_path = result.get('best_checkpoint')

    checkpoint_info = {
        'name': checkpoint_path,
        'dataset':base_config.get('dataset_path')
    }

    validator = ModelValidator(
        model_path=checkpoint_path,
        model=model,
        chechpoint_info=checkpoint_info,
        dataloader=val_dataset, # type:ignore
        class_names=['fake', 'real'],  
        log_dir=log_dir
    )
    # 运行验证
    validator.run_validation()







