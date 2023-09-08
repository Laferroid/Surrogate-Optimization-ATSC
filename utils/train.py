# %%
# region import
import sys
import os
import datetime
import torch
import numpy as np
import yaml
import argparse

# 工作区内的绝对引用，工作区文件夹路径添加到虚拟环境的环境变量PYTHONPATH中
from helper_func import parse_config
from models.uncertainty_surrogate_model import UncertaintySurrogateModel
from models.model_utils import train, MyDataset, get_dataloader

# endregion

# %% 训练模型
# Windows上多进程的实现问题。在Windows上，子进程会自动import启动它的文件，而在import的时候会执行这些语句，就会无限递归创建子进程报错。
if __name__ == "__main__":
    default_config_dir = "../configs/default_config.yaml"
    updated_config_dir = "../configs/updated_config.yaml"

    config = parse_config(default_config_dir, updated_config_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    dataset = MyDataset(config)
    print(len(dataset))
    train_dl, val_dl, test_dl = get_dataloader(dataset, config)

    model = UncertaintySurrogateModel(config)
    train(model, train_dl, val_dl, config)

    print("finished !")
