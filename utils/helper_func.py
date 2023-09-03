# %%
# region import
import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch

# endregion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def params_count(model):
    """
    Compute the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


# configuration优先级: cmd>hard code>default
def parse_config(default_config_dir, updated_config=None):
    # 1. load default configs
    with open(default_config_dir, "r+") as f:
        config = yaml.safe_load(f)

    # 2. update hardcode configs
    config.update(updated_config)

    # 3. cmd arguments
    parser = argparse.ArgumentParser("parse cmd arguments")
    parser.add_argument("--exp_group", help="the name of experiment group", type=str, default="test")
    parser.add_argument("--exp_name", help="the experiment id", type=str, default="baseline")
    parser.add_argument("--epochs", help="training epoch number", type=int, default=8)
    # parse_known_args返回两个元素，第一个为所求的NameSpace，第二个是unknown args的列表
    args = vars(parser.parse_known_args()[0])

    config.update(args)

    # 4. construct derived configs
    group_dir = config["result_dir"] + config["exp_group"] + "/"
    model_dir = group_dir + config["exp_name"] + "/"
    if not os.path.isdir(group_dir):
        os.mkdir(group_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    config["group_dir"] = group_dir
    config["exp_dir"] = model_dir
    config["model_dir"] = model_dir

    return config
