#%%
# region import
import os,sys,random
import argparse,yaml
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

def parse_config():
    parser = argparse.ArgumentParser("parse cmd arguments")
    parser.add_argument('--exp_group',help="the name of experiment group",type=str,default="test")
    parser.add_argument('--exp_id',help="the experiment id",type=str,default="baseline")
    # parse_known_args返回两个元素，第一个为所求的NameSpace，第二个是unknown args的列表
    args = parser.parse_known_args()[0]  
    return vars(args)
    
def update_config(default_config_dir,updated_config):
    # process argparse & yaml
    with open(default_config_dir,'r+') as f:
        config = yaml.safe_load(f)
    config.update(updated_config)

    # 补充细节路径
    group_dir = config['base_dir'] + config['exp_group'] + "/"
    id_dir = group_dir + config['exp_id'] + "/"

    if not os.path.isdir(group_dir):
        os.mkdir(group_dir)
    if not os.path.isdir(id_dir):
        os.mkdir(id_dir)

    config['group_dir'] = group_dir
    config['id_dir'] = id_dir
    
    return config
