#%%
# region import
import sys,os,datetime
import torch
import numpy as np
import yaml,argparse

sys.path.append("../models/")
from uncertainty_surrogate_model import UncertaintySurrogateModel
from model_utils import train,MyDataset,get_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# endregion

#%%
config_dir = "../configs/default_config.yaml"

parser = argparse.ArgumentParser("Surrogate model training")
parser.add_argument('--exp_name',help="experiment name",type=str,default="gamma-1.0")
parser.add_argument('--data_dir',help="data directory",type=str,default="../data/training_data/standard-1.0/")
args = parser.parse_known_args()[0]  # parse_known_args返回两个元素，第一个为所求的NameSpace，第二个是unknown args的列表

# process argparse & yaml
config = vars(args)
with open(config_dir,'r+') as f:
    # args = yaml.safe_load(f,Loader=yaml.FullLoader)
    args = yaml.safe_load(f)
config.update(args)

exp_dir = config['base_dir'] + config['exp_name'] + "/"
# model_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = 'baseline'
model_dir = exp_dir + model_name + "/"

if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

config['model_name'] = model_name
config['model_dir'] = model_dir
config['device'] = device

#%% 训练模型
if __name__ == "__main__":
    # Windows上多进程的实现问题。在Windows上，子进程会自动import启动它的文件，而在import的时候会执行这些语句，就会无限递归创建子进程报错。
    # 设置随机数种子
    # torch.backends.cudnn.deterministic = True
    # random.seed(config['seed'])
    np.random.seed(config['seed'])
    # torch.manual_seed(config['seed'])
    # torch.cuda.manual_seed_all(config['seed'])
    
    dataset = MyDataset(config)
    print(len(dataset))
    train_dl,val_dl,test_dl = get_dataloader(dataset,config)
    
    model = UncertaintySurrogateModel(config)
    train(model,8,train_dl,val_dl,config)
    
    print("finished !")
    