#%%
# region import
import os,sys

import numpy as np
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append("../models/")
from model_utils import MyDataset,train_val_test_split

mpl.rcParams['font.family'] = ['Times New Roman','SimSun']
mpl.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
mpl.rcParams['font.size'] = 9  # 按磅数设置的
mpl.rcParams['figure.dpi'] = 300
cm = 1/2.54  # centimeters in inches
mpl.rcParams['figure.figsize'] = (12*cm,8*cm)
mpl.rcParams['savefig.dpi'] = 900
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.grid.axis'] = 'both'
mpl.rcParams['axes.grid.which'] = 'both'
mpl.rcParams['axes.facecolor'] = 'white'

dataset_dir = "../results/experiment/dataset/"  # 结果保存目录
data_dir = "../data/training_data/standard-revised/"  # 数据读取目录

dl_config = {'seed': 0,
             'data_dir': data_dir,
             'batch_size': 32,
             'lookback': 8,
             'lookahead': 4,
             'sample_size': (8,4),
             'num_workers': 2}

# endregion

#%%
ds = MyDataset(dl_config)
train_ds,val_ds,test_ds = train_val_test_split(ds,[0.8,0.1,0.1],0)  # 固定的种子

#%% 延误数据提取与保存
def save_delay_data(config):
    ds_list = {'train':train_ds,'val':val_ds,'test':test_ds}
    for k,v in ds_list.items():
        num_samples = len(v)
        pbar = tqdm(total=num_samples,desc='Loading:')
        y = np.zeros((num_samples,config['lookahead']))
        for i in range(num_samples):
            sample = v[i]
            y[[i]] = sample[1]
            pbar.update(1)
        np.save(dataset_dir+k+'_delay_data.npy',y)

save_delay_data(dl_config)

#%%
def save_grid_data(config):  # 获取训练集的信控高阶参数
    num_samples = len(train_ds)
    pbar = tqdm(total=num_samples,desc='Loading:')
    mg_1 = np.zeros((num_samples,config['lookahead']))
    mg_2 = np.zeros((num_samples,config['lookahead']))
    for i in range(num_samples):
        mg_1[i] = train_ds[i][0]['tc'][0,config['lookback']:,[5,6]].sum(-1)
        mg_2[i] = train_ds[i][0]['tc'][0,config['lookback']:,[7,8]].sum(-1)
        pbar.update(1)
    np.savez(dataset_dir+'grid_data.npz',mg_1=mg_1,mg_2=mg_2)

save_grid_data(dl_config)

#%%
def save_curve_data(config):
    pass
        
#%%
def save_delay_data(config):
    ds_list = {'train':train_ds,'val':val_ds,'test':test_ds}
    for k,v in ds_list.items():
        num_samples = len(v)
        pbar = tqdm(total=num_samples,desc='Loading:')
        y = np.zeros((num_samples,config['lookahead']))
        for i in range(num_samples):
            sample = v[i]
            y[[i]] = sample[1]
            pbar.update(1)
        np.save(dataset_dir+k+'_delay_data.npy',y)

save_delay_data(dl_config)