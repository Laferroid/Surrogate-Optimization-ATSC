#%%
# region import 
import sys,os
import numpy as np
import torch
from tqdm import tqdm
import joblib
from joblib import Parallel,delayed
from sklearn.preprocessing import OneHotEncoder

sim_config = {'grid_length':2.0,
              'obs_range':200.0,
              'lane_num':4,
              'lookback':4,
              'lookahead':2,
              'gamma':0.9}

sim_config['grid_num'] = int(sim_config['obs_range']//sim_config['grid_length'])+3

GRID_LENGTH = sim_config['grid_length']
OBS_RANGE = sim_config['obs_range']
LANE_NUM = sim_config['lane_num']
LOOKBACK = sim_config['lookback']
LOOKAHEAD = sim_config['lookahead']
GRID_NUM = sim_config['grid_num']
# endregion

#%% 仿真数据处理成代理模型的输入
def agg_sim_data(data_dir,output_dir):
    agg_freq = 16  # 至多多少个仿真的文件同时进行处理
    data = {'obs':[],'timeloss':[],'tc':[]}
    pbar = tqdm(total=len(os.listdir(data_dir)),desc=f"loading ")
    chunk_index = 0
    sample2chunk = []
    for _,file in enumerate(os.listdir(data_dir)):
        with open(data_dir+file,'rb') as f:
            try:
                data_p = joblib.load(f)
            except EOFError:
                pbar.update(1)
                print("EOFError:",file)
                continue
        pbar.update(1)
        # 舍弃周期数过少的仿真序列
        if len(data_p['obs']) < LOOKBACK+LOOKAHEAD:
            print("Simulation too short:",file)
            continue
        data['obs'].append(data_p['obs'])
        data['timeloss'].append(data_p['timeloss'])
        data['tc'].append(data_p['tc'])
        
        if len(data['obs']) == agg_freq:   # 流式处理
            sample_num = pre_sim_data(chunk_index,data,output_dir)
            temp = [sample_num*[chunk_index],list(range(sample_num))]
            temp = list(zip(*temp))
            sample2chunk.extend(temp)
            chunk_index += 1
            data = {'obs':[],'timeloss':[],'tc':[]}
    # 末尾数据
    if len(data['obs']) != 0:
        sample_num = pre_sim_data(chunk_index,data,output_dir)
        temp = [sample_num*[chunk_index],list(range(sample_num))]
        temp = list(zip(*temp))
        sample2chunk.extend(temp)

    with open(output_dir+'sample2chunk.pkl','wb') as f:
        joblib.dump(sample2chunk, f)

def pre_sim_data(chunk_index,data,output_dir):
    obs = obs_process(data['obs'])
    timeloss = timeloss_process(data['timeloss'])

    tc = tc_process(data['tc'])
    
    # 时窗数据样本数
    sample_num = len(obs)
    # 每个样本存为一个文件
    for i in range(sample_num):
        # obs和其他数据分开保存
        torch.save({'timeloss':timeloss[i],'tc':tc[i]},
                    output_dir+'other_'+str(chunk_index)+'_'+str(i)+'.pth')
        np.savez_compressed(output_dir+'obs_'+str(chunk_index)+'_'+str(i)+'.npz',obs[i])
    
    return sample_num

def obs_process(obs_data):
    # input: (sample:list,cycle:list,frames:array)
    # output: (sample:2d-array,cycle:2d-array,frames:tensor)
    sample_num = len(obs_data)

    # 多进程，进程中的变量会丢失
    # 所以要么放到硬盘(写入文件),要么返回值(结果列表)
    if sample_num>1:
        obs = Parallel(n_jobs=1)(delayed(par_obs_process)(i,sample) for (i,sample) in enumerate(obs_data))
    else:
        obs = [par_obs_process(0,obs_data[0])]
    
    # lookback
    o = []
    for i in range(sample_num):
        obs_sample = obs[i]
        cycle_num = len(obs_sample)
        for j in range(LOOKBACK,cycle_num-LOOKAHEAD+1):
            o.append(obs_sample[j-LOOKBACK:j])
    # object array: unequal size in some dims
    o_temp = o
    o = np.empty((len(o_temp),LOOKBACK),dtype=object)
    # o: (sample,lookback,frames!,*)
    for i,sample in enumerate(o_temp):
        for j,cycle in enumerate(sample):
            o[i,j] = cycle
            
    return o

def par_obs_process(i,sample):
    # input: (cycle:list,frames:array)
    # output: (cycle:list,frames:array)
    # sample = sample[:-1]  # 去除末尾空位
    obs_i = []
    one_hot = OneHotEncoder(categories=(4*GRID_NUM*LANE_NUM)*[[-1.,0.,1.,2.]],
                            sparse_output=False)
    for _,cycle in enumerate(sample):
        obs_c = frame_process(one_hot,cycle)
        # 观测数据tensor的二维列表
        obs_i.append(obs_c)
    
    return obs_i

def frame_process(one_hot,cycle):
    # input: frame list
    # output: tensor
    # 车端数据的网格划分与one-hot编码
    if one_hot is None:
        one_hot = OneHotEncoder(categories=(4*GRID_NUM*LANE_NUM)*[[-1.,0.,1.,2.]],
                                sparse_output=False)
    frame_num = len(cycle)  # 周期的帧数量
    obs_speed = cycle[:,0]
    obs_move = cycle[:,1]
    
    obs_move = one_hot.fit_transform(obs_move.reshape(frame_num,-1))
    obs_move = obs_move.reshape(frame_num,4,GRID_NUM,LANE_NUM,4)
    obs_speed = obs_speed.reshape(frame_num,4,GRID_NUM,LANE_NUM,1)
    
    obs_c = np.concatenate([obs_move,obs_speed],axis=-1)
    obs_c = np.moveaxis(obs_c,-1,1).reshape(frame_num,20,GRID_NUM,LANE_NUM)  
    # 先type再inlet
    obs_c = torch.from_numpy(obs_c).to(torch.float32)

    return obs_c  # Tensor: (frame_num,20,GRID_NUM,LANE_NUM)

def timeloss_process(timeloss_data):
    # intput timeloss_data: (sample:list,cycle:list,frames:)
    # output d: tensor(sample,lookahead)
    d = []
    sample_num = len(timeloss_data)
    for i in range(sample_num):
        timeloss_sample = timeloss_data[i]
        cycle_num = len(timeloss_sample)
        timeloss_sample_new = np.zeros(cycle_num)
        
        for j in range(cycle_num):
            timeloss_cycle = timeloss_sample[j]
            cycle_length = len(timeloss_cycle)
            w = np.logspace(start=0,stop=cycle_length-1,num=cycle_length,base=sim_config['gamma'])
            timeloss_sample_new[j] = ((timeloss_cycle.sum(axis=(1,2))*w).sum())
        # lookahead
        for j in range(LOOKBACK,cycle_num-LOOKAHEAD+1):
            d.append(timeloss_sample_new[j:j+LOOKAHEAD])
    
    d = np.array(d).reshape(-1,LOOKAHEAD)
    d = torch.from_numpy(d).to(torch.float32)
    
    return d

def tc_process(tc_data):
    # input: (sample:list,cycle:list,control:dict)
    # output: tensor(sample,lookback+lookahead,13)
    tc_p = []  # phase
    tc_s = []  # split

    sample_num = len(tc_data)
    for i in range(sample_num):
        sample = tc_data[i]
        cycle_num = len(sample)
        # LOOKBACK+LOOKAHEAD
        # tc: (sample_num, cycle_num, dict:phase and split, array)
        for j in range(LOOKBACK,cycle_num-LOOKAHEAD+1):
            tc_p.append([sample[k]['phase'] for k in range(j-LOOKBACK,j+LOOKAHEAD)])
            tc_s.append([sample[k]['split'] for k in range(j-LOOKBACK,j+LOOKAHEAD)])
    
    tc_p = np.array(tc_p).reshape(-1,LOOKBACK+LOOKAHEAD,5)
    tc_s = np.array(tc_s).reshape(-1,LOOKBACK+LOOKAHEAD,8)
    
    tc = np.concatenate([tc_p,tc_s],axis=-1)
    tc = torch.from_numpy(tc).to(torch.float32)
    
    return tc

#%%
if __name__ == "__main__":
    data_dir = '../data/simulation_data/standard/'
    output_dir = '../data/training_data/standard/'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    agg_sim_data(data_dir,output_dir)