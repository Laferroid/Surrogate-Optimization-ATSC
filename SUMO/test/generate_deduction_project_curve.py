#%%
import os,sys,joblib
from joblib import Parallel,delayed
from tqdm import tqdm
import numpy as np
from runner_snapshot import run_snapshot

def discounted(timeloss):
    # 输入一个周期内各个流向各秒的timeloss, (time,inlet,movement)
    num = timeloss.shape[0]
    weight = np.logspace(start=0,stop=num,num=num,base=0.999)
    return (timeloss.sum((-1,-2))*weight).sum()

proposed_name = 'standard-seed-1'
control_dir = "../../results/experiment/control/"
result_dir = control_dir + proposed_name +'/results/'
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

num = 20  # 绘图密度
cycle_index = 60
output_dir = result_dir + f'snapshot_{cycle_index}/'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
step_ahead = 1  # 进行调整的前向位置

controls = np.load(output_dir+f'control_project_curve.npy')

snapshot_dir = control_dir + proposed_name +'/snapshots/snapshot_'+str(cycle_index)+'/'

def par_func(control):
    num = 8
    d = 0
    for i in range(num):
        delays = run_snapshot(control,snapshot_dir,i)  # list of array(time,inlet,movement)
        d += discounted(delays[step_ahead])
    return d/num

res = Parallel(n_jobs=8,verbose=10)(delayed(par_func)(controls[i]) for i in range(num))

with open(output_dir+"res_project_curve.npy",'wb') as f:
    np.save(f,np.array(res))