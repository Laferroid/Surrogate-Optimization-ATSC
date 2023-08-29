#%% import
import os,sys,time,argparse
import numpy as np
import joblib
from tqdm import tqdm
from multiprocessing import Pool
import yaml
from functools import partial

import traci
from traci import vehicle
from traci import lane
from traci import lanearea,multientryexit
from traci import junction
from traci import constants

# 导入与重载自定义模块
from traffic_controller import BaseTrafficController
from vehicle_generator import VehicleGenerator
from mpc_controller import MPCController
from utils import get_movement,inlet_map,try_connect

sys.path.append("../../models/")
from uncertainty_surrogate_model import UncertaintySurrogateModel
from model_utils import resume

from sim_modules import Monitor,Recorder,Clock,Observer
from runner import run_experiment,run_sample

#%% 多进程仿真，获取数据
MODE = 'experiment'

if MODE=='sample':
    RUN_NUM = 1
    CYCLE_TO_RUN = 15
    TIME_TO_RUN = 666.0
elif MODE=='experiment':
    CYCLE_TO_RUN = 666
    TIME_TO_RUN = 3600.0

def wrapper(index):
    run_sample(index,CYCLE_TO_RUN,TIME_TO_RUN)

def main():
    with Pool(8) as pool:
        pool.map(partial(run_sample,cycle_to_run=CYCLE_TO_RUN,time_to_run=TIME_TO_RUN),[i for i in range(RUN_NUM)])
        pool.close()
        pool.join()

if __name__ == "__main__":
    if MODE == 'sample':  # 多进程会各复制一份本文件，并且完全执行，需要把其他脚本部分注释掉或进行if判断
        start_time = time.perf_counter()
        # main()
        run_sample(0,CYCLE_NUM,TIME_TO_RUN)
        end_time = time.perf_counter()
        duration = (end_time - start_time)  # 毫秒
        print(f'运行时间:{duration//3600}h {(duration%3600)//60}m {int(duration%60)}s')
        
    elif MODE == 'experiment':
        res = run_experiment(CYCLE_NUM,TIME_TO_RUN)
        # 保存MPC控制器的信息
        with open(save_dir+"mpc_data.pkl",'wb') as f:
            joblib.dump({'surrogate_result': res['mpc_controller'].surrogate_result,
                         'control_result': res['mpc_controller'].control_result,
                         'context_result': res['mpc_controller'].context_result,
                         'horizon_result': res['mpc_controller'].horizon_result,
                         'valid_result': res['mpc_controller'].valid_result,
                         'surrogate_model': res['surrogate_model'],
                         'demand':res['demand']}, f)