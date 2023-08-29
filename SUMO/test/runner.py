#%%
# region import
import os,sys,time,argparse
import numpy as np
import joblib
from tqdm import tqdm
from multiprocessing import Pool
import yaml

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
# endregion

# region configuration
STEP_LENGTH = 1.0
TIME_INTERVAL = 1.0
STEP_NUM = int(TIME_INTERVAL/STEP_LENGTH)
WARM_UP = 400  # 热启动需要的时间, 4个默认配时的周期

GRID_LENGTH = 2.0
OBS_RANGE = 200.0
LANE_NUM = 4  # hardcode
GRID_NUM = int(OBS_RANGE//GRID_LENGTH)+3

sumocfg = ["sumo",
           "--route-files","test.rou.xml",
           "--net-file","test.net.xml",
           "--additional-files","test.e2.xml,test.e3.xml",
           "--gui-settings-file","gui.cfg",
           "--delay","0",
           "--time-to-teleport","600",
           "--step-length",f"{STEP_LENGTH}",
           "--no-step-log","true",
           "-X","never",
           "--quit-on-end",
           "--save-state.rng"]

MODE = 'experiment'

if MODE=='experiment':
    sumocfg += ["--seed","0"]  # 固定SUMO的随机数种子，如期望速度随机性和实际速度随机性

# 仿真采样数据保存路径
if MODE == 'sample':
    save_dir = '../../data/simulation_data/standard/'
elif MODE == 'experiment':
    exp_name = 'test-mpc'
    save_dir = '../../results/experiment/control/' + exp_name + '/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

model_dir = '../../results/gamma-1.0/baseline/'

model_config = {'model_dir': model_dir,
                'batch_size': 1,
                'lookback': 8,
                'lookahead': 4,
                'device': 'cpu',
                'train_predict_mode': 'direct',
                'sample_size': (8,4),
                'seed': 0,
                'num_components': 16}

config_dir = "../configs/default_config.yaml"

parser = argparse.ArgumentParser("Surrogate model training")
parser.add_argument('--exp_name',help="experiment name",type=str,default="gamma-1.0")
parser.add_argument('--data_dir',help="data directory",type=str,default="../../data/training_data/standard-1.0/")
args = parser.parse_known_args()[0]  # parse_known_args返回两个元素，第一个为所求的NameSpace，第二个是unknown args的列表

# process argparse & yaml
config = vars(args)
with open(config_dir,'r+') as f:
    # args = yaml.safe_load(f,Loader=yaml.FullLoader)
    args = yaml.safe_load(f)
config.update(args)

exp_dir = config['base_dir'] + config['exp_name'] + "/"
model_name = 'baseline'
model_dir = exp_dir + model_name + "/"

if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

config['model_name'] = model_name
config['model_dir'] = model_dir
config['device'] = device

mpc_config = {'lookback': 8,
              'lookahead': 4,
              'num_enumerations': 1,
              'num_restarts': 8,
              'gamma': 1.000,
              'alpha': 1.0,
              'beta': 1.0}
# endregion

#%%
def run_experiment(cycle_to_run,time_to_run,config):
    mode = 'experiment'
    np.random.seed(0)
    try_connect(8,sumocfg)

    clock = Clock(config['step_num'],config['warm_up'],cycle_to_run,time_to_run)
    monitor = Monitor()
    observer = Observer()
    tc = BaseTrafficController(config['step_length'],config['time_interval'])
    veh_gen = VehicleGenerator(mode='linear')  # 流量生成模式
    recorder = Recorder(mode)
    
    surrogate_model = UncertaintySurrogateModel(model_config)
    resume(surrogate_model)
    mpc_controller = MPCController(surrogate_model,mpc_config)
    
    pbar = tqdm(total=time_to_run,desc=f"simulation experiment")
    
    if not os.path.isdir(save_dir+'snapshots/'):
        os.mkdir(save_dir+'snapshots/')
    
    # 热启动完毕且当前周期结束时正式开始记录数据
    # 按周期数进行仿真
    while(not clock.is_end(mode)):
        # 仿真步进
        traci.simulationStep()
        clock.step()
        
        # 处理：单位时间(秒)结束时
        if clock.is_to_run():
            if clock.is_warm():  # 热启动
                monitor.run()  # 开始监测
                recorder.run(observer)  # 开始记录
            pbar.update(1)
            clock.run()
            veh_gen.run()  # 车辆生成器根据schedule生成车辆
            tc.run()

            # 处理：周期即将切换时
            if tc.is_to_update():
                if clock.is_warm():  # 开始控制
                    recorder.update(monitor,tc)
                    mpc_controller.update(recorder,tc)
                monitor.reset()  # 重置性能指标监测器
                clock.update()
                
                if clock.cycle_step in [24,32,48,60,72]:  # snapshot when cycle is terminated
                    snapshot_dir = save_dir+'snapshots/snapshot_'+str(clock.cycle_step)+'/'
                    if not os.path.isdir(snapshot_dir):
                        os.mkdir(snapshot_dir)
                    traci.simulation.saveState(snapshot_dir+'state.xml')  # 保存仿真中车辆的状态
                    with open(snapshot_dir+"veh_gen.pkl",'wb') as f:  # 保存车辆生成器的状态
                        joblib.dump(veh_gen,f)
                    with open(snapshot_dir+"monitor.pkl",'wb') as f:  # 保存monitor的存储信息
                        joblib.dump(monitor,f)
                
                # region generate traffic control scheme for next cycle
                method = 'mpc'  # mpc or baseline
                assert method in ['mpc','baseline']
                if method == 'mpc':
                    # MPC实验
                    if mpc_controller.warm_up == 0:
                        tc.update('mpc',monitor=monitor,mpc_controller=mpc_controller)
                    else:
                        tc.update('default')  # 使用默认方案
                elif method == 'baseline':
                    # 对比方案实验
                    if clock.time >= (monitor.vph_update_freq + clock.warm_up):
                        if mpc_controller.warm_up == 0:  # 即使不使用mpc也要记录context，方便绘制代理曲线(曲面)
                            mpc_controller.record_context()
                        tc.update('adaptive-nema',monitor=monitor)
                        # tc.update('fixed-time',param=(np.array([0,0,0,0,0]),20.0))
                    else:
                        tc.update('default')  # 使用默认方案
                # endregion
                    
    # 结束仿真
    traci.close()
    recorder.save(None,save_dir)
    demand = veh_gen.output()
    
    return {'mpc_controller':mpc_controller,'surrogate_model':surrogate_model,'demand':demand}

def run_sample(index,cycle_to_run,time_to_run,config):
    mode = 'sample'
    np.random.seed()  # 多进程跑数据，随机设置种子
    
    try_connect(8,sumocfg)

    clock = Clock(config['step_num'],config['warm_up'],cycle_to_run,time_to_run)
    monitor = Monitor()
    observer = Observer()
    tc = BaseTrafficController(config['step_length'],config['time_interval'])
    veh_gen = VehicleGenerator(mode='linear')
    recorder = Recorder(mode)
    
    pbar = tqdm(total=cycle_to_run,desc=f"simulation {index}")
    
    # 热启动完毕且当前周期结束时正式开始记录数据
    # 按周期数进行仿真
    while(not clock.is_end(mode)):
        # 仿真步进
        traci.simulationStep()  # 随机性
        clock.step()
        
        # 处理：单位时间(秒)结束时
        if clock.is_to_run():
            if clock.is_warm():  # 热启动
                monitor.run()  # 开始监测
            if clock.is_warm() and tc.is_start:  # 开始控制
                recorder.run(observer)  # 开始记录
            clock.run()
            veh_gen.run()  # 随机性
            tc.run()  # 随机性

            # 处理：周期即将切换时
            if tc.is_to_update():
                if clock.is_warm() and tc.is_start:  # 开始控制
                    recorder.update(monitor,tc)  # 内部调用monitor的output函数
                    clock.update()
                    pbar.update(1)
                
                # 控制方案更新
                if clock.time >= (monitor.vph_update_freq + clock.warm_up):
                    tc.update('sample',monitor)  # 使用webster配时等饱和度进行采样
                    tc.is_start = True
                else:
                    tc.update('default')  # 使用随机方案
                
                # 出现长度大于200m的排队，严重拥堵，终止仿真
                if clock.cycle_step>=1 and (monitor.queue_length_l[-1]>200.0).any():
                    break
                else:
                    monitor.reset()  # 重置性能指标监测器
    # 结束仿真
    traci.close()
    recorder.save(index,save_dir)
