# region import
import os,sys,time
import numpy as np
import joblib
from tqdm import tqdm
from multiprocessing import Pool

import traci

# 导入与重载自定义模块
from traffic_controller import BaseTrafficController
from vehicle_generator import VehicleGenerator
from utils import try_connect
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

sys.path.append("../../SUMO/test/")

# endregion

#%%
def run_snapshot(control,snapshot_dir,seed=0):
    # control: array(lookahead,13), plannig control shcmes for several following cycles
    # cycle_to_run equals to lookahead
    sumocfg = ["sumo",
            "--route-files","test.rou.xml",
            "--net-file","test.net.xml",
            "--additional-files","test.e2.xml,test.e3.xml",
            "--gui-settings-file","gui.cfg",
            "--delay","0",
            "--time-to-teleport","600",
            "--step-length",f"{STEP_LENGTH}",
            "--no-step-log","true",
            "--quit-on-end",
            "-X","never",
            "-W","true",
            "--duration-log.disable"]
    sumocfg = sumocfg + ["--seed",str(seed)]
    try_connect(8,sumocfg)
    mode = 'experiment'
    
    time_to_run = 666.0
    cycle_to_run = control.shape[0]
    
    with open(snapshot_dir+"monitor.pkl",'rb') as f:
        monitor = joblib.load(f)
    with open(snapshot_dir+"veh_gen.pkl",'rb') as f:
        veh_gen = joblib.load(f)
    
    clock = Clock(STEP_NUM,WARM_UP,cycle_to_run,time_to_run)
    recorder = Recorder(mode)
    
    tc = None
    
    observer = Observer()
    traci.simulation.loadState(snapshot_dir+'state.xml')
    
    while(clock.cycle_step <= 3):
        # 仿真步进
        if not tc:
            tc = BaseTrafficController(STEP_LENGTH,TIME_INTERVAL)
            tc.update('fixed-time',param=(control[0,:5],control[0,5:]))
            tc.run()
        traci.simulationStep()
        clock.step()
        
        # 处理：单位时间(秒)结束时
        if clock.is_to_run():
            monitor.run()  # 开始监测
            recorder.run(observer)  # 开始记录
            clock.run()
            veh_gen.run()  # 车辆生成器根据schedule生成车辆
            tc.run()

            # 处理：周期即将切换时
            if tc.is_to_update():
                recorder.update(monitor,tc)
                clock.update()
                if clock.cycle_step<=3:
                    tc.update('fixed-time',param=(control[clock.cycle_step,:5],control[clock.cycle_step,5:]))

                monitor.reset()  # 重置性能指标监测器
    # 结束仿真
    traci.close()
    # demand = veh_gen.output()
    
    # 把延误存下来输出即可
    return recorder.timeloss_list