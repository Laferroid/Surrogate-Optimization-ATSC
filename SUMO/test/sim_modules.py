# region import
import os,sys,time
import numpy as np
import joblib
from tqdm import tqdm
from multiprocessing import Pool

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
# endregion

STEP_LENGTH = 1.0
TIME_INTERVAL = 1.0
STEP_NUM = int(TIME_INTERVAL/STEP_LENGTH)
WARM_UP = 400  # 热启动需要的时间, 4个默认配时的周期

GRID_LENGTH = 2.0
OBS_RANGE = 200.0
LANE_NUM = 4  # hardcode
GRID_NUM = int(OBS_RANGE//GRID_LENGTH)+3

class Monitor():
    def __init__(self):
        # 维护两个集合，访问每一时间步进出交叉口的车辆，以及当前交叉口内的车辆
        self.prev_veh = set()
        self.cur_veh = set()
        
        self.time = 0  # monitor内部时钟

        # 到达
        self.vph_update_freq = 300 # 流量统计的更新频率
        self.vph_count_m = np.ones((4,3))  # 流向到达车辆计数
        self.vph_m = np.zeros((4,3))  # 流向小时流率指数平滑，用于计算信号配时
        
        self.arrive_m = []  # 实时流向到达

        # 排队
        self.queue_length_l = []  # 实时车道排队
        
        # 订阅排队长度检测器的last_step_vehice_number
        e2_id_list = lanearea.getIDList()
        for e2_id in e2_id_list:
            if e2_id[0] == 'm':
                lanearea.subscribe(e2_id,[constants.JAM_LENGTH_METERS])
        
        # 延误
        # 交叉口内车辆，到达时的损失时间
        self.veh_timeloss  = {}  # key: vehicle_id, value: timeloss
        self.timeloss_m = []  # 实时流向延误
        
        # 离去
        self.depart_m = []  # 实时流向离去

    def run(self):
        self.prev_veh = self.cur_veh
        self.cur_veh = set(multientryexit.getLastStepVehicleIDs("e3"))

        # 到达
        arrive = np.zeros((4,3))
        for veh_id in (self.cur_veh - self.prev_veh):  # 到达交叉口附近的车辆
            self.veh_timeloss[veh_id] = vehicle.getTimeLoss(veh_id)  # 初始化上一步延误
            o,turn,_ = get_movement(veh_id)  # 到达计数
            arrive[o,turn] += 1
            self.vph_count_m[o,turn] += 1
        self.arrive_m.append(arrive)

        # 排队
        e2_results = lanearea.getAllSubscriptionResults()
        queue_length = np.zeros(16)
        for key,result in e2_results.items():
            if key[0] == 'm':
                queue_length[int(key[4:])] = result[constants.JAM_LENGTH_METERS]
        self.queue_length_l.append(queue_length)

        # 延误
        timeloss = np.zeros((4,3))
        for veh_id in self.prev_veh.intersection(self.cur_veh):  # 仍在交叉口附近的车辆
            o,turn,_ = get_movement(veh_id) 
            timeloss[o,turn] += (vehicle.getTimeLoss(veh_id)-self.veh_timeloss[veh_id])
            self.veh_timeloss[veh_id] = vehicle.getTimeLoss(veh_id)
        self.timeloss_m.append(timeloss)

        # 离去
        depart = np.zeros((4,3))
        for veh_id in (self.prev_veh - self.cur_veh):
            o,turn,_ = get_movement(veh_id)
            depart[o,turn] += 1
            self.veh_timeloss.pop(veh_id)
        self.depart_m.append(depart)

        # 更新流量估计，重置到达计数
        alpha = 0.8
        if self.time % self.vph_update_freq == 0:
            self.vph_m *= (1-alpha)
            self.vph_m += alpha*self.vph_count_m/(self.vph_update_freq/3600.0)
            self.vph_count_m = np.zeros((4,3))  # 重置
        
        self.time += 1
    
    def output(self):
        arrive = np.stack(self.arrive_m,axis=0)
        queue_length = np.stack(self.queue_length_l,axis=0)
        timeloss = np.stack(self.timeloss_m,axis=0)
        depart = np.stack(self.depart_m,axis=0)

        return {'arrive':arrive,'queue_length':queue_length,'timeloss':timeloss,'depart':depart}
    
    def reset(self):
        # 到达
        self.arrive_m = []  # 实时流向到达
        # 排队
        self.queue_length_l = []  # 实时车道排队
        # 延误
        self.timeloss_m = []  # 实时流向延误
        # 离去
        self.depart_m = []  # 实时流向离去

class Observer():
    def __init__(self):
        self.id = 'J'   # 路段设备所在交叉口id
        self.max_obs_range = OBS_RANGE   # 路端设备观测范围 ,也即数据收集范围
        
        junction.subscribeContext(self.id,
                                  constants.CMD_GET_VEHICLE_VARIABLE,
                                  self.max_obs_range,
                                  [constants.VAR_SPEED,constants.VAR_ROUTE_ID])
        
    def output(self):
        obs = np.zeros((1+1,4,GRID_NUM,LANE_NUM),dtype=np.float32)
        obs[1] -= 1.0
        vehicle_info = junction.getContextSubscriptionResults(self.id)
        
        for vehicle_id in vehicle_info:
            lane_id = vehicle.getLaneID(vehicle_id)
            if lane_id[0] == ':':  # 排除越过停车线的交叉口中车辆，主要为不受信控的右转车
                continue
            if lane_id[2] == 'O':  # 排除交叉口附近正在离开的车辆
                continue
            inlet_index = inlet_map[lane_id[0]]
            lane_index = int(lane_id[-1])

            # 横向位置
            # lat_pos = vehicle.getLateralLanePosition(vehicle_id)
            
            # 纵向位置, 距交叉口的距离
            if lane_id[-3] == 'P':   # 在不可变道进口段
                lon_pos = (lane.getLength(lane_id) - (vehicle.getLanePosition(vehicle_id) - vehicle.getLength(vehicle_id)))
            else:   # 在可变道进口段
                lon_pos = lane.getLength(lane_id) - (vehicle.getLanePosition(vehicle_id) - vehicle.getLength(vehicle_id))
                lon_pos += lane.getLength(lane_id[:5] + 'P_' + lane_id[-1])

            grid_index = int(lon_pos//GRID_LENGTH)
            
            obs[:,inlet_index,grid_index,lane_index] = [vehicle.getSpeed(vehicle_id),get_movement(vehicle_id)[1]]
            
        return obs

class Recorder():
    def __init__(self,mode):
        self.obs_list = []  # 记录秒级观测数据
        self.obs_c = []  # 秒级观测数据的周期整合

        self.arrive_list = []
        self.queue_length_list = []  # 记录排队长度
        self.timeloss_list = []  # 记录流向实时延误数据
        self.depart_list = []
        
        self.tc_list = []  # 记录信号控制数据
        
        self.time_point = []  # 记录周期结束的时间点
        
        self.mode = mode
    
    def run(self,observer):
        self.obs_c.append(observer.output())  # 获取每秒的观测数据
    
    def update(self,monitor,tc):
        res = monitor.output()
        self.arrive_list.append(res['arrive'])
        self.queue_length_list.append(res['queue_length'])
        self.timeloss_list.append(res['timeloss'])
        self.depart_list.append(res['depart'])
        
        self.tc_list.append(tc.output())   # 上一周期的信号控制
        
        self.obs_c = np.stack(self.obs_c,axis=0)
        self.obs_list.append(np.array(self.obs_c))
        self.obs_c = []
        
        self.time_point.append(monitor.time)  # 记录的时间点
        
    def save(self,index,save_dir):
        # 保存<仿真数据>
        if self.mode == "experiment":
            data = {'tc':self.tc_list,'arrive':self.arrive_list,'queue_length':self.queue_length_list,'timeloss':self.timeloss_list,'depart':self.depart_list}
            with open(save_dir+'simulation_data.pkl', 'wb') as f:
                joblib.dump(data, f)
            with open(save_dir+'timestamp_data.pkl', 'wb') as f:
                joblib.dump(self.time_point, f)
        elif self.mode == "sample":
            data = {'obs':self.obs_list,'tc':self.tc_list,'arrive':self.arrive_list,'queue_length':self.queue_length_list,
                    'timeloss':self.timeloss_list,'depart':self.depart_list}
            with open(save_dir+'simulation_data_'+str(index)+'.pkl', 'wb') as f:
                joblib.dump(data, f)

class Clock():
    def __init__(self,step_num,warm_up,cycle_to_run,time_to_run):
        self.cycle_step = 0  # 已完成的周期数
        self.time = 0  # 仿真经过的时间（秒）
        
        self.step_num = step_num
        self.step_count = step_num
        self.warm_up = warm_up
        
        self.cycle_to_run = cycle_to_run
        self.time_to_run = time_to_run
    
    def step(self):
        self.step_count -= 1
        
    def run(self):
        self.step_count = self.step_num
        self.time += 1
    
    def update(self):
        self.cycle_step += 1

    def is_to_run(self):
        return self.step_count == 0
    
    def is_warm(self):
        return self.time >= self.warm_up
    
    def is_end(self,mode):
        if mode=='sample':
            return self.cycle_step >= self.cycle_to_run
        elif mode=='experiment':
            return self.time >= self.time_to_run
    