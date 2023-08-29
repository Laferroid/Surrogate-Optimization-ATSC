#%% import 
from traci import vehicle as vehicle
import numpy as np
import pandas as pd

from utils import get_movement

route_list = ['EN_S','EN_W','EN_N',
            'EE_S','EE_W','EE_N',
            'ES_S','ES_W','ES_N',
            'SE_W','SE_N','SE_E',
            'SS_W','SS_N','SS_E',
            'SW_W','SW_N','SW_E',
            'WS_N','WS_E','WS_S',
            'WW_N','WW_E','WW_S',
            'WN_N','WN_E','WN_S',
            'NW_E','NW_S','NW_W',
            'NN_E','NN_S','NN_W',
            'NE_E','NE_S','NE_W']

route_info = pd.DataFrame({'route':route_list,
                           'inlet':[get_movement(route)[0] for route in route_list],
                           'turn':[get_movement(route)[1] for route in route_list]})

#%% 车辆生成器
class VehicleGenerator():
    def __init__(self,mode,route_info=route_info,duration=60*30):
        self.route_info = route_info
        self.route_num = self.route_info.shape[0]
        self.duration = duration  # 持续时间/秒
        self.time = 0  # 仿真进度
        self.schedule = {}
        
        self.vph_m_list = []  # 流向流量信息
        self.exp_headway = False
        
        if mode=='static':
            self.generate_demand = self.generate_static_demand
        elif mode=='stepfunc':
            self.generate_demand = self.generate_stepfunc_demand
        elif mode=='linear':
            self.generate_demand = self.generate_linear_demand
        elif mode=='sine':
            self.generate_demand = self.generate_sine_demand
        
        self.rng = np.random.RandomState(0)
        
        self.v_level = 400
        
        # 初始化
        self.update()
        self.run()
        
    def run(self):            
        self.generate_vehicle()
        self.time +=1
        if self.time%self.duration == 0:
            self.rng = np.random.RandomState(int(self.time/self.duration)+1)
            self.update()
        
    def update(self):
        self.vph_m = np.zeros((self.duration,4,3))
        self.generate_demand()
        self.vph_m_list.append(self.vph_m)
        
    def generate_vehicle(self):
        for r in route_info['route'].values:
            turn = get_movement(r)[1]
            veh_type = 'LEFT' if turn==0 else 'THROUGH' if turn==1 else 'RIGHT'
            for i in range(self.schedule[r][self.time%self.duration]):
                vehicle.add(vehID=r+'.'+str(self.time)+'.'+str(i),
                            routeID=r,typeID=veh_type,departLane='best')
    
    def generate_static_demand(self):
        # 恒定vph
        # # 进口道基础vph
        self.v_level += 200
        vph_level = np.array(4*[800])
        # 进口道转向比
        turn_ratio = np.array([[0.25,0.5,0.25],
                               [0.25,0.5,0.25],
                               [0.25,0.5,0.25],
                               [0.25,0.5,0.25]])
        
        for r in self.route_info['route'].values:
            inlet,turn,_ = get_movement(r)
            self.schedule[r] = []
            second = 0.0
            self.vph_m[0,inlet,turn] = vph_level[inlet]*turn_ratio[inlet,turn]
            
            headway = 3600/(1/3*self.vph_m[0,inlet,turn])
            if self.exp_headway:
                headway = self.rng.exponential(headway)
            
            for t in range(self.duration):
                second += 1.0
                n_veh = 0
                self.vph_m[t,inlet,turn] = vph_level[inlet]*turn_ratio[inlet,turn]
                while (second-headway)>0:
                    second -= headway
                    n_veh += 1
                    headway = 3600/(1/3*self.vph_m[t,inlet,turn])
                    if self.exp_headway:
                        headway = self.rng.exponential(headway)
                self.schedule[r].append(n_veh)
    
    def generate_linear_demand(self):
        # 动态变化的vph: linear
        # 进口道基础vph
        scale = 1.3
        if len(self.vph_m_list) > 0:
            prev_vph = self.vph_m_list[-1][-1]
            vph_level_a = prev_vph.sum(-1)
            turn_ratio_a = prev_vph/prev_vph.sum(-1)[:,None]
        elif len(self.vph_m_list) == 0:
            vph_level_a = self.rng.uniform(200.0,1200.0,4)
            vph_level_a *= scale
            turn_ratio_a = self.rng.uniform(0.8,1.2,(4,3))*np.array(4*[[1,2,1]])
            turn_ratio_a = turn_ratio_a/turn_ratio_a.sum(axis=-1)[:,None]

        vph_level_b = self.rng.uniform(200.0,1200.0,4)
        vph_level_b *= scale
        vph_level = np.linspace(vph_level_a,vph_level_b,self.duration,axis=0)
        
        # 进口道转向比
        turn_ratio_b = self.rng.uniform(0.8,1.2,(4,3))*np.array(4*[[1,2,1]])
        turn_ratio_b = turn_ratio_b/turn_ratio_b.sum(axis=-1)[:,None]
        turn_ratio = np.linspace(turn_ratio_a,turn_ratio_b,self.duration,axis=0)
        
        for r in self.route_info['route'].values:
            inlet,turn,_ = get_movement(r)
            self.schedule[r] = []
            second = 0.0
            self.vph_m[0,inlet,turn] = vph_level[0,inlet]*turn_ratio[0,inlet,turn]
            
            headway = 3600/(1/3*self.vph_m[0,inlet,turn])
            if self.exp_headway:
                headway = self.rng.exponential(headway)
            
            for t in range(self.duration):
                second += 1.0
                n_veh = 0
                self.vph_m[t,inlet,turn] = vph_level[t,inlet]*turn_ratio[t,inlet,turn]
                while (second-headway)>0:
                    second -= headway
                    n_veh += 1
                    headway = 3600/(1/3*self.vph_m[t,inlet,turn])
                    if self.exp_headway:
                        headway = self.rng.exponential(headway)
                self.schedule[r].append(n_veh)
        
    def generate_sine_demand(self):
        # 动态变化的vph: linear
        x = np.linspace(0,2*np.pi,self.duration)
        # 进口道基础vph
        A = np.expand_dims(self.rng.uniform(100,400,4),axis=0)
        omega = np.expand_dims(0.5*self.rng.choice([1,2,3,4],4),axis=0)
        b = np.expand_dims(self.rng.uniform(400,1800,4),axis=0)
        vph_level = A*np.sin(omega*np.expand_dims(x,axis=-1)) + b  # (duration,4)
        vph_level *= 0.75
        
        # 进口道转向比
        A = np.expand_dims(self.rng.uniform(0.05,0.25,(4,3)),axis=0)
        omega = np.expand_dims(0.5*self.rng.choice([1,2,3,4],(4,3)),axis=0)
        b = np.expand_dims(np.array(4*[[1,2,1]]),axis=0)
        turn_ratio = A*np.sin(omega*np.expand_dims(x,axis=(-1,-2))) + b  # (duration,4,3)
        turn_ratio /= turn_ratio.sum(-1,keepdims=True)
        
        for r in self.route_info['route'].values:
            inlet,turn,_ = get_movement(r)
            self.schedule[r] = []
            second = 0.0
            self.vph_m[0,inlet,turn] = vph_level[0,inlet]*turn_ratio[0,inlet,turn]
            
            headway = 3600/(1/3*self.vph_m[0,inlet,turn])
            if self.exp_headway:
                headway = self.rng.exponential(headway)
            
            for t in range(self.duration):
                second += 1.0
                n_veh = 0
                self.vph_m[t,inlet,turn] = vph_level[t,inlet]*turn_ratio[t,inlet,turn]
                while (second-headway)>0:
                    second -= headway
                    n_veh += 1
                    headway = 3600/(1/3*self.vph_m[t,inlet,turn])
                    if self.exp_headway:
                        headway = self.rng.exponential(headway)
                self.schedule[r].append(n_veh)
    
    def output(self):
        return np.concatenate(self.vph_m_list,axis=0)[640:]  # (time,inlet,turn), 取warm up后的数据