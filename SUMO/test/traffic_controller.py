#%% import 
import numpy as np
import pandas as pd
from itertools import cycle

from traci import trafficlight

#%% 交通控制器
class BaseTrafficController():
    def __init__(self,step_length,time_interval):
        # NEMA相位，周期级
        self.id = 'J'
        
        self.step_length = step_length  # 仿真步长，单位：s
        self.time_interval = time_interval  # 时间间隔，单位：s
        
        # 黄灯时间与全红时间
        self.y = 3.0
        self.r = 2.0
        self.y_time = int(self.y)
        self.r_time = int(self.r)
        
        self.g_max = 60.0  # 最大绿灯时长
        self.g_min = 15.0  # 最小绿灯时长
        self.G_max = 180.0  # 最大总绿灯时长
        self.G_min = 60.0  # 最小总绿灯时长
        
        self.sfr = 1880

        self.state_num = 16
        # 流向编号映射到信号灯state，编号按照双环相位示意图
        # 0:R, 1:EL, 2:WT, 3:SL, 4:NT, 5:WL, 6:ET, 7:NL, 8:ST
        self.movement2state = [[0,4,8,12],[7],[13,14],[11],[1,2],[15],[5,6],[3],[9,10]]
        
        # 信号灯基础状态
        # 右转车流不受灯控
        self.basic_state = ['G' if i in self.movement2state[0] else 'r'for i in range(self.state_num)]
        trafficlight.setRedYellowGreenState(self.id,''.join(self.basic_state))
        
        # 当前绿灯相位
        self.green_phase_1 = None
        self.green_phase_2 = None
        # 当前相位
        self.phase_1 = None
        self.phase_2 = None
        # 当前相位时长
        self.phase_time_1 = 0  # ring1相位剩余的时间步
        self.phase_time_2 = 0  # ring2相位剩余的时间步
        # 周期时长
        self.cycle_time = 0
        # 相位循环
        self.phase_cycle_1 = None
        self.phase_cycle_2 = None
        # 各流向绿灯时间
        self.split = None
        
        # <优化格式>的控制方案
        self.control = None
        # 使用<随机>策略作为过渡
        self.update(mode='default')
        self.run()
        
    def run(self):
        # 计划下一秒的信号
        # 周期切换
        if self.cycle_time == 0:
            phase_cycle_1,phase_cycle_2,split = self.next_cycle()
            self.phase_cycle_1 = phase_cycle_1
            self.phase_cycle_2 = phase_cycle_2
            self.split = split
            self.cycle_time = int(np.array(split[:4]).sum())+4*(self.y_time+self.r_time)
        
        # ring 1 相位切换
        if self.phase_time_1 == 0:
            # 相位时长
            self.phase_1 = next(self.phase_cycle_1)
            if self.phase_1 == 'y':
                self.phase_time_1 = self.y_time
            elif self.phase_1 == 'r':
                self.phase_time_1 = self.r_time
            else:
                self.green_phase_1 = self.phase_1
                self.phase_time_1 = self.split[self.green_phase_1-1]
                
            # 相位设置
            for state in self.movement2state[self.green_phase_1]:
                if self.phase_1 == 'y':
                    trafficlight.setLinkState(self.id,state,'y')
                elif self.phase_1 == 'r':
                    trafficlight.setLinkState(self.id,state,'r')
                else:
                    trafficlight.setLinkState(self.id,state,'G')

        # ring 2 相位切换        
        if self.phase_time_2 == 0:
            # 相位时长
            self.phase_2 = next(self.phase_cycle_2)
            if self.phase_2 == 'y':
                self.phase_time_2 = self.y_time
            elif self.phase_2 == 'r':
                self.phase_time_2 = self.r_time
            else:
                self.green_phase_2 = self.phase_2
                self.phase_time_2 = self.split[self.green_phase_2-1]
            # 相位设置
            for state in self.movement2state[self.green_phase_2]:
                if self.phase_2 == 'y':
                    trafficlight.setLinkState(self.id,state,'y')
                elif self.phase_2 == 'r':
                    trafficlight.setLinkState(self.id,state,'r')
                else:
                    trafficlight.setLinkState(self.id,state,'G')
        
        self.phase_time_1 -= 1
        self.phase_time_2 -= 1
        self.cycle_time -= 1

    def phase_random(self,control):
        control['phase'] = np.random.choice(2,5)
        # control['phase'] = np.random.choice(1,5)
        return control
    
    def phase_fixed(self,control,order=np.zeros(5)):
        control['phase'] = order  # 固定对称放行相位
        return control
    
    def split_fixed(self,control,split=20.0): 
        if isinstance(split,np.ndarray):
            control['split'] = split
        elif isinstance(split,float) or isinstance(split,int):
            control['split'] = split*np.ones(8)
        return control
    
    def split_nema(self,control,vph_m):
        # 双环的webster配时，参考(Ma, 2022)
        # control需要已有phase的信息
        control['split'] = np.zeros(8)
        movement2number = [[7,4],[1,6],[3,8],[5,2]]  # 流向映射到编号
        for i in range(4):  # 流向的流量比
            control['split'][movement2number[i][0]-1] = vph_m[i,0]/(self.sfr*0.9)  # 左转一条车道
            control['split'][movement2number[i][1]-1] = vph_m[i,1]/(2*self.sfr)  # 直行两条车道
        
        y1 = control['split'][0] + control['split'][1]
        y2 = control['split'][2] + control['split'][3]
        y3 = control['split'][4] + control['split'][5]
        y4 = control['split'][6] + control['split'][7]
        
        if y1>=y3:
            control['split'][[4,5]] *= (y1/y3)
        elif y1<y3:
            control['split'][[0,1]] *= (y3/y1)
        
        if y2>=y4:
            control['split'][[6,7]] *= (y2/y4)
        elif y2<y4:
            control['split'][[2,3]] *= (y4/y2)
        
        Y = max(y1,y3) + max(y2,y4)
        L = 4.0*(self.r + 3.0)
        
        if Y > 0.9:
            Y = 0.9
        C = (1.5*L+5.0)/(1-Y)
        G = C - 4.0*(3.0+self.r)
        
        control['split'] /= control['split'].sum()*0.5
        lb = max(self.g_min/control['split'].min(),self.G_min)
        ub = min(self.g_max/control['split'].max(),self.G_max)
        G = lb if G<lb else ub if G>ub else G
        control['split'] = control['split']*G
        
        return control 
    
    def split_random_nema(self,control,vph_m):
        # 双环的webster配时，参考(Ma, 2022)
        # control需要已有phase的信息
        self.sfr = 1368  # 所有车道的饱和流率, 使用文献中推荐的值，左转修正系数0.9
        control['split'] = np.zeros(8)
        movement2number = [[7,4],[1,6],[3,8],[5,2]]  # 流向映射到编号
        for i in range(4):  # 流向的流量比
            control['split'][movement2number[i][0]-1] = vph_m[i,0]/(self.sfr*0.9)  # 左转一条车道
            control['split'][movement2number[i][1]-1] = vph_m[i,1]/(2*self.sfr)  # 直行两条车道
        
        control['split'] = control['split']/(0.5*control['split'].sum())
        lb = max(self.g_min/control['split'].min(),self.G_min)
        ub = min(self.g_max/control['split'].max(),self.G_max)
        G = np.random.uniform(lb,ub)
        
        control['split'] = control['split']*G
        
        return control
        
    def split_webster(self,control,vph_m):
        # 四相位Webster配时
        # control需要已有phase的信息
        control['split'] = np.zeros(8)
        movement2number = [[7,4],[1,6],[3,8],[5,2]]  # 流向映射到编号
        for i in range(4):  # 流向的流量比
            control['split'][movement2number[i][0]-1] = vph_m[i,0]/(self.sfr*0.9)  # 左转一条车道
            control['split'][movement2number[i][1]-1] = vph_m[i,1]/(2*self.sfr)  # 直行两条车道
        
        m_order = self.swap2order(control['phase'])
        control['split'][:4] = control['split'][m_order-1].max(axis=0)
        control['split'][4:] = control['split'][:4]
        
        Y = control['split'][:4].sum()
        L = 4.0*((self.y + self.r)/2 + 3.0)
        
        if Y > 0.9:
            Y = 0.9
        C = (1.5*L+5.0)/(1-Y)
        G = C - 4.0*(self.y+self.r)
        G = self.G_min if G<self.G_min else self.G_max if G>self.G_max else G
        
        # TODO
        
        control['split'] /= control['split'].sum()/2
        control['split'] = control['split']*G
        
        return control
    
    def split_random_webster(self,control,vph_m):
        # 四相位Webster配时
        # control需要已有phase的信息
        self.sfr = 1368  # 所有车道的饱和流率, 使用文献中推荐的值，左转修正系数0.9
        control['split'] = np.zeros(8)
        movement2number = [[7,4],[1,6],[3,8],[5,2]]  # 流向映射到编号
        for i in range(4):  # 流向的流量比
            control['split'][movement2number[i][0]-1] = vph_m[i,0]/(self.sfr*0.9)  # 左转一条车道
            control['split'][movement2number[i][1]-1] = vph_m[i,1]/(2*self.sfr)  # 直行两条车道
        
        m_order = self.swap2order(control['phase'])
        control['split'][:4] = control['split'][m_order-1].max(axis=0)
        control['split'][4:] = control['split'][:4]  # 流向绿灯时间对齐
        
        control['split'] = control['split']/(0.5*control['split'].sum())
        lb = max(self.g_min/control['split'].min(),self.G_min)
        ub = min(self.g_max/control['split'].max(),self.G_max)
        G = np.random.uniform(lb,ub)
        
        control['split'] = control['split']*G
        
        return control

    def generate_default(self):
        # 用作热启动和过渡
        control = {}
        
        control = self.phase_fixed(control)  # 相序随机
        control = self.split_fixed(control)  # 绿灯时间固定
        
        return control
    
    def generate_sample(self,vph_m):
        control = {}
        
        control = self.phase_random(control)
        control = self.split_random_nema(control,vph_m)

        return control
        
    def generate_adaptive_webster(self,vph_m):
        # 给出webster配时方案
        control = {}
        
        control = self.phase_fixed(control)
        control = self.split_webster(control,vph_m)
        
        return control

    def generate_adaptive_nema(self,vph_m):
        control = {}
        
        control = self.phase_fixed(control)
        control = self.split_nema(control,vph_m)
        
        return control
    
    def generate_mpc(self,vph_m,mpc_controller):
        control,is_valid =  mpc_controller.generate_control(vph_m)
        if is_valid == True:
            return control
        elif is_valid == False:
            return self.generate_adaptive_nema(vph_m)

    def generate_fixed_time(self,param):
        order,time = param
        control = {}
        
        control = self.phase_fixed(control,order)
        control = self.split_fixed(control,time)
        
        return control

    def update(self,mode,monitor=None,mpc_controller=None,param=None):
        # 生成或获取控制方案
        # 采样策略，随机生成<优化格式>的控制方案
        # <优化格式>
        
        # sample: 绿灯时间的webster采样, <Webster>
        # random: 绿灯时间的随机采样，<随机>
        # webster: 绿灯时间基于Webster公式
        # mpc: 绿灯时间的优化采样，<优化>
        assert mode in ['default','sample','mpc',
                        'fixed-webster','fixed-nema',
                        'adaptive-webster','adaptive-nema','fixed-time']
        
        if mode == 'default':
            self.control = self.generate_default()
        elif mode == 'sample':
            self.control = self.generate_sample(monitor.vph_m)
        elif mode == 'mpc':
            self.control = self.generate_mpc(monitor.vph_m,mpc_controller)
        elif mode == 'adaptive-webster':
            self.control = self.generate_adaptive_webster(monitor.vph_m)
        elif mode == 'adaptive-nema':
            self.control = self.generate_adaptive_nema(monitor.vph_m)
        elif mode == 'fixed-time':
            self.control = self.generate_fixed_time(param)
    
    def swap2order(self,phase):
        # 输入表示相序交换的决策变量, 输出流向编号序列[1,2,...,8]
        g_swap,r1g1_swap,r1g2_swap,r2g1_swap,r2g2_swap = phase
        m_order_1 = np.array([1+1*r1g1_swap+2*g_swap,2-1*r1g1_swap+2*g_swap,3+1*r1g2_swap-2*g_swap,4-1*r1g2_swap-2*g_swap],dtype=int)
        m_order_2 = np.array([5+1*r2g1_swap+2*g_swap,6-1*r2g1_swap+2*g_swap,7+1*r2g2_swap-2*g_swap,8-1*r2g2_swap-2*g_swap],dtype=int)
        
        return np.stack([m_order_1,m_order_2],axis=0)
        
    def next_cycle(self):
        # 将下一个周期的控制方案部署到控制器
        # 接收<优化格式>的控制方案，转换成<执行格式>
        control = self.control  # phase and split
        
        # 各个流向1,2,...,8的绿灯时间，取整并应用约束
        split = control['split']
        split = [int(t) for t in split]
        split[5] = split[0]+split[1]-split[4]
        split[7] = split[2]+split[3]-split[6]
        
        # 5个0-1变量确定环内流向顺序，六个连续变量确定相位分隔
        m_order = self.swap2order(control['phase'])
        m_order_1 = m_order[0,:]
        m_order_2 = m_order[1,:]
        phase_cycle_1 = [m_order_1[0],'y','r',m_order_1[1],'y','r',m_order_1[2],'y','r',m_order_1[3],'y','r']
        phase_cycle_2 = [m_order_2[0],'y','r',m_order_2[1],'y','r',m_order_2[2],'y','r',m_order_2[3],'y','r']
        
        return cycle(phase_cycle_1),cycle(phase_cycle_2),split
    
    def output(self):
        # 输出<优化格式>的控制方案
        return self.control
    
    def is_to_update(self):
        return self.cycle_time == 0