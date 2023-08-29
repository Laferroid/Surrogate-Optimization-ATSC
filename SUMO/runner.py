# %%
# region import
import os
import sys
import time
import argparse
import numpy as np
import joblib
from tqdm import tqdm
from multiprocessing import Pool
import yaml

import traci
from traci import vehicle
from traci import lane
from traci import lanearea, multientryexit
from traci import junction
from traci import constants

# 导入与重载自定义模块
from traffic_controller import BaseTrafficController
from vehicle_generator import VehicleGenerator
from mpc_controller import MPCController
from utils import get_movement, inlet_map, try_connect, sumo_configurate

from uncertainty_surrogate_model import UncertaintySurrogateModel
from model_utils import resume

from sim_modules import Monitor, Recorder, Clock, Observer, Snapshooter

# endregion


# %%
def run_experiment(config):
    np.random.seed(0)

    sumocfg = sumo_configurate(config)
    sumocfg += ["--seed", "0"]  # 固定SUMO的随机数种子，如期望速度随机性和实际速度随机性
    try_connect(8, sumocfg)

    clock = Clock(config["step_num"], config["warm_up"], config["cycle_to_run"], config["time_to_run"])
    monitor = Monitor()
    observer = Observer(config)
    tc = BaseTrafficController(config["step_length"], config["time_interval"])
    veh_gen = VehicleGenerator(mode="linear")  # 流量生成模式
    recorder = Recorder("experiment")

    # 代理模型和mpc控制器
    surrogate_model = UncertaintySurrogateModel(config)
    resume(surrogate_model)
    mpc_controller = MPCController(surrogate_model, config)

    pbar = tqdm(total=config["time_to_run"], desc="Simulation experiment")

    snapshooter = Snapshooter(config)

    # 热启动完毕且当前周期结束时正式开始记录数据
    # 按周期数进行仿真
    while not clock.is_end("experiment"):
        # 仿真步进
        traci.simulationStep()
        clock.step()

        # 处理：单位时间(秒)结束时
        if clock.is_to_run():
            if clock.is_warm():  # 热启动
                monitor.run()  # 开始监测
                recorder.run(observer.output())  # 开始记录
            pbar.update(1)
            clock.run()
            veh_gen.run()  # 车辆生成器根据schedule生成车辆
            tc.run()

            # 处理：周期即将切换时
            if tc.is_to_update():
                if clock.is_warm():  # 开始控制
                    recorder.update(monitor.output(), tc.control)
                    mpc_controller.update(recorder.obs_list[-1], tc.control)
                clock.update()

                if clock.cycle_step in snapshooter.snapshot_point:  # snapshot when cycle is terminated
                    snapshot_dir = snapshooter.snapshot_dir + "snapshot_" + str(clock.cycle_step) + "/"
                    if not os.path.isdir(snapshot_dir):
                        os.mkdir(snapshot_dir)
                    snapshooter.snapshot(snapshot_dir, veh_gen, monitor)

                # region generate traffic control scheme for next cycle
                method = "mpc"  # mpc or baseline
                assert method in ["mpc", "benchmark"]
                if method == "mpc":
                    # MPC实验
                    if mpc_controller.warm_up == 0:
                        tc.update("mpc", monitor=monitor, mpc_controller=mpc_controller)
                    else:
                        tc.update("default")  # 使用默认方案
                elif method == "benchmark":
                    # 对比方案实验
                    if clock.time >= (monitor.vph_update_freq + clock.warm_up):
                        if mpc_controller.warm_up == 0:  # 即使不使用mpc也要记录context以绘制代理曲线(曲面)
                            mpc_controller.record_context()
                        tc.update("adaptive-nema", monitor=monitor)
                        # tc.update('fixed-time',param=(np.array([0,0,0,0,0]),20.0))
                    else:
                        tc.update("default")  # 使用默认方案
                # endregion

                monitor.reset()  # 重置性能指标监测器

    # 结束仿真
    traci.close()
    recorder.save(None, config["exp_dir"] + "result/")

    return {"mpc_controller": mpc_controller, "surrogate_model": surrogate_model, "demand": veh_gen.output()}


def run_sample(index, config):
    np.random.seed()  # 多进程跑数据，随机设置种子

    sumocfg = sumo_configurate(config)
    try_connect(8, sumocfg)

    clock = Clock(config["step_num"], config["warm_up"], config["cycle_to_run"], config["time_to_run"])
    monitor = Monitor()
    observer = Observer(config)
    tc = BaseTrafficController(config["step_length"], config["time_interval"])
    veh_gen = VehicleGenerator(mode="linear")
    recorder = Recorder("sample")

    pbar = tqdm(total=config["cycle_to_run"], desc=f"Simulation sampling {index}")

    # 热启动完毕且当前周期结束时正式开始记录数据
    # 按周期数进行仿真
    while not clock.is_end("sample"):
        # 仿真步进
        traci.simulationStep()  # 随机性
        clock.step()

        # 处理：单位时间(秒)结束时
        if clock.is_to_run():
            if clock.is_warm():  # 热启动
                monitor.run()  # 开始监测
                recorder.run(observer)  # 开始记录
            clock.run()
            veh_gen.run()  # 随机性考虑
            tc.run()  # 随机性考虑

            # 处理：周期即将切换时
            if tc.is_to_update():
                if clock.is_warm():  # 热启动
                    recorder.update(monitor, tc)  # 内部调用monitor的output函数
                    clock.update()
                    pbar.update(1)

                # 控制方案更新
                if clock.time >= (monitor.vph_update_freq + clock.warm_up):
                    tc.update("sample", monitor=monitor)  # 使用webster配时等饱和度进行采样
                else:
                    tc.update("default")  # 使用默认方案

                # 出现长度大于200m的排队，严重拥堵，终止仿真
                if clock.cycle_step > 0 and (monitor.queue_length_l[-1] > config["congestion_threshold"]).any():
                    break
                else:
                    monitor.reset()  # 重置性能指标监测器
    # 结束仿真
    traci.close()
    recorder.save(index, config["data_dir"] + f'simulation_data/{config["data_name"]}')