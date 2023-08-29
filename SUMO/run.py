# %% import
import os
import sys
import time
import argparse
import numpy as np
import joblib
from tqdm import tqdm
from multiprocessing import Pool
import yaml
from functools import partial

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
from utils import get_movement, inlet_map, try_connect

from uncertainty_surrogate_model import UncertaintySurrogateModel
from model_utils import resume

from helper_func import parse_config, update_config

from sim_modules import Monitor, Recorder, Clock, Observer
from runner import run_experiment, run_sample

# %% 多进程仿真，获取数据
if __name__ == "__main__":
    MODE = "experiment"

    default_config_dir = "../configs/default_config.yaml"

    updated_config = parse_config()
    config = update_config(default_config_dir, updated_config)

    if MODE == "sample":  # 多进程会各复制一份本文件，并且完全执行，需要把其他脚本部分注释掉或进行if判断
        start_time = time.perf_counter()

        with Pool(8) as pool:
            pool.map(
                partial(run_sample, cycle_to_run=config["cycle_to_run"], time_to_run=config["time_to_run"]),
                [i for i in range(config["run_num"])],
            )
            pool.close()
            pool.join()
        # run_sample(0,config['cycle_to_run'],config['time_to_run'])

        end_time = time.perf_counter()
        duration = end_time - start_time  # 毫秒
        print(f"运行时间:{duration//3600}h {(duration%3600)//60}m {int(duration%60)}s")

    elif MODE == "experiment":
        start_time = time.perf_counter()

        res = run_experiment(config["time_to_run"], config["time_to_run"])
        # 保存demand信息
        np.save(config["exp_name"] + "demand_data.npy", res["demand"])
        # 保存MPC控制器的信息
        with open(config["exp_name"] + "mpc_data.pkl", "wb") as f:
            joblib.dump(
                {
                    "surrogate_result": res["mpc_controller"].surrogate_result,
                    "control_result": res["mpc_controller"].control_result,
                    "context_result": res["mpc_controller"].context_result,
                    "horizon_result": res["mpc_controller"].horizon_result,
                    "valid_result": res["mpc_controller"].valid_result,
                    "surrogate_model": res["surrogate_model"],
                },
                f,
            )

        end_time = time.perf_counter()
        duration = end_time - start_time  # 毫秒
        print(f"运行时间:{duration//3600}h {(duration%3600)//60}m {int(duration%60)}s")
