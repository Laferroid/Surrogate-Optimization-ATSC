# %% import
import time
import numpy as np
import joblib
from multiprocessing import Pool
from functools import partial

# 导入与重载自定义模块
from utils.helper_func import parse_config, update_config
from SUMO.runner import run_experiment, run_sample

# %% 多进程仿真，获取数据
if __name__ == "__main__":
    default_config_dir = "../configs/default_config.yaml"
    updated_config = parse_config()
    config = update_config(default_config_dir, updated_config)

    MODE = config['mode']

    if MODE == "sample":  # 多进程会各复制一份本文件，并且完全执行，需要把其他脚本部分注释掉或进行if判断
        start_time = time.perf_counter()

        with Pool(8) as pool:
            pool.map(
                partial(run_sample, config=config),
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

        res = run_experiment(config)
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
