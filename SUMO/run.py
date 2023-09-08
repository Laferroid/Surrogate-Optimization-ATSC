# %% import
import time
import numpy as np
import joblib
from multiprocessing import Pool
from functools import partial

# 导入与重载自定义模块
from utils.helper_func import parse_config
from SUMO.runner import run_experiment, run_sample

# %% 多进程仿真，获取数据
if __name__ == "__main__":
    default_config_dir = "../configs/default_config.yaml"
    updated_config_dir = "../configs/updated_config.yaml"

    config = parse_config(default_config_dir,updated_config_dir)

    MODE = config['mode']

    if MODE == "sample":  # 多进程会各复制一份本文件，并且完全执行，需要把其他脚本部分注释掉或进行if判断
        start_time = time.perf_counter()

        with Pool(24) as pool:
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

        run_experiment(config)

        end_time = time.perf_counter()
        duration = end_time - start_time  # 毫秒
        print(f"运行时间:{duration//3600}h {(duration%3600)//60}m {int(duration%60)}s")
