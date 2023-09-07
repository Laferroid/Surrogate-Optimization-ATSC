# %%
# region import
import os

from utils.helper_func import parse_config
from utils.process_func import agg_sim_data

# endregion


# %%
if __name__ == "__main__":
    default_config_dir = "../configs/default_config.yaml"

    config = parse_config(default_config_dir)

    data_dir = config["data_dir"] + config["data_name"] + "/simulation_data/"
    output_dir = config["data_dir"] + config["data_name"] + "/training_data/"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    agg_sim_data(data_dir, output_dir, config)
