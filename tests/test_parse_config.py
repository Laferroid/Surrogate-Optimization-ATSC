from utils.helper_func import parse_config

default_config_dir = "../configs/default_config.yaml"

config = parse_config(default_config_dir)

print(config["epochs"])
