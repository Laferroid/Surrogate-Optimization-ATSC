from utils.helper_func import parse_config

default_config_dir = "../configs/default_config.yaml"
updated_config_dir = "../configs/updated_config.yaml"

config = parse_config(default_config_dir,updated_config_dir)

print(config["epochs"])
