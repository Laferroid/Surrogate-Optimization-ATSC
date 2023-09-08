#%%
import numpy as np
import joblib
from utils.helper_func import parse_config

default_config_dir = "../configs/default_config.yaml"
updated_config_dir = "../configs/updated_config.yaml"

config = parse_config(default_config_dir,updated_config_dir)

with open(config['data_dir']+config['data_name']+'/simulation_data/'+'simulation_data_4.pkl',"rb") as f:
    data = joblib.load(f)

#%%
obs = data['obs']
assert obs[0].dtype == np.float32
assert data['obs'][0]

data['tc']
data['arrive']
data['depart']
data['queue_length']
data['timeloss']
