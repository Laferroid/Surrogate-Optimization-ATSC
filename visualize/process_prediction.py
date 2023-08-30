# %%
# import
import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

sys.path.append("../models/")
from uncertainty_surrogate_model import UncertaintySurrogateModel
from model_utils import MyDataset, train_val_test_split, resume
from mdn_model import mdn_mean

# region

SAMPLE_SIZE = (8, 4)
BATCH_SIZE = 16
LOOKBACK = 8
LOOKAHEAD = 4

mpl.rcParams["font.family"] = ["Times New Roman", "SimSun"]
mpl.rcParams["mathtext.fontset"] = "stix"  # 设置数学公式字体为stix
mpl.rcParams["font.size"] = 9  # 按磅数设置的
mpl.rcParams["figure.dpi"] = 300
cm = 1 / 2.54  # centimeters in inches
mpl.rcParams["figure.figsize"] = (12 * cm, 8 * cm)
mpl.rcParams["savefig.dpi"] = 900
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.axis"] = "both"
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["axes.facecolor"] = "white"

surrogate_dir = "../results/experiment/surrogate/"  # 结果保存目录
data_dir = "../data/training_data/standard-revised/"  # 数据读取目录
model_dir = "../results/standard-test/test-revised-standard/"

model_config = {
    "model_dir": model_dir,
    "batch_size": 1,
    "lookback": 8,
    "lookahead": 4,
    "device": "cpu",
    "train_predict_mode": "recursive",
    "sample_size": (8, 4),
    "seed": 0,
    "num_components": 48,
}

dl_config = {
    "seed": 0,
    "data_dir": data_dir,
    "batch_size": 32,
    "lookback": 8,
    "lookahead": 4,
    "sample_size": (8, 4),
    "num_workers": 2,
    "split": [0.8, 0.1, 0.1],
}
# endregion

# %% 数据集
ds = MyDataset(dl_config)
train_ds, val_ds, test_ds = train_val_test_split(ds, dl_config["split"], dl_config["seed"])

# %% 获取点模型离线预测结果
# pmodel_config = model_config.copy()
# pmodel_config['model_dir'] = "../results/standard-test/pmodel/"
# def save_value_prediction(dataset,model_config):
#     model =  ValueSurrogateModel(model_config)
#     resume(model)
#     model.eval()

#     num_samples = len(dataset)
#     y_pred = torch.zeros((num_samples,LOOKAHEAD))

#     pbar = tqdm(total=num_samples)
#     for i in range(num_samples):
#         x,_ = dataset[i]
#         with torch.no_grad():
#             y_pred[i] = model.forward(x)
#         pbar.update(1)

#     np.save(surrogate_dir+'test_value_pred.npy',y_pred)

# save_value_prediction(test_ds,pmodel_config)


# %% 获取分布模型离线预测结果
def save_uncertainty_prediction(dataset, model_config):
    model = UncertaintySurrogateModel(model_config)
    resume(model)
    model.eval()

    num_samples = len(dataset)
    c = torch.zeros((num_samples, model_config["lookahead"], model_config["num_components"], 1))
    mu = torch.zeros((num_samples, model_config["lookahead"], model_config["num_components"], 1))
    sigma = torch.zeros((num_samples, model_config["lookahead"], model_config["num_components"], 1))
    y_pred = torch.zeros((num_samples, model_config["lookahead"]))

    pbar = tqdm(total=num_samples, desc="Loading:")
    for i in range(num_samples):
        x, _ = dataset[i]
        with torch.no_grad():
            c[[i]], mu[[i]], sigma[[i]] = model.forward(x)
            mu[[i]], sigma[[i]] = model.label_norm.backward(mu[[i]], sigma[[i]])
            y_pred[[i]] = mdn_mean((c[[i]], mu[[i]], sigma[[i]])).squeeze(dim=-1)
        pbar.update(1)

    np.save(surrogate_dir + "val_uncertainty_pred.npy", y_pred)
    np.savez(surrogate_dir + "val_uncertainty_output.npz", c=c, mu=mu, sigma=sigma)


save_uncertainty_prediction(val_ds, model_config)
