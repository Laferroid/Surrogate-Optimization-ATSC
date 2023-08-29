# %% import
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter


# %% process
def train_step(model, x, y):
    model.train()

    model.optimizer.zero_grad()
    y_pred = model(x)
    loss = model.loss_func(y_pred, y)
    with torch.no_grad():
        metric = model.metric_func(y_pred, y)

    # with torch.autograd.detect_anomaly():
    #     loss.backward()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
    model.optimizer.step()

    return loss.cpu().item(), metric


def validate_step(model, x, y):
    model.eval()

    y_pred = model(x)
    loss = model.loss_func(y_pred, y)
    with torch.no_grad():
        metric = model.metric_func(y_pred, y)

    return loss.cpu().item(), metric


def resume(model, model_dir):
    # 断点续训：寻找并导入checkpoint
    saved_step = 0
    saved_file = None
    current_step = 0

    global_epoch = 1
    global_step = 1
    global_loss = 0.0

    # model_dir = model.model_dir
    if os.path.isdir(model_dir + "checkpoints/"):
        checkpoints = os.listdir(model_dir + "checkpoints/")
        if len(checkpoints) > 0:
            for file in checkpoints:
                if file.startswith("checkpoint"):
                    tokens = file.split(".")[0].split("-")
                    if len(tokens) != 2:
                        continue
                    current_step = int(tokens[1])
                    if current_step >= saved_step:
                        saved_file = file
                        saved_step = current_step
            checkpoint_path = model_dir + "checkpoints/" + saved_file
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint["model"])
            global_epoch = checkpoint["epoch"] + 1  # 设置开始的epoch
            global_step = checkpoint["step"] + 1  # 设置开始的step
            global_loss = checkpoint["loss"]
        else:
            print("No exisiting model !")
    else:
        print("No exisiting model !")

    return global_epoch, global_step, global_loss


def check(model, epoch, step, loss, model_dir):
    # 断点续训，保存checkpoint
    checkpoint = {"model": model.state_dict(), "epoch": epoch, "step": step, "loss": loss}
    if not os.path.isdir(model_dir + "checkpoints/"):
        os.mkdir(model_dir + "checkpoints/")
    torch.save(checkpoint, model_dir + "checkpoints/" + "checkpoint-%s.pth" % (str(step)))


def train(model, train_dl, val_dl, config):
    tb_logger = TBLogger(model, config)
    tb_logger.setup_training()

    # wandb_logger = WandBLogger(model, config)
    # wandb_logger.setup_training()

    # train loop
    for _ in range(config["epochs"]):
        pbar = tqdm(total=len(train_dl), desc=f"training epoch {tb_logger.global_epoch}")
        # train
        for _, (x, y) in enumerate(train_dl):
            loss, _ = train_step(model, x, y)
            tb_logger.global_loss += loss
            pbar.update(1)

            if tb_logger.global_step % tb_logger.train_log_freq == 0:
                tb_logger.step_log(label="loss", data={"train": tb_logger.global_loss / tb_logger.train_log_freq})

                # wandb_logger.step_log(label=None, data={"train_loss": tb_logger.global_loss / tb_logger.train_log_freq})

                tb_logger.global_loss = 0.0

            if tb_logger.global_step % tb_logger.val_log_freq == 0:
                torch.cuda.empty_cache()  # 开始验证，清理一下缓存
                # validate
                loss_sum_val = 0.0
                metric_sum_val = {metric: 0.0 for metric in model.metrics}

                for step_val, (x, y) in enumerate(val_dl, 1):
                    with torch.no_grad():
                        loss_val, metric_val = validate_step(model, x, y)
                    loss_sum_val += loss_val
                    for metric in model.metrics:
                        metric_sum_val[metric] += metric_val[metric]

                for metric in model.metrics:
                    name_list = [str(i) + "-step-ahead" for i in range(1, model.lookahead + 1)]

                    tb_logger.step_log(
                        metric, {key: value / step_val for (key, value) in zip(name_list, metric_sum_val[metric])}
                    )

                    # wandb_logger.step_log(
                    #     label=None,
                    #     data={
                    #         (metric + ":" + key): value / step_val
                    #         for (key, value) in zip(name_list, metric_sum_val[metric])
                    #     },
                    # )

                tb_logger.step_log("loss", {"val": loss_sum_val / step_val})

                # wandb_logger.step_log(label=None, data={"val_loss": loss_sum_val / step_val})

            # 断点续训，stepwise-checkpoint
            tb_logger.step_check()

            tb_logger.global_step += 1

        pbar.close()

        # 断点续训，epochs级checkpoint
        tb_logger.epoch_check()
        tb_logger.global_epoch += 1

        model.scheduler.step()  # 随epoch衰减

    # wandb_logger.finish()


class TBLogger:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.train_log_freq = config["train_log_freq"]
        self.val_log_freq = config["val_log_freq"]
        self.step_check_freq = config["step_check_freq"]

    def setup_training(self):
        model_dir = self.config["model_dir"]

        # 断点续训：寻找并导入checkpoint，正在处理的epoch和step，以及对应loss
        res = resume(self.model, self.config["model_dir"])
        self.global_epoch, self.global_step, self.global_loss = res
        self.model.to(self.config["device"])

        # tensorboard:初始化，若路径不存在会创建路径
        self.step_writer = SummaryWriter(log_dir=model_dir + "tb-step-logs", purge_step=self.global_step)
        self.epoch_writer = SummaryWriter(log_dir=model_dir + "tb-epoch-logs", purge_step=self.global_epoch)

    def step_log(self, label, data):
        self.step_writer.add_scalars(label, data, self.global_step)

    def step_check(self):
        if self.global_step % self.step_check_freq == 0:
            check(self.model, self.global_epoch, self.global_step, self.global_loss, self.config['model_dir'])

    def epoch_check(self):
        check(self.model, self.global_epoch, self.global_step - 1, self.global_loss, self.config['model_dir'])


class WandBLogger:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def setup_training(self):
        wandb.init(
            project="surrogate-model",
            job_type=self.config["job_type"],
            group=self.config["exp_group"],
            name=self.config["exp_name"],
            id=self.config["exp_id"],
            config=self.config,
            save_code=False,
            resume="allow",
        )
        # artifact = wandb.Artifact(name='train_script',type='code')
        # artifact.add_file('/models/model_utils.py')
        wandb.watch(models=self.model, log="all", log_freq=200, log_graph=True)

    def step_log(self, label, data):
        wandb.log(data)

    def step_check(self):
        pass

    def finish(self):
        wandb.finish()


def hook_model(model, logger):
    def tb_hook(name):
        def hook(module, input, output):
            if isinstance(module, (nn.ReLU)):  # 所有ReLU激活函数的输出
                try:
                    logger.step_wirter.add_histogram(name + "/output", output[0], global_step=logger.global_step)
                except Exception:
                    pass

        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(tb_hook(name))


# %% data
class MyDataset(Dataset):
    def __init__(self, config):
        super(MyDataset, self).__init__()
        self.config = config
        self.dir = self.config["data_dir"] + "training_data/" + self.config["data_name"] + "/"
        self.lookback = self.config["lookback"]  # 从数据中选取的时窗大小
        self.lookahead = self.config["lookahead"]  # 从数据中选取的时窗大小
        self.sample_size = self.config["sample_size"]

        assert self.lookback <= self.sample_size[0], "lookback out of range"
        assert self.lookahead <= self.sample_size[1], "lookahead out of range"
        with open(self.dir + "sample2chunk.pkl", "rb") as f:
            self.sample2chunk = joblib.load(f)

    def __getitem__(self, index):
        # 在这里把数据方法device上
        chunk_index, sample_index = self.sample2chunk[index]

        other = torch.load(self.dir + f"other_{chunk_index}_{sample_index}.pth")  # other: (delay,delay_m,tc)
        obs = np.empty(self.lookback, dtype=object)
        # obs: (lookahead,frames!,***)
        for i, v in enumerate(np.load(self.dir + f"obs_{chunk_index}_{sample_index}.npz", allow_pickle=True)["arr_0"]):
            obs[i] = v  # 别动，内存不够

        obs = obs[None, (self.sample_size[0] - self.lookback) : self.sample_size[0]]

        tc = other["tc"][
            None, (self.sample_size[0] - self.lookback) : (self.sample_size[0] + self.lookahead)
        ]  # tc:(1,lookback + lookahead,*)
        x = {"obs": obs, "tc": tc}
        y = other["timeloss"][None, : self.lookahead]  # (1,lookahead), 数据可以更长但只取lookahead那么长

        return (x, y)

    def __len__(self):
        return len(self.sample2chunk)


def train_val_test_split(dataset, splits, seed):
    train_size = int(splits[0] * len(dataset))
    val_size = int(splits[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # 随机数种子需要设置
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    return train_ds, val_ds, test_ds


def my_collate_fn(samples):
    x = {}
    x["obs"] = np.concatenate([sample[0]["obs"] for sample in samples], axis=0)
    x["tc"] = torch.cat([sample[0]["tc"] for sample in samples], dim=0)

    y = torch.cat([sample[1] for sample in samples], dim=0)

    return (x, y)


def get_dataloader(dataset, config):
    train_ds, val_ds, test_ds = train_val_test_split(dataset, config["split"], seed=config["seed"])

    # dataloader使用固定的随机数种子
    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        collate_fn=my_collate_fn,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        num_workers=config["num_workers"],
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        collate_fn=my_collate_fn,
        shuffle=False,
        drop_last=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        num_workers=config["num_workers"],
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        collate_fn=my_collate_fn,
        shuffle=False,
        drop_last=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        num_workers=1,
    )
    # 为严格保证batch_size，设置drop_last=True
    return train_dl, val_dl, test_dl
