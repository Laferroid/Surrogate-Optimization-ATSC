# %%
# region
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .mdn_model import MixtureDensityNetwork, LinearResBlock, ReversibleBatchNorm
from .mdn_model import mdn_mean

# endregion


# %% model
class UncertaintySurrogateModel(nn.Module):
    def __init__(self, config):
        super(UncertaintySurrogateModel, self).__init__()
        self.config = config

        # self.model_dir = self.config["model_dir"]
        self.device = self.config["device"]

        self.lookback = self.config["lookback"]
        self.lookahead = self.config["lookahead"]
        self.batch_size = self.config["batch_size"]
        self.num_components = self.config["num_components"]
        self.num_channels = self.config["num_channels"]
        self.hidden_size = self.config["hidden_size"]

        self.label_norm = ReversibleBatchNorm(4)  # 标签归一化

        self.conv = nn.Sequential(
            nn.Conv2d(20, self.num_channels, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2), padding=0),
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2), padding=0),
            nn.Flatten(),
        )
        hidden_size = self.hidden_size
        self.D = 2  # 1:单向，2:双向
        l_enc_num_layers = 2

        self.l_enc_input_layer = LinearResBlock(6 * self.num_channels, hidden_size, hidden_size)

        # lower encoder
        # sequential的模型只接受单输入,因此LSTM需单独写出
        self.l_enc_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=l_enc_num_layers,
            dropout=0.0,
            bidirectional=bool(self.D - 1),
        )
        self.l_enc_output_layer = LinearResBlock(self.D * hidden_size, hidden_size, hidden_size)

        # size: phase=5 + split=8 + target_func=3*8
        tc_size = 13  # 控制方案维度，不包括可变车道切换相位
        self.tc_output_layer = LinearResBlock(tc_size, hidden_size, hidden_size)

        # upper encoder
        self.u_enc_input_layer = LinearResBlock(hidden_size + hidden_size, hidden_size, hidden_size)
        self.u_enc_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, dropout=0.0)
        # upper context
        self.u_enc_output_layer = LinearResBlock(hidden_size, hidden_size, hidden_size)

        # decoder
        self.dec_input_layer = LinearResBlock(tc_size, hidden_size, hidden_size)  # feed mean forward

        # self.dec_layer = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=1,dropout=0.0)
        self.dec_layer = nn.LSTMCell(
            input_size=hidden_size + 1, hidden_size=hidden_size
        )  # because of recursive feeding
        # output delay
        self.dec_output_layer = LinearResBlock(hidden_size, hidden_size, hidden_size)

        # 每一步预测都使用相同的mdn
        self.mdn_layer = MixtureDensityNetwork(hidden_size, hidden_size, 1, self.num_components)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])  # ,weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, last_epoch=-1)

        self.metrics = ["rmse", "mae", "mape", "wape"]

    def _lower_encoding(self, obs):
        lower_context = []
        for i in range(self.lookback):
            seq_lens = []
            for s in obs[:, i]:  # obs[:,i]: (batch,frames!,***)
                seq_lens.append(s.shape[0])
                for j in range(self.batch_size):
                    obs[j, i] = obs[j, i].to(self.device)  # 在这里把观测移到GPU中
            # 卷积层
            frame = torch.cat(list(obs[:, i]), dim=0)
            obs_conv = self.conv(frame)  # obs_conv: (batch&frames!,*)
            obs_conv = self.l_enc_input_layer(obs_conv)  # obs_conv: (batch&frames!,*)
            obs_conv = torch.split(obs_conv, seq_lens, dim=0)  # obs_conv: (batch,frames!,*)

            # lower encoder
            obs_padded = pad_sequence(obs_conv)  # obs_padded: (frames,batch,*)
            obs_packed = pack_padded_sequence(obs_padded, seq_lens, enforce_sorted=False)
            # _,(l_enc_h,_) = self.l_enc_layer(obs_packed,(None,None))   # 错误的，None会参与计算
            _, (l_enc_h, _) = self.l_enc_layer(obs_packed)  # l_enc_h: (D*num_layers,batch,*)
            if self.D == 2:
                lower_context.append(torch.cat([l_enc_h[-2, :, :], l_enc_h[-1, :, :]], dim=-1))  # append: (batch,2*)
            elif self.D == 1:
                lower_context.append(l_enc_h[-1, :, :])  # append: (batch,*)

        lower_context = torch.stack(lower_context, dim=1)  # lower_context: (batch,lookback,*)
        lower_context = self.l_enc_output_layer(lower_context)  # lower_context: (batch,lookback,*)

        return lower_context

    def _upper_encoding(self, tc_back, lower_context):
        tc_back = self.tc_output_layer(tc_back)  # tc_back: (batch,lookback,*)

        u_enc_input = torch.cat([tc_back, lower_context], dim=-1)  # u_enc_input: (batch,lookback,*)
        u_enc_input = self.u_enc_input_layer(u_enc_input).transpose(0, 1)  # u_enc_input: (lookback,batch,*)
        _, (u_enc_h, _) = self.u_enc_layer(u_enc_input)  # u_enc_h: (num_layers,batch,*)

        upper_context = u_enc_h[-1, :, :]
        upper_context = self.u_enc_output_layer(upper_context)  # upper_context: (batch,*)

        return upper_context

    def encoding(self, obs, tc_back):
        lower_context = self._lower_encoding(obs)
        upper_context = self._upper_encoding(tc_back, lower_context)
        return upper_context

    def _decoding(self, upper_context, tc_ahead):
        upper_context = upper_context.contiguous()  # upper_context: (batch,*)

        tc_ahead = self.dec_input_layer(tc_ahead)  # tc_ahead: (batch,lookahead,*)
        h_state = upper_context  # 用upper_context初始化hidden_state
        c_state = torch.zeros_like(upper_context, device=self.device)

        if self.config["train_predict_mode"] == "recursive":  # 将mdn输出的mean进行feed forward
            out = torch.zeros((self.batch_size, 1), device=self.device)
            c_list, mu_list, sigma_list = [], [], []
            for i in range(tc_ahead.shape[1]):
                (h_state, c_state) = self.dec_layer(
                    torch.cat([tc_ahead[:, i, :], out], dim=-1), (h_state, c_state)
                )  # dec_output: (batch,*)
                out = self.dec_output_layer(h_state)  # (batch,*)
                c, mu, sigma = self._uncertainty_qualify(out)
                out = mdn_mean((c, mu, sigma))
                c_list.append(c)
                mu_list.append(mu)
                sigma_list.append(sigma)
            c = torch.stack(c_list, dim=1)
            mu = torch.stack(mu_list, dim=1)
            sigma = torch.stack(sigma_list, dim=1)

        elif self.config["train_predict_mode"] == "direct":
            dec_output = []
            z = torch.zeros((self.batch_size, 1), device=self.device)
            for i in range(tc_ahead.shape[1]):
                (h_state, c_state) = self.dec_layer(
                    torch.cat([tc_ahead[:, i, :], z], dim=-1), (h_state, c_state)
                )  # dec_output: (batch,*)
                dec_output.append(h_state)
            dec_out = self.dec_output_layer(torch.stack(dec_output, dim=1))  # (batch,lookahead,*)
            c, mu, sigma = self._uncertainty_qualify(dec_out)  # (batch,lookahead,num_components,1)

        y_pred = (c, mu, sigma)

        return y_pred

    def _uncertainty_qualify(self, x):
        # 单个周期的不确定性估计, 根据Bishop的原始论文这是合理的
        # x: (*,input_size)
        c, mu, sigma = self.mdn_layer.forward(x)
        # c: (*,num_components,1)
        # mu: (*,num_components,output_size)
        # sigma: (*,num_components,output_size)

        return (c, mu, sigma)

    def forward(self, x):
        obs = x["obs"]  # obs: (batch,lookback,frames!,***)
        tc = x["tc"].to(self.device)  # tc: (batch,lookback+lookahead,*)
        lower_context = self._lower_encoding(obs)

        tc_back = tc[:, : self.lookback, :]  # tc_back: (batch,lookback,*)
        tc_ahead = tc[:, self.lookback :, :]  # tc_ahead: (batch,lookahead,*)

        upper_context = self._upper_encoding(tc_back, lower_context)

        y_pred = self._decoding(upper_context, tc_ahead)

        return y_pred

    def predict(self, upper_context, tc_ahead):
        # 函数_decoding的简单封装
        y_pred = self._decoding(upper_context, tc_ahead)
        c, mu, sigma = y_pred
        mu, sigma = self.label_norm.backward(mu, sigma)
        return (c, mu, sigma)  # 输出预测的分布

    def loss_func(self, y_pred, y):
        # y: (batch,lookahead)
        # c: (batch,lookahead,num_components,1)
        # mu: (batch,lookahead,num_components,output_size)
        # sigma: (batch,lookahead,num_components,output_size)
        y = y.unsqueeze(-1).to(self.device)
        y = self.label_norm.forward(y)
        loss = self.mdn_layer.loss_func(y_pred, y)

        return loss

    def metric_func(self, y_pred, y):
        """_summary_

        Parameters
        ----------
        y_pred : _type_
            (c,mu,sigma)
            c : Tensor
                (batch_size,lookahead,num_components)
            mu : Tensor
                (batch_size,lookahead,num_components,output_size)
            sigma : Tensor
                (batch_size,lookahead,num_components,output_size)
        y : _type_
            (batch_size,lookahead)

        Returns
        -------
        _type_
            _description_
        """
        self.eval()
        c, mu, sigma = y_pred
        mu, sigma = self.label_norm.backward(mu, sigma)
        with torch.no_grad():
            y_pred = mdn_mean((c, mu, sigma)).squeeze(dim=-1)

        y = y.to(self.device)

        rmse = torch.sqrt(((y - y_pred) ** 2).mean(dim=0))
        mae = torch.abs(y - y_pred).mean(dim=0)
        mape = torch.abs((y - y_pred) / y).mean(dim=0)
        wape = torch.abs(y - y_pred).sum(dim=0) / y.sum(dim=0)

        self.train()
        return {
            "rmse": rmse.cpu().numpy(),
            "mae": mae.cpu().numpy(),
            "mape": mape.cpu().numpy(),
            "wape": wape.cpu().numpy(),
        }
