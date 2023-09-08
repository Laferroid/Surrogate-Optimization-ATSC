# %%
# region import
import importlib as imp
import os
import sys
from functools import partial

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed

from scipy.linalg import block_diag
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import norm
from torch import nn

from models.mdn_model import aleatoric_uncertainty, epistemic_uncertainty, mdn_mean, variance
from utils.process_func import frame_process

# endregion


# %%
class MPCController:
    def __init__(self, model, config):
        self.config = config

        self.lookback = config["lookback"]
        self.lookahead = config["lookahead"]
        self.warm_up = self.lookback  # 过去时间窗作为热启动周期
        self.num_enumerations = config["num_enumerations"]  # 比较关键，可能需要进行敏感性分析
        self.num_restarts = config["num_restarts"]

        self.g_min, self.g_max = config["g_range"]
        self.G_min, self.G_max = config["C_range"]

        self.gamma = config["gamma"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]

        self.y = config["yellow"]
        self.r = config["red"]
        self.tl = config["timeloss"]

        # 延误预测模型
        self.model = model
        self.model.batch_size = 1  # 在线代理时，模型的输入不成batch

        self.upper_context = None

        # 记录并更新代理模型的输入，因此神经网络输入的格式
        self.obs_back = np.empty((1, self.lookback), dtype=object)
        self.tc_back = torch.zeros((1, self.lookback, 8+5))

        # 记录MPC控制器的一些内部变量
        self.surrogate_result = []  # 代理结果，用于可视化，列表中元素是array的格式
        self.control_result = []  # 控制结果，用于计算不确定性输出，是神经网络输入的格式
        self.horizon_result = []  # 动态预测时间窗，每次最优控制的时窗, list of int
        self.valid_result = []  # 最优控制是否有效, list of boolean
        self.context_result = []  # 记录过程中的upper context用于分析代理模型

        self.phase_list = []
        for i in range(2**5):
            self.phase_list.append(
                np.array(
                    [(i // 2**4) % 2, (i // 2**3) % 2, (i // 2**2) % 2, (i // 2**1) % 2, (i // 2**0) % 2]
                )
            )

    # 相位方案随机选取
    def _multi_enumerations(self, lookahead):
        # 枚举整数变量的方案, 包括多步lookahead
        indexes = np.random.choice(32, (self.num_enumerations - 1, lookahead))
        phases = np.stack(self.phase_list, axis=0)[indexes, :]
        phases = np.concatenate([np.expand_dims(np.zeros((lookahead, 5)), axis=0), phases], axis=0)
        return phases  # array(num_enumerations,lookahead,5)

    # 绿灯时间基于等饱和度
    def _multi_restarts(self, vph_m, lookahead):
        # 基于等饱和度原则，多次随机启动
        sfr = 1880  # 所有车道的饱和流率, 仿真测量，左转修正系数0.9
        split = np.zeros(8)
        movement2number = [[7, 4], [1, 6], [3, 8], [5, 2]]  # 流向映射到编号
        for i in range(4):  # 流向的流量比
            split[movement2number[i][0] - 1] = vph_m[i, 0] / (sfr * 0.9)  # 左转一条车道
            split[movement2number[i][1] - 1] = vph_m[i, 1] / (2 * sfr)  # 直行两条车道

        y1 = split[0] + split[1]
        y2 = split[2] + split[3]
        y3 = split[4] + split[5]
        y4 = split[6] + split[7]

        if y1 >= y3:
            split[[4, 5]] *= y1 / y3
        elif y1 < y3:
            split[[0, 1]] *= y3 / y1

        if y2 >= y4:
            split[[6, 7]] *= y2 / y4
        elif y2 < y4:
            split[[2, 3]] *= y4 / y2

        Y = max(y1, y3) + max(y2, y4)
        L = 4.0 * (self.r + 3.0)

        if Y > 0.9:
            Y = 0.9
        C = (1.5 * L + 5.0) / (1 - Y)
        G = C - 4.0 * (3.0 + self.r)

        split = split / (0.5 * split.sum())
        lb = max(self.g_min / split.min(), self.G_min)
        ub = min(self.g_max / split.max(), self.G_max)
        G = lb if G < lb else ub if G > ub else G

        G_all = G * np.ones(self.num_restarts)
        G_all[:-1] = np.random.uniform(G * 0.8, G * 1.2, size=self.num_restarts - 1)

        G_all = np.random.uniform(lb, ub, size=self.num_restarts)
        splits = split[None, :] * G_all[:, None]
        splits = np.tile(splits[:, None, :], (1, lookahead, 1))  # (num_restarts,lookahead,8)
        # 外推到后续周期，这是初始解

        return splits  # array(num_restarts,lookahead,8)

    # 优化问题求解
    '''
    def _optimize_d(self, phases, splits):
        # phases: array(num_enumerations,lookahead,5)
        # splits: array(num_restarts,lookahead,8)
        lookahead = phases.shape[1]
        optimal_point = None  # 最优解
        optimal_value = 1e6  # 最优值

        # results = Parallel(n_jobs=4)(delayed(par_obs_process)(i) for (i,sample) in enumerate(obs_data))

        for i in range(len(phases)):
            phase = phases[i]  # phase: array(lookahead,5)
            objective_func = partial(self._objective_func, phase=phase)
            for j in range(len(splits)):
                split = splits[j]  # split: array(lookahead,8)

                # split = split[:,1:7].reshape(-1)  # array(lookahead*6)
                split = split.reshape(-1)

                bounds = Bounds(15.0, 60.0)
                A1 = block_diag(*(lookahead * [np.array([1, 1, 1, 1, 0, 0, 0, 0])]))
                A2 = block_diag(*(lookahead * [np.array([1, 1, 0, 0, -1, -1, 0, 0])]))
                A3 = block_diag(*(lookahead * [np.array([0, 0, 1, 1, 0, 0, -1, -1])]))
                lb1, ub1 = self.G_min * np.ones(lookahead), self.G_max * np.ones(lookahead)
                lb2, ub2 = np.zeros(2 * lookahead), np.zeros(2 * lookahead)
                linear_constraint = LinearConstraint(
                    np.vstack((A1, A2, A3)), np.concatenate([lb1, lb2]), np.concatenate([ub1, ub2]), keep_feasible=True
                )
                res = minimize(
                    fun=objective_func,
                    x0=split,
                    method="trust-constr",
                    jac=True,
                    bounds=bounds,
                    constraints=linear_constraint,
                    options={"disp": True, "verbose": 0},
                )
                # res = minimize(fun=objective_func,x0=split,method='BFGS',jac=True,
                #                options={'disp':True,'gtol':1e-8,'maxiter':100})
                # res = minimize(fun=objective_func,x0=split,method='Nelder-Mead',options={'disp':True})

                if res.fun < optimal_value:
                    optimal_value = res.fun
                    split = res.x
                    # 复原
                    # x = np.zeros((lookahead,8))
                    # x[:,1:7] = split.reshape(lookahead,6)
                    # x[:,0] = x[:,4] + x[:,5] - x[:,1]
                    # x[:,7] = x[:,2] + x[:,3] - x[:,6]
                    # split = x  # (lookahead,8)
                    split = split.reshape(lookahead, 8)

                    # 输出<代理格式>， (1,lookahead,5+8)
                    optimal_point = (
                        torch.from_numpy(np.concatenate([phase, split], axis=-1)).unsqueeze(0).to(torch.float32)
                    )

        return optimal_point, optimal_value
    '''

    def _optimize(self, phases, splits):
        # phases: array(num_enumerations,lookahead,5)
        # splits: array(num_restarts,lookahead,8)
        lookahead = phases.shape[1]
        optimal_point = None  # 最优解
        optimal_value = 1e6  # 最优值

        def par_func(j):
            split = splits[j]  # split: array(lookahead,8)
            split = split.reshape(-1)  # split: array(lookahead*8)

            bounds = Bounds([self.g_min,self.g_max])
            A1 = block_diag(*(lookahead * [np.array([1, 1, 1, 1, 0, 0, 0, 0])]))
            A2 = block_diag(*(lookahead * [np.array([1, 1, 0, 0, -1, -1, 0, 0])]))
            A3 = block_diag(*(lookahead * [np.array([0, 0, 1, 1, 0, 0, -1, -1])]))
            lb1, ub1 = self.G_min * np.ones(lookahead), self.G_max * np.ones(lookahead)
            lb2, ub2 = np.zeros(2 * lookahead), np.zeros(2 * lookahead)
            linear_constraint = LinearConstraint(
                np.vstack((A1, A2, A3)), np.concatenate([lb1, lb2]), np.concatenate([ub1, ub2]), keep_feasible=False
            )
            res = minimize(
                fun=objective_func,
                x0=split,
                method="trust-constr",
                jac=False,
                bounds=bounds,
                constraints=linear_constraint,
                options={"disp": True, "verbose": 0, "gtol": 1e-8},
            )

            optimal_value = res.fun
            split = res.x

            split = split.reshape(lookahead, 8)

            # 输出<代理格式>， (1,lookahead,5+8)
            optimal_point = torch.from_numpy(np.concatenate([phase, split], axis=-1)).unsqueeze(0).to(torch.float32)

            return optimal_point, optimal_value

        for i in range(len(phases)):
            phase = phases[i]  # phase: array(lookahead,5)
            objective_func = partial(self._objective_func, phase=phase)

            results = []
            results = Parallel(n_jobs=8)(delayed(par_func)(j) for j in range(len(splits)))
            # for j in range(len(splits)):
            #     results.append(par_func(j))

            point, value = min(results, key=lambda x: x[1])
            if value < optimal_value:
                optimal_point = point
                optimal_value = value

        return optimal_point, optimal_value

    def _objective_func(self, split, phase):
        # split: array(lookahead*6)
        # phase: array(lookahead,5)
        lookahead = phase.shape[0]
        self.model.eval()

        split = split.reshape(lookahead, 8)

        cycle_length = torch.from_numpy(self.split_refined(split).sum(-1) * 0.5)  # array(lookahead)

        # 每次优化都重新创建tc_ahead张量，因此无需清空梯度
        tc_ahead = (
            torch.from_numpy(np.concatenate([phase, split], axis=-1)).to(torch.float32).unsqueeze(dim=0)
        )  # (1,lookahead,8+5)
        tc_ahead.requires_grad = True

        output = mdn_mean(self.model.predict(self.upper_context, tc_ahead))[0]  # (lookahead)

        f = output[0] + (self.gamma ** cycle_length[:-1].cumsum(0) * output[1:]).sum()
        # f /= tc_ahead[0,:,5:].sum()*0.5
        # f += output[-1] * (self.gamma ** (cycle_length.sum())) / (1 - self.gamma ** cycle_length[-1])

        value = f.item()

        # f.backward()
        # derivative = tc_ahead.grad[:,:,5:].detach().numpy().reshape(-1)  # array(lookahead*8)

        self.model.optimizer.zero_grad()  # 清空模型权重的梯度

        return value  # ,derivative

    def split_refined(self, split):
        # split: array(lookahead,8)
        # 由绿灯时间的优化格式转换成控制格式
        lookahead = len(split)
        split_output = split.copy()
        for i in range(lookahead):
            split_temp = np.array([int(g) for g in split_output[i]])
            split_temp[5] = split_temp[0:2].sum(0) - split_temp[4]
            split_temp[7] = split_temp[2:4].sum(0) - split_temp[6]
            split_output[i, :] = split_temp
        return split_output  # array(lookahead,8)

    def update(self, obs_data, tc_data):
        # 维护代理模型的状态tensor：obs_back, tc_back
        # tc: traffic controller
        # (frames:list,info) -> tenor (frames,C,H,W)
        obs = frame_process(obs_data,self.config)

        # obs_back更新
        self.obs_back[0, :-1] = self.obs_back[0, 1:]  # tensor
        self.obs_back[0, -1] = obs

        # tc_back更新
        control = torch.from_numpy(np.concatenate([tc_data["phase"], tc_data["split"]]))
        self.tc_back[0, :-1, :] = self.tc_back[0, 1:, :].clone()  # tensor
        self.tc_back[0, -1, :] = control

        if self.warm_up > 0:  # 更新足够次数后可以进行MPC控制
            self.warm_up -= 1

        if self.warm_up == 0:
            with torch.no_grad():
                self.upper_context = self.model.encoding(self.obs_back, self.tc_back)

    def generate_control(self, vph_m):
        for lookahead in range(self.lookahead, 0, -1):
            lookahead = 1  # bypass
            phases = self._multi_enumerations(lookahead)
            splits = self._multi_restarts(vph_m, lookahead)

            optimal_point, _ = self._optimize(phases, splits)  # (1,lookahead,8+5)

            with torch.no_grad():
                optimal_surrogate = self._predict(optimal_point)  # (c,mu,sigma)
                mean = mdn_mean(optimal_surrogate).sum()  # (1,lookahead,1)
            au = aleatoric_uncertainty(optimal_surrogate).sum()  # (1,lookahead,1)
            if norm(loc=mean, scale=np.sqrt(au)).cdf((1 + self.alpha) * mean) > 0.95 or True:  # bypass
                self.horizon_result.append(lookahead)
                break

        optimal_surrogate = (x[0] for x in optimal_surrogate)
        optimal_point = optimal_point[0]  # remove batch dim
        self.surrogate_result.append(optimal_surrogate)
        self.control_result.append(optimal_point)
        self.record_context()

        control = {}
        # 取第一个周期
        control["phase"] = optimal_point[0, :5].detach().numpy()  # array(5)
        control["split"] = optimal_point[0, 5:].detach().numpy()  # array(8)

        eu = epistemic_uncertainty(optimal_surrogate).sum()
        is_valid = (np.sqrt(eu) / mean < self.beta).item()
        self.valid_result.append(is_valid)
        # print(lookahead, is_valid)
        is_valid = True  # bypass

        return control, is_valid

    def record_context(self):
        self.context_result.append(self.upper_context)

    def _predict(self, tc_ahead):
        self.model.eval()
        y_pred = self.model.predict(self.upper_context, tc_ahead)
        return y_pred
