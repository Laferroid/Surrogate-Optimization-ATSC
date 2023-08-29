#%%
# region
import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import norm
from pynverse import inversefunc
# endregion

#%%
# 模型结构
class ReversibleBatchNorm(nn.Module):
    def __init__(self,num_features):
        # 可逆的归一化层
        super(ReversibleBatchNorm,self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(self.num_features,affine=False)  # 沿着第2维度进行归一化，即对第2维度以外的维度求期望和标准差
        
    def forward(self,x):
        return self.bn(x)
    
    def backward(self,mu,sigma):
        # x: (batch_size,lookahead,num_componnets,output_size)
        lookahead = mu.shape[1]
        # self.bn.running_mean: tensor: (lookahead)
        # self.bn.running_var: tensor: (lookahead)
        running_mean = self.bn.running_mean[None,:lookahead,None,None]
        running_std = torch.sqrt(self.bn.running_var+self.bn.eps)[None,:lookahead,None,None]
        mu = mu*running_std + running_mean
        sigma = sigma*running_std
        return (mu,sigma)

class SequenceBatchNorm(nn.Module):
    def __init__(self,num_features,affine=True):
        # 沿-1维度归一化
        super(SequenceBatchNorm,self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(self.num_features,affine=affine)  # 沿着第2维度进行归一化，即对第2维度意外的以外求期望和标准差
    
    def forward(self,x):
        if x.dim() == 2:  # (batch_size,num_features)
            return self.bn(x)
        elif x.dim() == 3:  # (batch_size,lookahead,num_features)
            shp = x.shape
            return self.bn(x.reshape(-1,self.num_features)).reshape(*shp)

class LinearResBlock(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,is_shortcut=True):
        super(LinearResBlock,self).__init__()
        self.is_shortcut = is_shortcut
        # 更换不同的激活函数
        # self.mainline = nn.Sequential(nn.Linear(input_size,hidden_size),SequenceBatchNorm(hidden_size),nn.ReLU(),
        #                               nn.Linear(hidden_size,hidden_size),SequenceBatchNorm(hidden_size),nn.ReLU(),
        #                               nn.Linear(hidden_size,output_size),SequenceBatchNorm(output_size),nn.ReLU())
        # self.mainline = nn.Sequential(nn.Linear(input_size,hidden_size),nn.Tanh(),
        #                               nn.Linear(hidden_size,hidden_size),nn.Tanh(),
        #                               nn.Linear(hidden_size,output_size),nn.Tanh())
        self.mainline = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),
                                      nn.Linear(hidden_size,hidden_size),nn.ReLU())
        if self.is_shortcut:
            self.shortcut = nn.Linear(input_size,output_size)
        
    def forward(self,x):
        y = self.mainline(x)
        if self.is_shortcut:
            y = y + self.shortcut(x)  # +=是inplace操作，反向传播时会出错
        return y
    
class CoefficientLayer(nn.Module):
    def __init__(self):
        super(CoefficientLayer,self).__init__()
  
    def forward(self,x):
        mode = 'softmax'
        if mode=='softmax':
            return F.softmax(x,dim=-1)
        elif mode=='elu+1':
            x = F.elu(x)+1.0
            return x/x.sum(-1,keepdim=True)
        elif mode=='relu':
            x = F.relu(x)
            return x/(x.sum(-1,keepdim=True)+1e-4)

class SigmaLayer(nn.Module):
    def __init__(self):
        super(SigmaLayer,self).__init__()
    def forward(self,x):
        mode = 'exp'
        if mode=='elu+1':
            return F.elu(x)+1.0
        elif mode=='exp':
            return torch.exp(x)

class MixtureDensityNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_componets):
        super(MixtureDensityNetwork,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_componets = num_componets
        
        self.block_1 = LinearResBlock(self.input_size,self.hidden_size,self.hidden_size)
        self.block_2 = LinearResBlock(self.hidden_size,self.hidden_size,self.hidden_size)
        
        self.c_layer = nn.Sequential(nn.Linear(self.hidden_size,self.num_componets),CoefficientLayer())
        
        self.mu_layer = nn.Sequential(nn.Linear(self.hidden_size,self.num_componets*self.output_size))
        
        self.sigma_layer = nn.Sequential(nn.Linear(self.hidden_size,self.num_componets*self.output_size),SigmaLayer())
        
    def forward(self,x):
        """_summary_

        Parameters
        ----------
        x : Tensor
            (*,input_size) 

        Returns
        -------
        c : Tensor
            (*,num_components)
        mu : Tensor
            (*,num_components,output_size)
        sigma : Tensor
            (*,num_components,output_size)
        """
        # x: (*,input_size)
        x = self.block_1(self.block_1(x))
        c = self.c_layer(x).unsqueeze(dim=-1)
        mu = torch.stack(self.mu_layer(x).chunk(self.output_size,dim=-1),dim=-1)
        sigma = torch.stack(self.sigma_layer(x).chunk(self.output_size,dim=-1),dim=-1)
        
        return c,mu,sigma
    
    def loss_func(self,y_pred,y):
        """_summary_

        Parameters
        ----------
        y_pred : tuple
            (c,mu,sigma)
            c : Tensor
                (*,num_components,1)
            mu : Tensor
                (*,num_components,output_size)
            sigma : Tensor
                (*,num_components,output_size)
        y : Tensor
            (*,output_size)

        Returns
        -------
        nll : scalar tensor
            negative log likelihood
        """
        c,mu,sigma = y_pred
        device = 'cuda:0' if y.is_cuda else 'cpu'
        
        # region 各种分布通用但是计算缓慢的做法
        # https://pytorch.org/docs/stable/generated/torch.diag_embed.html
        # https://pytorch.org/docs/stable/generated/torch.diagonal.html#torch.diagonal
        # components = D.MultivariateNormal(mu,torch.diag_embed(sigma))
        # c_log_prob = components.log_prob(y)
        # endregion
        
        # 使用分布表达式直接计算对数概率，快得多
        c_log_prob_1 = - (self.output_size/2)*torch.log(torch.tensor(2*math.pi,device=device))*torch.ones_like(c,device=device) 
        c_log_prob_2 = - torch.log(torch.prod(sigma,dim=-1,keepdim=True))
        c_log_prob_3 = (-1/2*(y.unsqueeze(-2)-mu)**2*sigma**(-2))
        c_log_prob = c_log_prob_1 + c_log_prob_2 + c_log_prob_3
        nll = - torch.logsumexp(c_log_prob+torch.log(c+1e-4),dim=-2).mean()
        return nll
    
    @torch.no_grad()
    def predict(self,x):
        # 不记录梯度的前向传播
        return self.forward(x)

# 下游计算任务
def aleatoric_uncertainty(y_pred):
    # 计算并输出偶然不确定性，numpy格式的输入
    c,_,sigma = y_pred
    # (~,num_components,1)
    # (~,num_components,output_size)
    return (c*sigma**2).sum(-2)

def epistemic_uncertainty(y_pred):
    # 计算并输出认知不确定性，numpy格式的输入
    c,mu,_ = y_pred
    # (~,num_components,1)
    # (~,num_components,output_size)
    return (c*(mu-(c*mu).sum(-2,keepdims=True))**2).sum(-2)

def variance(y_pred):
    return aleatoric_uncertainty(y_pred)+epistemic_uncertainty(y_pred)

def mdn_mean(y_pred):
    c,mu,_ = y_pred
    # (~,num_components,1)
    # (~,num_components,output_size)
    return (c*mu).sum(-2)  # (~,output_size)

def prediction_interval(y_pred,z):
    c,mu,sigma = y_pred
    cdf = lambda x: (c*norm(loc=mu,scale=sigma).cdf(np.expand_dims(x,0))).sum(0)
    ub = inversefunc(cdf,0.5+z/2,domain=[500.0,18000.0])
    lb = inversefunc(cdf,0.5-z/2,domain=[500.0,18000.0])
    return lb.squeeze(),ub.squeeze()

def quantile(y_pred,q):
    c,mu,sigma = y_pred
    cdf = lambda x: (c*norm(loc=mu,scale=sigma).cdf(np.expand_dims(x,0))).sum(0)
    qp = inversefunc(cdf,q,domain=[500.0,16000.0])
    return qp