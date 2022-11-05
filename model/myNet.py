# Author: 李沛橦
# Description： 构建对于问题三和问题四的模型，共分为两种：
# 1、Net：MLP结构，一个主分支+6个分支
# 2、NetLstm：LSTM结构，一个主分支+6个分支


import torch
import torch.nn as nn
from IPython import embed
import torch.nn.functional as F

from .make_model import baseNetwork,my_LSTM

class Net(nn.Module):
    def __init__(self, in_chanl, n_hidden, n_output):
        super(Net, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(in_chanl, n_hidden),
            nn.ReLU(),
        )
        self.base1 = baseNetwork(n_hidden, n_output)
        self.base2 = baseNetwork(n_hidden, n_output)
        self.base3 = baseNetwork(n_hidden, n_output)
        self.base4 = baseNetwork(n_hidden, n_output)
        self.base5 = baseNetwork(n_hidden, n_output)
        self.base6 = baseNetwork(n_hidden, n_output)

    def forward(self, x):
        res = self.base(x)
        loss1 = self.base1(res)
        loss2 = self.base2(res)
        loss3 = self.base3(res)
        loss4 = self.base4(res)
        loss5 = self.base5(res)
        loss6 = self.base6(res)
        loss_list = []
        loss_list.append(loss1)
        loss_list.append(loss2)
        loss_list.append(loss3)
        loss_list.append(loss4)
        loss_list.append(loss5)
        loss_list.append(loss6)
        return loss_list



class NetLstm(nn.Module):
    # 21, 100, 1
    def __init__(self, in_chanl, n_hidden, n_output):
        super(NetLstm, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(in_chanl, n_hidden),
            nn.ReLU(),
        )
        self.base1 = my_LSTM(n_hidden, 50, n_output)
        self.base2 = my_LSTM(n_hidden, 50, n_output)
        self.base3 = my_LSTM(n_hidden, 50, n_output)
        self.base4 = my_LSTM(n_hidden, 50, n_output)
        self.base5 = my_LSTM(n_hidden, 50, n_output)
        self.base6 = my_LSTM(n_hidden, 50, n_output)

    def forward(self, x):
        res = self.base(x)
        loss1 = self.base1(res)
        loss2 = self.base2(res)
        loss3 = self.base3(res)
        loss4 = self.base4(res)
        loss5 = self.base5(res)
        loss6 = self.base6(res)
        loss_list = []
        loss_list.append(loss1)
        loss_list.append(loss2)
        loss_list.append(loss3)
        loss_list.append(loss4)
        loss_list.append(loss5)
        loss_list.append(loss6)
        return loss_list



