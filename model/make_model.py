# Author: 李沛橦
# Description： 构建的两种基础网络：baseNetwork（MLP），my_LSTM（LSTM），是6个分支的部件



from IPython import embed

import torch
import torch.nn as nn
import torch.nn.functional as F

class baseNetwork(nn.Module):
    def __init__(self, in_dim, n_output):
        super(baseNetwork, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_output),
        )

    def forward(self, x):
        x = self.base(x)
        return x


class my_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(my_LSTM, self).__init__()
        self.base = nn.LSTM(in_dim, hidden_dim, 2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 10),
            nn.Tanh(),
            nn.Linear(10, n_classes),
        )

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        out, (h_n, c_n) = self.base(x)
        # 此时可以从out中获得最终输出的状态h
        # x = out[:, -1, :]
        x = h_n[-1, :, :]
        x = self.classifier(x)
        return x
