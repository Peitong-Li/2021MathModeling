# Author: 李沛橦
# Description：制作优化器

import torch

def make_optimizer(model, lr):
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    return optimizer