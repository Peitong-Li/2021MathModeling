# Author: 李沛橦
# Description：制作损失函数，此处放入加权分数算法，将计算出来的预测值重新加权

import torch
from IPython import embed
def make_loss(loss_name="cross_entropy"):
    if loss_name == "MSE":
        # MSE损失函数
        criterion = torch.nn.MSELoss()
    else:
        # CrossEntropyLoss
        criterion = torch.nn.CrossEntropyLoss()
    def loss_function(score, target):

        # 重新加权预测分数
        target_list_sum = [target[:, i].sum()/target.shape[0] for i in range(target.shape[1])]
        target_weight = [i/sum(list(target_list_sum)) for i in target_list_sum]
        # sum_leiji = [(1-i) for i in target_weight]
        # target_weight_against = [i/sum(sum_leiji) for i in sum_leiji]
        # 计算各分支loss
        if isinstance(score, list):
            pre_loss = [criterion(sco, target[:, i].unsqueeze(dim=1)) for i, sco in enumerate(score)]
            sum_loss = 0
            sum_loss_list = [target_weight[0].item() * loss_item for i, loss_item in enumerate(pre_loss)]
            for i in sum_loss_list:
                sum_loss += i
            sum_loss = sum_loss / len(sum_loss_list)
        else:
            pre_loss = criterion(score, target)
        return sum_loss

    return loss_function
