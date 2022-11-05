# Author: 李沛橦
# Description：对测试集的预测

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython import embed
import matplotlib.pyplot as plt

import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

from utils.make_loss import make_loss
from utils.make_data import make_data, make_test_data
from model.myNet import Net, NetLstm


if __name__ == '__main__':
    test_data, df = make_test_data()
    scale = StandardScaler()
    scale.fit_transform(test_data)
    X_test_s = scale.transform(test_data)
    test_x = torch.from_numpy(X_test_s.astype(np.float32))

    # 展示分布图
    def show_data_fenbu():
        colname = df.columns.values
        plt.figure(figsize=(20,15))
        for ii in range(len(colname)):
            plt.subplot(7, 3, ii+1)
            sns.kdeplot(df[colname[ii]], gridsize=100)
            plt.title(colname[ii])
        plt.subplots_adjust(hspace=0.15)
        plt.show()
    show_data_fenbu()

    # 求相关系数热图
    def show_heatmap():
        plt.figure(figsize=(8,6))
        ax = sns.heatmap(df.corr(), square=True, annot=True, fmt=".3f", linewidths=.5,
                          cbar_kws={"fraction": 0.046, "pad": 0.03})
        plt.show()
    show_heatmap()

    # model = NetLstm(15, 32, 1)
    model = Net(21, 100, 1)
    model.eval()
    resume = './model.pth'

    model.load_state_dict(torch.load(resume)['state_dict'])
    train_loss_all = torch.load(resume)['train_loss_all']

    def test_prediction():
        model.eval()
        pre_y = model(test_x)
        a1 = np.column_stack((pre_y[0].detach().numpy()[:, 0], pre_y[1].detach().numpy()[:, 0]))
        a2 = np.column_stack((a1, pre_y[2].detach().numpy()[:, 0]))
        a3 = np.column_stack((a2, pre_y[3].detach().numpy()[:, 0]))
        a4 = np.column_stack((a3, pre_y[4].detach().numpy()[:, 0]))
        pre_y = np.column_stack((a4, pre_y[5].detach().numpy()[:, 0]))
        return pre_y
    pre_y = test_prediction()
    # pd.DataFrame(pre_y).to_csv('./data/step7/res_q3.csv')
    pd.DataFrame(pre_y).to_csv('./data/q4/step7/res_q4.csv')



