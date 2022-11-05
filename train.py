# Author: 李沛橦
# Description：训练运行脚本


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
from utils.make_data import make_data
from model.myNet import Net, NetLstm


if __name__ == '__main__':

    # 加载数据
    data_numpy = make_data()
    X_train, X_test, Y_train, Y_test = train_test_split(data_numpy[0], data_numpy[1], test_size=0.3, random_state=42)
    # 数据标准化
    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    X_test_s = scale.transform(X_test)

    data_df = pd.DataFrame(data=X_train_s, columns=range(X_train_s.shape[1]))
    data_df['target1'] = Y_train[:,0]
    data_df['target2'] = Y_train[:,1]
    data_df['target3'] = Y_train[:,2]
    data_df['target4'] = Y_train[:,3]
    data_df['target5'] = Y_train[:,4]
    data_df['target6'] = Y_train[:,5]
    data_df.head()

    # 展示分布图
    def show_data_fenbu():
        colname = data_df.columns.values
        plt.figure(figsize=(20,15))
        for ii in range(len(colname)):
            plt.subplot(9, 3, ii+1)
            sns.kdeplot(data_df[colname[ii]], gridsize=100)
            plt.title(colname[ii])
        plt.subplots_adjust(hspace=0.15)
        plt.show()
    show_data_fenbu()

    # 求相关系数热图
    def show_heatmap():
        plt.figure(figsize=(8,6))
        ax = sns.heatmap(data_df.corr(), square=True, annot=True, fmt=".3f", linewidths=.5,
                          cbar_kws={"fraction": 0.046, "pad": 0.03})
        plt.show()
    show_heatmap()

    # 将Numpy数据转化为tensor
    train_x = torch.from_numpy(X_train_s.astype(np.float32))
    train_y = torch.from_numpy(Y_train.astype(np.float32))
    test_x = torch.from_numpy(X_test_s.astype(np.float32))
    test_y = torch.from_numpy(Y_test.astype(np.float32))

    train_data = Data.TensorDataset(train_x, train_y)
    test_data = Data.TensorDataset(test_x, test_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=1)
    # test_loader = Data.DataLoader(dataset=test_data, batch_size=256, shuffle=False, num_workers=4)

    def showDataloader():
        for batch_id, (b_x, b_y) in enumerate(train_loader):
            if batch_id > 0:
                break
        print("b_x.shape", b_x.shape)
        print("b_y.shape", b_y.shape)


    # 构建模型：两种方式
    # model = NetLstm(21, 32, 1)
    model = Net(21, 100, 1)
    model.train()
    Epoch = 30
    # 训练函数
    def train():
        # print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_func = make_loss("MSE")
        # loss_func = nn.MSELoss()
        train_loss_all = []
        for epo in range(Epoch):
            train_loss = 0
            train_num = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                pre = model(b_x)
                loss = loss_func(pre, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_num += b_x.size(0)
                if (step+1)%10 == 0:
                    print(f"Epoch[{epo}/{Epoch}], Batch[{step}/{len(train_loader)}], Loss:{loss.item()}")
            train_loss_all.append(train_loss/train_num)
        return train_loss_all

    # 保存并加载模型
    resume = ''
    if resume == '':
        train_loss_all = train()
        torch.save(model, './model.pth')
    else:
        model.load_state_dict(torch.load(resume))
        # train_loss_all = torch.load(resume)['train_loss_all']

    # 展示Loss曲线
    def show_loss_curve():
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_all, "ro-", label="Train loss")
        plt.legend()
        plt.grid()
        plt.xlabel("Epoch", size=13)
        plt.ylabel("Loss", size=13)
        plt.show()
    # show_loss_curve()

    # 测试集的预测
    def test_prediction():
        model.eval()
        pre_y = model(test_x)
        # pre_y = pre_y.detach().numpy() # (6个(128,1))
        a1 = np.column_stack((pre_y[0].detach().numpy()[:,0], pre_y[1].detach().numpy()[:,0]))
        a2 = np.column_stack((a1, pre_y[2].detach().numpy()[:,0]))
        a3 = np.column_stack((a2, pre_y[3].detach().numpy()[:,0]))
        a4 = np.column_stack((a3, pre_y[4].detach().numpy()[:,0]))
        pre_y = np.column_stack((a4, pre_y[5].detach().numpy()[:,0]))
        # 几种评价指标
        mae = mean_absolute_error(Y_test, pre_y)
        mse = mean_squared_error(Y_test, pre_y)
        r2 = r2_score(Y_test, pre_y)
        mae1 = mean_absolute_error(data_numpy[1], data_numpy[2])
        mse1 = mean_squared_error(data_numpy[1], data_numpy[2])
        r21 = r2_score(data_numpy[1], data_numpy[2])
        print("在测试集上的绝对值误差为(MAE)：", mae)
        print("在测试集上的均方误差为(MSE)：", mse)
        print("在测试集上的R平方值为(R2)：", r2)
        print("一次预报的绝对值误差为(MAE)：", mae1)
        print("一次预报的均方误差为(MSE)：", mse1)
        print("一次预报的R平方值为(R2)：", r21)
        return pre_y
    pre_y = test_prediction()

    # 展示数据真实值和预测值的差距
    def show_difference():
        for i in range(Y_test.shape[1]):
            index = np.argsort(Y_test[:, i])
            plt.figure(figsize=(12,5))
            plt.plot(np.arange(len(Y_test)), Y_test[index, i], "r", label="Original Y")
            plt.scatter(np.arange(pre_y.shape[0]), pre_y[index, i], s=3, c='b', label="Prediction")
            plt.legend(loc="upper left")
            plt.grid()
            plt.xlabel("Index")
            plt.ylabel("Y")
            plt.show()
    show_difference()

