# Author: 李沛橦
# Description：制作数据的脚本，具体内容请看每个Function


import math

import numpy as np
import torch
import time
import pandas as pd

from IPython import embed
from torch.utils.data import TensorDataset
from collections import Counter


global data_orgin
data_orgin = 0

# 计算点与点间的距离
def calc_dist_dict(pointA, pointB):
    dist = math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)
    return dist

# q4 make_fusion_dist_data
def make_fusion_dist_data():
    print('q4: make_fusion_dist_data')
    # url_pre = r'./data/q4/step5/fusion_pre_xA1A2A3.csv'
    url_pre = r'./data/q4/step7/data.csv'
    df_pre = pd.read_csv(url_pre)
    dict_place = {
        'A': (0,0),
        'A1':(-14.4846,-1.9699),
        'A2':(-6.6716,-7.5953),
        'A3':(-3.3543,-5.0138),
    }
    all_data = []
    for row in df_pre.itertuples():
        # 当前行时间
        current_row_time =  row[2]
        print("current: ", row[0])
        # 当前行监测点
        current_row_place = row[3]
        # 文件中所有与当前行时间相同的行
        dict_row = df_pre[df_pre['time'] == current_row_time].to_dict()
        # 文件中所有与当前行时间相同的行的索引号
        find_index = df_pre[df_pre['time'] == current_row_time].index.tolist()
        # 寻找到的行属于哪个监测点
        place_dict = dict_row['place']
        data_list = []
        if len(find_index) != 1:
            dist_list = []
            for i, same_time_index in enumerate(find_index):
                place_name = place_dict[same_time_index]
                dist_list.append(calc_dist_dict(dict_place[current_row_place], dict_place[place_name]))
            weight = [(sum(dist_list)-j)/(sum(dist_list)*2) for j in dist_list]
            data_list = []
            for i in range(len(weight)):
                data_list.append(df_pre.iloc[[find_index[i]]].iloc[:, 3:].values * weight[i])
        else:
            data_list.append(df_pre.iloc[[find_index[0]]].iloc[:, 3:].values)
        row_data_numpy = np.array(data_list).squeeze(axis=1)
        res_data = np.sum(row_data_numpy, axis=0)
        all_data.append(res_data)
    res_data = np.array(all_data)
    pd.DataFrame(res_data).to_csv('./data/q4/step7/test_data.csv')
    # pd.DataFrame(res_data).to_csv('./data/q4/step6/train_data.csv')

# step7：制作测试数据
def make_test_data():
    # url_data = r'./data/step7/data.csv'
    url_data = r'./data/q4/step7/test_data.csv'
    df = pd.read_csv(url_data, encoding='utf-8')
    return df.values, df


# step6: 制作dataloader
def make_dataloader():
    print('step6: make_dataloader')
    url_pre = r'./data/q4/step6/train_data.csv'
    url_real = r'./data/q4/step2/del_A1A2A32.csv'
    # url_pre = r'./data/step5/fusion_pre_ABC.csv'
    # url_real = r'./data/step2/del_ABC2.csv'
    df_pre = pd.read_csv(url_pre)
    df_real = pd.read_csv(url_real)
    # pre_x = torch.from_numpy(df_pre.iloc[:, 1:-6].values)
    # pre_y = torch.from_numpy(df_pre.iloc[:, -6:].values)
    # real_x = torch.from_numpy(df_real.iloc[:, -5:].values)
    # real_y = torch.from_numpy(df_real.iloc[:, 2:-5].values)
    pre_x = df_pre.iloc[:, 1:].values
    pre_y = df_pre.iloc[:, -6:].values
    real_x = df_real.iloc[:, -5:].values
    # real_x = df_real.iloc[:, -5:].values
    real_y = df_real.iloc[:, 3:-5].values
    # real_y = df_real.iloc[:, 3:-5].values
    # pd.DataFrame(pre_x).to_csv('./data/step6/pre_xA.csv.csv')
    # pd.DataFrame(pre_y).to_csv('./data/step6/pre_yA.csv.csv')
    # pd.DataFrame(real_x).to_csv('./data/step6/real_xA.csv.csv')
    # pd.DataFrame(real_y).to_csv('./data/step6/real_yA.csv.csv')
    return pre_x, pre_y, real_x, real_y

# step5: 将同一个一次预报时间的数据融合
def fusion_pre():
    print('step5: fusion_pre')
    weight = [0.7, 0.2, 0.1]
    url_real_x = r'./data/q4/step3/sheet2_xA32.csv'
    url_pre_x = r'./data/q4/step2/del_A31.csv'
    df_real_x = pd.read_csv(url_real_x)
    df_pre_x = pd.read_csv(url_pre_x)
    list_data = []

    for row in df_real_x.itertuples():
        print(row[0])
        dict_row = df_pre_x[df_pre_x['2'] == row[2]].to_dict()
        find_index = df_pre_x[df_pre_x['2'] == row[2]].index.tolist()
        list_values = [i for i in list(dict_row.values())]
        time_num = list_values[2]
        # today = row[2], 查看当前查找到的dict长度
        if len(time_num) == 1:
            data_forback = df_pre_x.iloc[[find_index[-1]]].iloc[:, 3:].values
            # 当前行等于当前行
        elif len(time_num) == 2:
            data1 = df_pre_x.iloc[[find_index[-1]]].iloc[:, 3:].values * weight[0]
            data2 = df_pre_x.iloc[[find_index[-2]]].iloc[:, 3:].values * (1-weight[0])
            data_forback = data1 + data2
            # 当前行*weight[0] + 上一行*1-weight[0]
        else:
            data1 = df_pre_x.iloc[[find_index[-1]]].iloc[:, 3:].values * weight[0]
            data2 = df_pre_x.iloc[[find_index[-2]]].iloc[:, 3:].values * weight[1]
            data3 = df_pre_x.iloc[[find_index[-2]]].iloc[:, 3:].values * weight[2]
            data_forback = data1 + data2 + data3
            # 当前行*weight[0] + 上一行*weight[1] + 再上一行*weight[2]
        list_data.append(data_forback)
    data_numpy = np.array(list_data).squeeze(axis=1)
    df = pd.DataFrame(data_numpy)
    df.to_csv('./data/q4/step5/fusion_pre_xA3.csv')

# step4: 把真实值扩充成预测值大小
def process_real_xy():
    print('step4: process_real_xy')
    url_pre_x = r'../data/sheet1_xA1.csv'
    url_pre_y = r'../data/sheet1_yA1.csv'
    url_real_x = r'../data/sheet2_xA2.csv'
    url_real_y = r'../data/sheet2_yA2.csv'
    df_pre_x = pd.read_csv(url_pre_x)
    df_real_x = pd.read_csv(url_real_x)
    df_real_y = pd.read_csv(url_real_y)
    df_feat_real_x = pd.read_csv('../data/feat_real_xC.csv')
    df_feat_real_y = pd.read_csv('../data/feat_real_yC.csv')

    for row in df_pre_x.itertuples():
        print(f"{row[0]}/{df_pre_x.shape[0]}")
        dict_row = df_real_x[df_real_x['1'] == row[3]].to_dict()
        find_index = df_real_x[df_real_x['1'] == row[3]].index.tolist()
        list_values = [i[find_index[0]] for i in list(dict_row.values())]
        df_feat_real_x.loc[str(row[1])] = list_values

        dict_row = df_real_y[df_real_y['1'] == row[3]].to_dict()
        list_values = [i[find_index[0]] for i in list(dict_row.values())]
        df_feat_real_y.loc[str(row[1])] = list_values
    df_feat_real_x.to_csv('../data/feat_real_xC.csv')
    df_feat_real_y.to_csv('../data/feat_real_yC.csv')

# step3：将删除后的数据分为pre_x,pre_y,real_x,real_y
def get_data():
    print('step3: get_data')
    url_x = r'E:\Workspace\Learning\MathModeling21\data\q4\step2\del_A31.csv'
    url_y = r'E:\Workspace\Learning\MathModeling21\data\q4\step2\del_A32.csv'
    df_x = pd.read_csv(url_x, encoding='utf-8')
    df_y = pd.read_csv(url_y, encoding='utf-8')
    sheet1_x = df_x.iloc[:,1:-6]
    sheet1_y = df_x.iloc[:,[ 1, 2, -6, -5,-4,-3,-2,-1]]
    sheet2_x = df_y.iloc[:, [1,-4,-3,-2,-1]]
    sheet2_y = df_y.iloc[:, 1:-4]
    counter1 = Counter(sheet1_x['2'])
    counter2 = Counter(sheet2_x['1'])
    print(len(counter1))
    print(len(counter2))
    sheet1_x.to_csv('./data/q4/step3/sheet1_xA31.csv')
    sheet1_y.to_csv('./data/q4/step3/sheet1_yA31.csv')
    sheet2_x.to_csv('./data/q4/step3/sheet2_xA32.csv')
    sheet2_y.to_csv('./data/q4/step3/sheet2_yA32.csv')

# step2: 删除数据表时间相互不存在的时间
def del_label():
    print('step2: del_label')
    url_x = r'E:\Workspace\Learning\MathModeling21\data\q4\fileA31.csv'
    url_y = r'E:\Workspace\Learning\MathModeling21\data\q4\fileA32.csv'
    df_x = pd.read_csv(url_x, encoding='utf-8')
    df_y = pd.read_csv(url_y, encoding='utf-8')
    df_x_time = df_x['2'].values
    df_y_time = df_y['1'].values
    delList = []
    for row in df_y.itertuples():
        if not row[2] in df_x_time:
            delList.append(row[0])
    df_y = df_y.drop(delList)
    print(f"y标签删除了：{len(delList)}行")
    print(f"Yshape: {df_y.shape}")

    delList = []
    for row in df_x.itertuples():
        if not row[3] in df_y_time:
            delList.append(row[0])
    df_x = df_x.drop(delList)
    print(f"x标签删除了：{len(delList)}行")
    print(f"Xshape: {df_x.shape}")

    df_x = df_x.drop(['0'], axis=1)
    df_y = df_y.drop(['0'], axis=1)
    df_x.to_csv('./data/q4/step2/del_A31.csv')
    df_y.to_csv('./data/q4/step2/del_A32.csv')

# step1： 将时间修改一致
def change_time():
    print('step1: change_time')
    url_x = r'E:\Workspace\Learning\MathModeling21\data\q4\step1\fileA31.xlsx'
    url_y = r'E:\Workspace\Learning\MathModeling21\data\q4\step1\fileA32.xlsx'
    df_x = pd.read_excel(url_x)
    df_y = pd.read_excel(url_y)
    d_index = list(df_x.columns).index('预测时间')
    d_index1 = list(df_y.columns).index('监测时间')
    for row in df_x.itertuples():
        timeA = (str(row[2]).split()[0] + ' ' + str(int(str(row[2]).split()[1].split(':')[0]))).replace(r'/','-').replace("-0", "-")
        df_x.iloc[row[0], d_index] = timeA

    for row in df_y.itertuples():
        timeA = (str(row[2]).split(":")[0].split()[0] + ' ' + str(int(str(row[2]).split(":")[0].split()[-1]))).replace("-0", "-")
        df_y.iloc[row[0], d_index1] = timeA
    df_x.to_csv('./data/q4/fileA31.csv')
    df_y.to_csv('./data/q4/fileA32.csv')

# main()
def read_csv_get_data():
    # step1: change_time()
    # step2: del_label()
    # step3: get_data()
    # step4: process_real_xy()
    # step5: fusion_pre()
    # step6: make_dataloader()
    pre_x, pre_y, real_x, real_y = make_dataloader()
    return pre_x, pre_y, real_x, real_y

# MinMaxScaler:反归一化
def denormalization(x):
    x = x.detach().numpy()
    return torch.from_numpy(x*(np.max(data_orgin) - np.min(data_orgin)) + np.min(data_orgin))

# MinMaxScaler:归一化
def normalization(data1):
    _range = np.max(data1) - np.min(data1)
    return torch.from_numpy((data1 - np.min(data1)) / _range)

# 训练、测试调用的接口
def make_data():
    x, pre_y, real_x, y = read_csv_get_data()
    return x, y, pre_y

# file转tensor
def file2tensor(dataurl):
    df = pd.read_excel(dataurl, sheet_name=3)
    head_name = df.keys()
    for i in range(len(df[head_name[0]])):
        df[head_name[0]][i] = str(df[head_name[0]][i])
    data = df.values[:, 2:]
    data = torch.from_numpy(data.astype(float))
    return data
