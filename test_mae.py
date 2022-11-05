# Author: 李沛橦
# Description：对测试集的预测结果的评价，利用已有的前8个小时的实测值

import pandas as pd
from IPython import embed
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

url = r'./data/step7/data_8.csv'
df = pd.read_csv(url)
real_data = df.iloc[0:8, :].values
pre_data = df.iloc[8:, :].values
mae = mean_absolute_error(real_data, pre_data)
mse = mean_squared_error(real_data, pre_data)
