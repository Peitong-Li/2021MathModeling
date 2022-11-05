# Author: 李沛橦
# Description：可视化模型结构

from torchsummary import summary

from model.myNet import Net, NetLstm

model = Net(21,100,1)
# model = NetLstm(21,100,1)

summary(model,input_size=[(21,)], batch_size=256, device='cpu')
