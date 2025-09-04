"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
# 3d fourier layers
################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        # 保持傅里叶层结构不变（仅负责特征变换，不改变空间/时间维度）
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 傅里叶模式数（需 ≤ 64//2 + 1）
        self.modes2 = modes2  # 需 ≤ 64//2 + 1
        self.modes3 = modes3  # 需 ≤ 10//2 + 1（适配新的时间维度10）

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # 复数乘法：(B, in_c, x, y, t) × (in_c, out_c, x, y, t) → (B, out_c, x, y, t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])  # 傅里叶变换（保留批次和通道维度）

        # 仅对关键傅里叶模式进行操作（不改变维度大小）
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))  # 逆变换回物理空间（维度不变）
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()
        """
        输入形状：(B, 64, 64, 10, 1)  # (批次, x维度, y维度, t维度, 特征数)
        输出形状：(B, 64, 64, 10, 1)  # 与输入形状完全一致
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3  # 需 ≤ 10//2 + 1（适配t=10）
        self.width = width
        self.padding = 3  # 较小的padding避免维度膨胀（原6可能导致t维度过大）

        # 1. 输入特征映射：将输入特征数（1）+ 位置特征（3）映射到width维度
        self.fc0 = nn.Linear(5, self.width)  # 输入特征1 + grid的3个位置特征 → 总4维

        # 2. 傅里叶层和卷积层（保持通道数和空间/时间维度）
        self.conv0 = SpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)  # 1x1x1卷积（不改变空间维度）
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = nn.BatchNorm3d(self.width)  # 批归一化（稳定训练）
        self.bn1 = nn.BatchNorm3d(self.width)
        self.bn2 = nn.BatchNorm3d(self.width)

        # 3. 输出特征映射：从width维度映射回1个特征（与输入一致）
        self.fc1 = nn.Linear(self.width, 64)  # 中间压缩
        self.fc2 = nn.Linear(64, 1)  # 最终输出1个特征

    def forward(self, x,t):
        # 输入x形状：(B, 64, 64, 10, 1)
        batchsize = x.shape[0]
        # import ipdb ;ipdb.set_trace()
        # 生成位置特征（x, y, t坐标）并拼接：(B, 64, 64, 10, 1) → (B, 64, 64, 10, 4)
        grid = self.get_grid(x.shape, x.device)  # grid形状：(B, 64, 64, 10, 3)
        x = torch.cat((x, t,grid), dim=-1)  # 拼接后特征数：1 + 3 = 4

        # 输入映射：(B, 64, 64, 10, 4) → (B, 64, 64, 10, width)
        x = self.fc0(x)
        
        # 维度重排：(B, x, y, t, c) → (B, c, x, y, t)（适配3D卷积/傅里叶层）
        x = x.permute(0, 4, 1, 2, 3)  # 形状：(B, width, 64, 64, 10)
        
        #  padding：仅在时间维度补零（避免空间维度变形）
        x = F.pad(x, [0, self.padding])  # 形状：(B, width, 64, 64, 10+padding)

        # 傅里叶层 + 残差连接（保持维度）
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.gelu(x)  # 激活函数

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)  # 最后一层可不加激活，保留更多信息

        # 裁剪padding：恢复时间维度为10
        x = x[..., :-self.padding]  # 形状：(B, width, 64, 64, 10)
        
        # 维度重排：(B, c, x, y, t) → (B, x, y, t, c)（还原原始维度顺序）
        x = x.permute(0, 2, 3, 4, 1)  # 形状：(B, 64, 64, 10, width)

        # 输出映射：从width维度 → 1个特征
        x = self.fc1(x)  # (B, 64, 64, 10, 64)
        x = F.gelu(x)
        x = self.fc2(x)  # (B, 64, 64, 10, 1) → 与输入形状一致

        return x

    def get_grid(self, shape, device):
        # 生成x, y, t三个维度的坐标网格（范围0~1）
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        # x坐标：(1, 64, 1, 1, 1) → 重复为(B, 64, 64, 10, 1)
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1, 1).repeat(batchsize, 1, size_y, size_z, 1)
        # y坐标：(1, 1, 64, 1, 1) → 重复为(B, 64, 64, 10, 1)
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1, 1).repeat(batchsize, size_x, 1, size_z, 1)
        # t坐标：(1, 1, 1, 10, 1) → 重复为(B, 64, 64, 10, 1)
        gridz = torch.linspace(0, 1, size_z, device=device).reshape(1, 1, 1, size_z, 1).repeat(batchsize, size_x, size_y, 1, 1)
        return torch.cat((gridx, gridy, gridz), dim=-1)  # 合并为3个特征
    
# ################################################################
# # configs
# ################################################################

# TRAIN_PATH = '/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.matt'
# TEST_PATH = '/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat'

# ntrain = 1000
# ntest = 200

# modes = 8
# width = 20

# batch_size = 10
# batch_size2 = batch_size

# epochs = 500
# learning_rate = 0.001
# scheduler_step = 100
# scheduler_gamma = 0.5

# print(epochs, learning_rate, scheduler_step, scheduler_gamma)

# path = 'test'
# # path = 'ns_fourier_V100_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
# path_model = 'model/'+path
# path_train_err = 'results/'+path+'train.txt'
# path_test_err = 'results/'+path+'test.txt'
# path_image = 'image/'+path


# runtime = np.zeros(2, )
# t1 = default_timer()


# sub = 1
# S = 64 // sub
# T_in = 10
# T = 40

# ################################################################
# # load data
# ################################################################

# reader = MatReader(TRAIN_PATH)
# train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
# train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

# reader = MatReader(TEST_PATH)
# test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
# test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

# print(train_u.shape)
# print(test_u.shape)
# assert (S == train_u.shape[-2])
# assert (T == train_u.shape[-1])


# a_normalizer = UnitGaussianNormalizer(train_a)
# train_a = a_normalizer.encode(train_a)
# test_a = a_normalizer.encode(test_a)

# y_normalizer = UnitGaussianNormalizer(train_u)
# train_u = y_normalizer.encode(train_u)

# train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
# test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

# t2 = default_timer()

# print('preprocessing finished, time used:', t2-t1)
# device = torch.device('cuda')

# ################################################################
# # training and evaluation
# ################################################################
# model = FNO3d(modes, modes, modes, width).cuda()
# # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

# print(count_params(model))
# optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


# myloss = LpLoss(size_average=False)
# y_normalizer.cuda()
# for ep in range(epochs):
#     model.train()
#     t1 = default_timer()
#     train_mse = 0
#     train_l2 = 0
#     for x, y in train_loader:
#         x, y = x.cuda(), y.cuda()

#         optimizer.zero_grad()
#         out = model(x).view(batch_size, S, S, T)

#         mse = F.mse_loss(out, y, reduction='mean')
#         # mse.backward()

#         y = y_normalizer.decode(y)
#         out = y_normalizer.decode(out)
#         l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
#         l2.backward()

#         optimizer.step()
#         train_mse += mse.item()
#         train_l2 += l2.item()

#     scheduler.step()

#     model.eval()
#     test_l2 = 0.0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.cuda(), y.cuda()

#             out = model(x).view(batch_size, S, S, T)
#             out = y_normalizer.decode(out)
#             test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

#     train_mse /= len(train_loader)
#     train_l2 /= ntrain
#     test_l2 /= ntest

#     t2 = default_timer()
#     print(ep, t2-t1, train_mse, train_l2, test_l2)
# # torch.save(model, path_model)


# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0
#         x, y = x.cuda(), y.cuda()

#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out

#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})




