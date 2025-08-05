import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess
from script.plot import plot_2d_results, plot_results, hinton, plot_3d_compare_with_diff
from script.ode_data import Eq1, WaveEquation, PoissonEquation, HeatEquation
from rectified.rectified_flow import RectFlow
import time
import matplotlib.pyplot as plt
from script.dataset import MyDataset_ns, BurgersDataset, darcyDataset
from torch.utils.data import DataLoader
# from scorenet.scorenet import  MLP2d_Darcy, CNN_add, CNN_ns,  MLP2d_Darcy,GNN_Darcy
from scorenet.FNO3d import FNO3d
import argparse
import logging
from Adam import Adam
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer



torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True


def setup_logger(save_path):
    """配置日志记录器"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_path, 'training.log')),
            logging.StreamHandler()
        ]
    )
def calculate_metrics(pred, true):
    with torch.no_grad():
        # 只计算L2相对误差（用于评估）
        lp_loss = LpLoss(p=2, reduction='mean')
        l2_rel = lp_loss(pred, true, mode='rel')
        return l2_rel.item()
def squared_absolute_error_loss(output, target):
    return torch.mean((torch.abs(output - target)) ** 2)
################################################################
# configs
################################################################
def main(config):
    # 动态设置GPU，优先使用GPU 1，不可用则使用GPU 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("No GPU available, using CPU mode")
    
    save_path = config['save_path']
    para_path = config['para_path']
    setup_logger(save_path)
    logging.info(f"Training device: {device}")
    
    eq_T = config['eq_T']
    N = config['N']
    niter = config['niter']
    lr = config['lr']
    batch_size = config['batch_size']
    eq_dt = config['eq_dt']
    rf_dt = config['rf_dt']
    h_dim = config['h_dim']
    train = config['train']
    scorenet_model_class = config['scorenet_model_class']
    model_name = config['model_name']
    rf = config['rf']
    eq = config['eq']
    # target_size=config['target_size']
    TRAIN_PATH = '/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat'
    TEST_PATH = '/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat'

    ntrain = 1000
    ntest = 200

    modes = 8
    width = 20

    batch_size = 10
    batch_size2 = batch_size

    epochs = 500
    learning_rate = lr
    scheduler_step = 100
    scheduler_gamma = 0.5

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)


    runtime = np.zeros(2, )
    t1 = default_timer()


    sub = 1
    S = 64 // sub
    T_in = 10
    T = 40

    ################################################################
    # load data
    ################################################################

    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
    train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
    test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

    print(train_u.shape)
    print(test_u.shape)
    assert (S == train_u.shape[-2])
    assert (T == train_u.shape[-1])


    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

    train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
    test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    t2 = default_timer()

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO3d(modes, modes, modes, width).cuda()
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

    print(count_params(model))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            t = torch.rand(batch_size, 1, device=device).to(dtype=torch.float32)
            t = t.view(batch_size, *([1] * (len(x.shape) - 1)))
            t = t.repeat(1, S,S,1,T,T_in)
            optimizer.zero_grad()
            out = model(x,t).view(batch_size, S, S, T)
            # mse.backward()

            y = y_normalizer.decode(y)
            x = a_normalizer.decode(x)
            out = y_normalizer.decode(out)
            l2 = myloss(out.view(batch_size, -1), (y-x).view(batch_size, -1))
            l2.backward()

            optimizer.step()
            train_l2 += l2.item()

        scheduler.step()
        if ep//5 == 0:
            model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    for t_val in np.arange(0.0, 1, rf_dt):
                        t = torch.ones(batch_size, 1, device=device).to(dtype=torch.float32)
                        t = t.view(batch_size, *([1] * (len(x.shape) - 1)))
                        t = t.repeat(1, S,S,1,T,T_in)
                        score = model(xt[-1], t_tensor)
                        xt_ = rf.forward_process(xt[-1], score, dt=rf_dt)
                        xt.append(xt_)
                    out = y_normalizer.decode(xt[-1].view(batch_size, S, S, T))
                    test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

            train_l2 /= ntrain
            test_l2 /= ntest

            t2 = default_timer()
            print(ep, t2-t1, train_l2, test_l2)
    torch.save(score_net.state_dict(), f"{para_path}{model_name}")


    pred = torch.zeros(test_u.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.cuda(), y.cuda()
            for t_val in np.arange(0.0, 1, rf_dt):
                t = torch.ones(batch_size, 1, device=device).to(dtype=torch.float32)
                t = t.view(batch_size, *([1] * (len(x.shape) - 1)))
                t = t.repeat(1, S,S,1,T,T_in)
                score = model(xt[-1], t_tensor)
                xt_ = rf.forward_process(xt[-1], score, dt=rf_dt)
                xt.append(xt_)
            out = y_normalizer.decode(xt[-1].view(batch_size, S, S, T))
            pred[index] = out

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            print(index, test_l2)
            index = index + 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    config = {}
    with open(args.config, 'r') as f:
        exec(f.read(), config)
    
    main(config)


