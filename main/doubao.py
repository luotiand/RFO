import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess
from script.plot import plot_2d_results
import matplotlib.pyplot as plt
from script.ode_data import Eq1, WaveEquation, PoissonEquation, HeatEquation
from rectified.rectified_flow import RectFlow
import time
from script.dataset import MyDataset_ns, BurgersDataset, darcyDataset
from torch.utils.data import DataLoader
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


# 绘制训练/测试损失曲线（训练连续+测试5个点）
def plot_loss_curves(train_losses, test_losses, test_epochs, save_path, model_class, target_len):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train L2 Loss')
    ax.plot(test_epochs, test_losses, 'ro-', label='Test L2 Loss (5 points)')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("L2 Loss", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Training and Test Loss Curves", fontsize=14)
    ax.legend()
    ax.grid(True)
    plt.savefig(f"{save_path}{model_class.lower()}_{target_len}_loss_curves.png", dpi=300)
    plt.close()


################################################################
# 主函数
################################################################
def main(config):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("No GPU available, using CPU mode")
    
    save_path = config['save_path']
    para_path = config['para_path']
    setup_logger(save_path)
    logging.info(f"Training device: {device}")
    
    # 核心参数
    niter = config['niter']
    lr = config['lr']
    batch_size = config['batch_size']
    rf_dt = config['rf_dt']
    train = config['train']  # 训练控制标志
    scorenet_model_class = config['scorenet_model_class']
    model_name = config['model_name']
    rf = config['rf']
    target_len = T = 10  # 目标时间步长
    
    # 数据路径与参数
    TRAIN_PATH = '/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat'
    TEST_PATH = '/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat'
    ntrain = 1000
    ntest = 20
    modes = 4
    width = 20
    epochs = niter
    learning_rate = lr
    scheduler_step = 100
    scheduler_gamma = 0.5

    # 数据维度
    sub = 1
    S = 64 // sub
    T_in = 10
    T = 10

    ################################################################
    # 加载数据并标准化（训练和推理共用）
    ################################################################
    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in].cuda()
    train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in].cuda()

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in].cuda()
    test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in].cuda()

    # 标准化器（用训练集参数）
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a_norm = a_normalizer.encode(train_a)
    test_a_norm = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u_norm = y_normalizer.encode(train_u)
    test_u_norm = y_normalizer.encode(test_u)
    
    # 调整形状（添加通道维度）
    train_a_norm = train_a_norm.reshape(ntrain, S, S, 1, T_in)
    test_a_norm = test_a_norm.reshape(ntest, S, S, 1, T_in)
    train_u_norm = train_u_norm.reshape(ntrain, S, S, 1, T_in)
    test_u_norm = test_u_norm.reshape(ntest, S, S, 1, T_in)

    # 数据加载器
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_a_norm, train_u_norm), 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_a_norm, test_u_norm), 
        batch_size=batch_size, 
        shuffle=False
    )

    ################################################################
    # 模型初始化（训练和推理共用）
    ################################################################
    model = FNO3d(modes, modes, modes, width).cuda()
    logging.info(f"模型参数数量: {count_params(model)}")
    
    # 优化器仅在训练时需要
    if train:
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_step, 
            gamma=scheduler_gamma
        )
    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    a_normalizer.cuda()

    ################################################################
    # 训练流程（仅当train=True时执行）
    ################################################################
    if train:
        train_losses = []
        test_losses = []
        test_epochs = []
        eval_epochs = [int(epochs * i / 4) for i in range(5)]  # 5个测试点
        
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_l2 = 0
            
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                current_bs = x.shape[0]

                # 生成时间张量
                t = torch.rand(current_bs, 1, device=device).to(dtype=torch.float32)
                t = t.view(current_bs, *([1] * (len(x.shape) - 1)))
                t = t.repeat(1, S, S, 1, T)
                
                optimizer.zero_grad()
                xt = rf.straight_process(x, y, t)
                out = model(xt, t).view(current_bs, S, S, T)
                
                # 计算训练损失
                l2 = myloss(out.view(current_bs, -1), (y - x).view(current_bs, -1))
                l2.backward()
                optimizer.step()
                train_l2 += l2.item()
            
            # 记录训练损失
            avg_train_loss = train_l2 / ntrain
            train_losses.append(avg_train_loss)
            scheduler.step()
            
            # 5个测试点评估
            if ep in eval_epochs:
                model.eval()
                test_l2 = 0.0
                with torch.no_grad():
                    for x, y in test_loader:
                        current_bs = x.shape[0]
                        xt = [x.cuda()]
                        
                        # 正向推理（训练中的测试）
                        for t_val in np.arange(0.0, 1, rf_dt):
                            t = t_val * torch.ones(current_bs, 1, device=device).to(dtype=torch.float32)
                            t = t.view(current_bs, *([1] * (len(x.shape) - 1)))
                            t = t.repeat(1, S, S, 1, T)
                            score = model(xt[-1], t)
                            xt_ = rf.forward_process(xt[-1], score, dt=rf_dt)
                            xt.append(xt_)
                        
                        # 计算测试损失
                        y_decoded = y_normalizer.decode(y.cuda().squeeze(-2))
                        xt_last = xt[-1].view(current_bs, S, S, T)
                        out_decoded = y_normalizer.decode(xt_last)
                        test_l2 += myloss(out_decoded.view(current_bs, -1), y_decoded.view(current_bs, -1)).item()
                
                avg_test_loss = test_l2 / ntest
                test_losses.append(avg_test_loss)
                test_epochs.append(ep + 1)
                logging.info(f"测试点 Epoch {ep+1} - Test Loss: {avg_test_loss:.6f}")
            
            # 打印训练进度
            if (ep + 1) % 50 == 0:
                logging.info(f"Epoch {ep+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
        
        # 保存模型
        torch.save(model.state_dict(), f"{para_path}{model_name}")
        logging.info(f"模型已保存至: {para_path}{model_name}")

        # 绘制损失曲线
        plot_loss_curves(
            train_losses, 
            test_losses, 
            test_epochs, 
            save_path, 
            scorenet_model_class, 
            target_len
        )
    
    ################################################################
    # 推理流程（无论是否训练，最后都执行：正向+反向）
    ################################################################
    # 加载模型（如果是推理模式，必须加载预训练模型）
    if not train:
        model.load_state_dict(torch.load(f"{para_path}{model_name}", map_location=device))
        logging.info(f"已加载预训练模型: {para_path}{model_name}")
    model.eval()
    logging.info("开始推理（正向+反向）...")
    
    with torch.no_grad():
        # 正向推理：从初始状态a到目标状态u
        xt_forward = [test_a_norm.cuda()]  # 初始状态（标准化）
        for t_val in np.arange(0.0, 1, rf_dt):
            current_bs = xt_forward[-1].shape[0]
            t = t_val * torch.ones(current_bs, 1, device=device).to(dtype=torch.float32)
            t = t.view(current_bs, *([1] * (len(xt_forward[-1].shape) - 1)))
            t = t.repeat(1, S, S, 1, T)
            score = model(xt_forward[-1], t)
            xt_ = rf.forward_process(xt_forward[-1], score, dt=rf_dt)
            xt_forward.append(xt_)
        
        # 反向推理：从目标状态u到初始状态a
        xt_reverse = [test_u_norm.cuda()]  # 目标状态（标准化）
        for t_val in np.arange(0.0, 1, rf_dt):
            current_bs = xt_reverse[-1].shape[0]
            # 反向推理时间为1-t（从目标到初始）
            t = t_val * torch.ones(current_bs, 1, device=device).to(dtype=torch.float32)
            t = t.view(current_bs, *([1] * (len(xt_reverse[-1].shape) - 1)))
            t = t.repeat(1, S, S, 1, T)
            score = model(xt_reverse[-1], 1 - t)  # 关键：反向时间用1-t
            xt_ = rf.reverse_process(xt_reverse[-1], score, dt=rf_dt)
            xt_reverse.append(xt_)
        
        ################################################################
        # 结果解码与可视化
        ################################################################
        # 正向推理结果（t=0时刻）
        pred_u_forward = y_normalizer.decode(xt_forward[-1].view(ntest, S, S, T))
        true_u = y_normalizer.decode(test_u_norm.squeeze(-2))
        pred_u_t0 = pred_u_forward[:, :, :, 0]
        true_u_t0 = true_u[:, :, :, 0]
        
        # 反向推理结果（t=0时刻，对应初始状态a）
        pred_a_reverse = a_normalizer.decode(xt_reverse[-1].view(ntest, S, S, T))
        true_a = a_normalizer.decode(test_a_norm.squeeze(-2))
        pred_a_t0 = pred_a_reverse[:, :, :, 0]
        true_a_t0 = true_a[:, :, :, 0]
        
        # 正向推理可视化（t=0）
        plot_2d_results(
            data1=pred_u_t0,
            data2=true_u_t0,
            labels=['Forward Predicted (t=0)', 'Ground Truth (t=0)'],
            title=f'Forward Inference at t=0',
            filename=f'{save_path}{scorenet_model_class.lower()}_forward_t0.png'
        )
        
        # 反向推理可视化（t=0）
        plot_2d_results(
            data1=pred_a_t0,
            data2=true_a_t0,
            labels=['Reverse Predicted (t=0)', 'Ground Truth (t=0)'],
            title=f'Reverse Inference at t=0',
            filename=f'{save_path}{scorenet_model_class.lower()}_reverse_t0.png'
        )
        
        logging.info("正向/反向推理可视化结果已保存")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    config = {}
    with open(args.config, 'r') as f:
        exec(f.read(), config)

    main(config)
