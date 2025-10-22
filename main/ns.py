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


# 新增：分开绘制训练损失和测试损失曲线
def plot_separate_loss_curves(train_losses, test_losses, test_epochs, save_path, model_class, target_len):
    """
    分开绘制训练损失曲线和测试损失曲线
    - 上子图：训练L2损失
    - 下子图：测试L2损失
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)  # 共享x轴（epoch）
    
    # 1. 训练损失曲线（上子图）
    ax1.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train L2 Loss')
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_yscale("log")  # 对数刻度
    ax1.set_title("Training L2 Loss Curve", fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 测试损失曲线（下子图）
    ax2.plot(test_epochs, test_losses, 'g-s', label='Test L2 Loss')  # 方形标记点
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_yscale("log")  # 对数刻度
    ax2.set_title("Test L2 Loss Curve", fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    # 保存图片
    plt.savefig(f"{save_path}{model_class.lower()}_{target_len}_loss_curves.png", dpi=300)
    plt.close()


def main(config):
    # 统一设备管理（优先GPU 1，否则CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    save_path = config['save_path']
    para_path = config['para_path']
    # 物理时间步可视化目录（最后一维）
    forward_vis_path = os.path.join(save_path, "forward_phys_time/")
    reverse_vis_path = os.path.join(save_path, "reverse_phys_time/")
    os.makedirs(forward_vis_path, exist_ok=True)
    os.makedirs(reverse_vis_path, exist_ok=True)
    setup_logger(save_path)
    
    # 核心参数
    niter = config['niter']
    lr = config['lr']
    batch_size = config['batch_size']
    rf_dt = config['rf_dt']
    train = config['train']
    scorenet_model_class = config['scorenet_model_class']
    model_name = config['model_name']
    rf = config['rf']
    phys_time_steps = 10  # 数据最后一维：物理时间步t=0~9
    check_interval = 50  # 测试评估间隔（与代码中(ep+1)%50==0对应）
    
    # 数据路径与参数
    TRAIN_PATH = '/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat'
    TEST_PATH = '/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat'
    ntrain = 1000
    ntest = 20
    modes = 18
    width = 30
    epochs = niter
    learning_rate = lr
    scheduler_step = 100
    scheduler_gamma = 0.5

    # 数据维度（最后一维为物理时间）
    sub = 1
    S = 64 // sub
    T_in = phys_time_steps
    T_out = 20
    T = 10
    ################################################################
    # 加载数据（统一用to(device)）
    ################################################################
    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in].to(device)
    train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_out:T+T_out].to(device)

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in].to(device)
    test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_out:T+T_out].to(device)

    logging.info(f"数据形状（最后一维为物理时间）: {test_u.shape}")
    
    # 标准化
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a_norm = a_normalizer.encode(train_a)
    test_a_norm = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u_norm = y_normalizer.encode(train_u)
    test_u_norm = y_normalizer.encode(test_u)
    
    # 调整形状（添加通道维度）
    # train_a_norm = train_a_norm.reshape(ntrain, S, S, 1, T_in).repeat([1,1,1,T,1]).to(device)
    # test_a_norm = test_a_norm.reshape(ntest, S, S, 1, T_in).repeat([1,1,1,T,1]).to(device)
    train_a_norm = train_a_norm.reshape(ntrain, S, S, T_in, 1).to(device)
    test_a_norm = test_a_norm.reshape(ntest, S, S, T_in, 1).to(device)


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
    # 模型初始化
    ################################################################
    model = FNO3d(modes, modes, 4, width).to(device)
    logging.info(f"模型参数数量: {count_params(model)}")
    
    if train:
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_step, 
            gamma=scheduler_gamma
        )
    myloss = LpLoss(size_average=False)

    ################################################################
    # 训练流程
    ################################################################
    if train:
        train_losses = []  # 训练损失列表（每个epoch）
        test_losses = []   # 测试损失列表（每50个epoch）
        test_epochs = []   # 测试对应的epoch列表

        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_l2 = 0
            
            for x, y in train_loader:
                current_bs = x.shape[0]
                # t = torch.rand(current_bs, 1, device=device).to(dtype=torch.float32)
                time_options = torch.tensor([0.0, 0.5, 1.0], device=device, dtype=torch.float32)  # 定义可选时间值
                rand_indices = torch.randint(0, 3, (current_bs, 1), device=device)  # 生成0-2的随机索引
                t = time_options[rand_indices]  # 根据索引选取时间值（形状：(current_bs, 1)）
                t = t.view(current_bs, *([1] * (len(x.shape) - 1)))
                t = t.repeat(1, S, S, T, 1)
                
                optimizer.zero_grad()
                xt = rf.straight_process(x, y.unsqueeze(-1), t)
                out = model(xt, t).view(current_bs, S, S, T)
                l2 = myloss(out.view(current_bs, -1), (y - x.squeeze(-1)).view(current_bs, -1))
                l2.backward()
                optimizer.step()
                train_l2 += l2.item()
            
            # 记录训练损失
            avg_train_loss = train_l2 / ntrain
            train_losses.append(avg_train_loss)
            scheduler.step()
            logging.info(f"Epoch {ep+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
            
            # 每50个epoch评估测试损失
            if (ep+1) % check_interval == 0:
                model.eval()
                test_l2 = 0.0
                with torch.no_grad():
                    for x, y in test_loader:
                        current_bs = x.shape[0]
                        xt = [x]
                        
                        # 正向推理
                        for t_val in np.arange(0.0, 1, rf_dt):
                            t = torch.full((current_bs, 1), t_val, device=device, dtype=torch.float32)
                            t = t.view(current_bs, *([1] * (len(x.shape) - 1)))
                            t = t.repeat(1, S, S, T, 1)
                            score = model(xt[-1], t)
                            xt_ = rf.forward_process(xt[-1], score, dt=rf_dt)
                            xt.append(xt_)
                        
                        # 计算测试损失
                        y_decoded = y_normalizer.decode(y.squeeze(-2))
                        xt_last = xt[-1].view(current_bs, S, S, T)
                        out_decoded = y_normalizer.decode(xt_last)
                        test_l2 += myloss(out_decoded.view(current_bs, -1), y_decoded.view(current_bs, -1)).item()
                
                avg_test_loss = test_l2 / ntest
                test_losses.append(avg_test_loss)
                test_epochs.append(ep + 1)
                logging.info(f"测试点 Epoch {ep+1} - Test Loss: {avg_test_loss:.6f}")
        
        # 保存模型
        torch.save(model.state_dict(), f"{para_path}{model_name}")
        logging.info(f"模型已保存至: {para_path}{model_name}")

        # 绘制分开的损失曲线（核心修改）
        plot_separate_loss_curves(
            train_losses, 
            test_losses, 
            test_epochs, 
            save_path, 
            scorenet_model_class, 
            phys_time_steps
        )
        logging.info("分开的训练/测试损失曲线已保存")
    
    ################################################################
    # 推理流程（保持不变）
    ################################################################
    if not train:
        model.load_state_dict(torch.load(f"{para_path}{model_name}", map_location=device))
        logging.info(f"已加载预训练模型: {para_path}{model_name}")
    model.eval()
    logging.info("开始推理，提取数据最后一维（物理时间步）可视化...")
    
    with torch.no_grad():
        # 正向推理
        xt_forward = [test_a_norm]
        for t_val in np.arange(0.0, 1, rf_dt):
            current_bs = xt_forward[-1].shape[0]
            t = torch.full((current_bs, 1), t_val, device=device, dtype=torch.float32)
            t = t.view(current_bs, *([1] * (len(xt_forward[-1].shape) - 1)))
            t = t.repeat(1, S, S, T, 1)
            score = model(xt_forward[-1], t)
            xt_ = rf.forward_process(xt_forward[-1], score, dt=rf_dt)
            xt_forward.append(xt_)
        
        # 反向推理
        xt_reverse = [test_u_norm.unsqueeze(-1)]
        for t_val in np.arange(0.0, 1, rf_dt):
            current_bs = xt_reverse[-1].shape[0]
            t = torch.full((current_bs, 1), t_val, device=device, dtype=torch.float32)
            t = t.view(current_bs, *([1] * (len(xt_reverse[-1].shape) - 1)))
            t = t.repeat(1, S, S, T,1)
            score = model(xt_reverse[-1], 1 - t)
            xt_ = rf.reverse_process(xt_reverse[-1], score, dt=rf_dt)
            xt_reverse.append(xt_)
        
        # 解码并可视化10个物理时间步
        pred_u_forward = y_normalizer.decode(xt_forward[-1].view(ntest, S, S, phys_time_steps))
        true_u = y_normalizer.decode(test_u_norm.view(ntest, S, S, phys_time_steps))
        pred_a_reverse = a_normalizer.decode(xt_reverse[-1].view(ntest, S, S, phys_time_steps))
        true_a = a_normalizer.decode(test_a_norm.view(ntest, S, S, phys_time_steps))
        
        # 正向推理可视化
        for t_step in range(phys_time_steps):
            pred_u_t = pred_u_forward[..., t_step]
            true_u_t = true_u[..., t_step]
            plot_2d_results(
                data1=true_u_t,
                data2=pred_u_t,                
                labels=[f'Ground Truth',f'Prediction' ],
                base_filename=f'{forward_vis_path}t{t_step}'
            )
        
        # 反向推理可视化
        for t_step in range(phys_time_steps):
            pred_a_t = pred_a_reverse[..., t_step]
            true_a_t = true_a[..., t_step]
            plot_2d_results(
                data1=true_a_t,
                data2=pred_a_t,
                labels=[f'Ground Truth',f'Prediction' ],
                base_filename=f'{reverse_vis_path}t{t_step}'
            )
        
        logging.info(f"可视化已保存至:\n正向: {forward_vis_path}\n反向: {reverse_vis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    config = {}
    with open(args.config, 'r') as f:
        exec(f.read(), config)

    main(config)