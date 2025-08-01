import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import subprocess
from script.plot import plot_1d_results, plot_results, hinton, plot_3d_compare_with_diff
from script.ode_data import Eq1, WaveEquation, PoissonEquation, HeatEquation
from rectified.rectified_flow import RectFlow
import time
import matplotlib.pyplot as plt
from script.dataset import MyDataset_ns, BurgersDataset
from torch.utils.data import DataLoader
from scorenet.scorenet import CNN_add, CNN_ns, MLP2d_burger
import argparse
import logging

torch.set_default_dtype(torch.double)
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

def calculate_mape(pred, true):
    """
    正确计算MAPE：先计算每个样本的相对误差，再对所有样本取平均
    
    Args:
        pred: 预测值，shape: [batch_size, 特征维度1, 特征维度2, ...]
        true: 真实值，shape: 与pred一致
    Returns:
        所有样本的平均MAPE（百分比）
    """
    with torch.no_grad():
        # 1. 计算每个样本的绝对误差（保留batch维度）
        abs_error = torch.abs(pred - true)  # shape: [batch_size, 特征维度1, ...]
        
        # 2. 计算每个样本的真实值（保留batch维度）
        true_abs = torch.abs(true)         # shape: [batch_size, 特征维度1, ...]
        
        # 3. 避免除以零（替换接近零的真实值）
        true_safe = torch.where(
            true_abs < 1e-10, 
            torch.ones_like(true_abs) * 1e-10, 
            true_abs
        )
        
        # 4. 计算每个样本的相对误差（百分比）
        relative_error = (abs_error / true_safe) * 100  # shape: [batch_size, 特征维度1, ...]
        
        # 5. 先对每个样本的所有特征取平均（得到单个样本的MAPE），再对所有样本取平均
        sample_mape = relative_error.view(relative_error.shape[0], -1).mean(dim=1)  # [batch_size]
        overall_mape = sample_mape.mean()  # 所有样本的平均MAPE
        
        return overall_mape.item()

def squared_absolute_error_loss(output, target):
    return torch.mean((torch.abs(output - target)) ** 2)

def main(config):
    # 强制指定使用cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logging.warning("No GPU available, using CPU")
    
    save_path = config['save_path']
    para_path = config['para_path']
    setup_logger(save_path)
    logging.info(f"Initializing training on device: {device}")
    
    eq_T = config['eq_T']
    N = config['N']
    niter = config['niter']
    lr = config['lr']
    batch_size = config['batch_size']
    T = config['T']
    eq_dt = config['eq_dt']
    rf_dt = config['rf_dt']
    h_dim = config['h_dim']
    train = config['train']
    scorenet_model_class = config['scorenet_model_class']
    model_name = config['model_name']
    rf = config['rf']
    eq = config['eq']
    target_len=config['target_len']
    
    # 加载数据集（不进行标准化）
    train_dataset = BurgersDataset('/data5/store1/dlt/rectified_flow/data/burgers_data_R10.mat', mode='train', target_len=target_len)
    test_dataset = BurgersDataset('/data5/store1/dlt/rectified_flow/data/burgers_data_R10.mat', mode='test', target_len=target_len)
    
    # 创建数据加载器（禁用多线程避免设备问题）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    dataloader = train_loader
    test_loader = test_loader

    # 计算训练集的标准化参数（在主函数中进行）
    a_mean, a_std = 0.0, 0.0
    x_mean, x_std = 0.0, 0.0
    
    # 计算a的均值和标准差
    a_all = []
    for a_batch, _ in train_loader:
        a_all.append(a_batch)
    a_all = torch.cat(a_all, dim=0)
    a_mean = a_all.mean()
    a_std = a_all.std() + 1e-6
    
    # 计算x的均值和标准差
    x_all = []
    for _, x_batch in train_loader:
        x_all.append(x_batch)
    x_all = torch.cat(x_all, dim=0)
    x_mean = x_all.mean()
    x_std = x_all.std() + 1e-6
    
    logging.info(f"训练集标准化参数: a_mean={a_mean:.6f}, a_std={a_std:.6f}, x_mean={x_mean:.6f}, x_std={x_std:.6f}")

    # 动态加载模型类并移至指定设备
    scorenet_model = globals()[scorenet_model_class](input_dim=target_len, h_dim=h_dim)
    scorenet_model = scorenet_model.to(device)
    logging.info(f"Model initialized on {device}, class: {scorenet_model_class}")

    # 初始化数据并移至设备
    sample_a, sample_x = test_dataset.get_full_data()
    a = sample_a.to(device).to(dtype=torch.float64)
    x = sample_x.to(device).to(dtype=torch.float64)
    
    # 应用标准化
    a = (a - a_mean) / a_std
    x = (x - x_mean) / x_std
    
    logging.info(f"Data loaded to {device}, shapes: a={a.shape}, x={x.shape}")

    # 初始化模型（单设备模式，不使用DataParallel）
    score_net = scorenet_model
    logging.info(f"Using single GPU mode on {device}")

    if train:
        opt = optim.Adam(params=score_net.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.MSELoss()

        training_loss = [None for _ in range(niter)]
        mape_records = []
        mape_iterations = []
        
        start_time = time.time()
        iter_start_time = start_time
        check_interval = max(1, niter // 10)

        for it in range(niter):
            for a_batch, x_batch in dataloader:
                # 确保数据移至指定设备
                a_ = a_batch.to(device).to(dtype=torch.float64)
                x_ = x_batch.to(device).to(dtype=torch.float64)
                
                # 应用标准化
                a_ = (a_ - a_mean) / a_std
                x_ = (x_ - x_mean) / x_std
                opt.zero_grad()
                score = score_net(a_)
                # import ipdb;ipdb.set_trace()
                loss = criterion(score, x_)

                loss.backward()
                opt.step()
                
                # 释放显存
                del a_, x_, t, xt_, exact_score, score
                torch.cuda.empty_cache()
            
            training_loss[it] = loss.item()

            # 学习率衰减
            if (it+1) % (niter//4) == 0:
                new_lr = opt.param_groups[0]['lr'] / 5
                logging.info(f"Reducing learning rate from {opt.param_groups[0]['lr']} to {new_lr}")
                opt.param_groups[0]['lr'] = new_lr

            # 进度日志
            if (it + 1) % 5 == 0:
                iter_end_time = time.time()
                elapsed_time = iter_end_time - iter_start_time
                iter_start_time = iter_end_time
                estimated_remaining_time = (niter - it - 1) * (elapsed_time / 5)
                logging.info(f"Iteration {it + 1}/{niter}, Loss: {loss.item():.8f}")
                logging.info(f"Elapsed time for last 5 iterations: {elapsed_time:.2f} seconds")
                logging.info(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes")

        # 保存模型
        torch.save(score_net.state_dict(), f"{para_path}{model_name}_steps")

        # 绘制训练曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(range(1, niter+1), training_loss, color='blue')
        ax1.set_ylabel("Training Loss", fontdict={"size": 14})
        ax1.set_yscale("log")
        ax1.set_title("Rectified Flow Training Loss", fontdict={"size": 16})
        ax1.grid(True)
        
        ax2.plot(mape_iterations, mape_records, 'r-o', label='relative error')
        ax2.set_xlabel("Iteration", fontdict={"size": 14})
        ax2.set_ylabel("error (%)", fontdict={"size": 14})
        ax2.set_title("relative error", fontdict={"size": 16})
        ax2.grid(True)
        ax2.legend()
        
        plt.savefig(f"{save_path}{scorenet_model_class.lower()}_{target_len}_training_steps.png", dpi=300)
        plt.close()

    # 评估模型
    score_net.load_state_dict(torch.load(f"{para_path}{model_name}_steps", map_location=device))
    score_net.eval()
    import ipdb ;ipdb.set_trace()
    with torch.no_grad():
        x_pre = score_net(a)
    plot_1d_results(
        data1=x_pre,
        data2=x,
        labels=['xt (2D)', 'x (exact 2D)'],
        title='Operator Learning: xt vs x (2D)',
        filename=f'{save_path}{scorenet_model_class.lower()}_{target_len}_2steps_1d.png'
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    config_path = args.config
    config = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config)

    main(config)