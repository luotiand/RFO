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
from scorenet.scorenet import MLP1d, MLP2d, CNN, MLP2d_Darcy, CNN_add, CNN_ns, FNO3d, MLP2d_bg, MLP2d_Darcy,GNN_Darcy
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
    """计算两种形式的平均绝对百分比误差(MAPE)"""
    with torch.no_grad():
        diff_t = abs((pred - true).mean(axis=0))
        data_t = abs(true.mean(axis=0))
        mask = data_t > 1e-10
        
        if mask.sum() == 0:
            mape1 = torch.tensor(0.0)
        else:
            mape1 = (diff_t[mask] / data_t[mask]).mean() * 100
        
        if data_t.mean() < 1e-10:
            mape2 = torch.tensor(0.0)
        else:
            mape2 = (diff_t.mean() / data_t.mean()) * 100
        
        return mape1.item(), mape2.item()
def squared_absolute_error_loss(output, target):
    return torch.mean((torch.abs(output - target)) ** 2)

def main(config):
    # 动态设置GPU，优先使用GPU 1，不可用则使用GPU 0
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
    T = config['T']
    eq_dt = config['eq_dt']
    rf_dt = config['rf_dt']
    h_dim = config['h_dim']
    train = config['train']
    scorenet_model_class = config['scorenet_model_class']
    model_name = config['model_name']
    rf = config['rf']
    eq = config['eq']
    target_size=config['target_size']
    
    # 加载数据集（不进行标准化）
    train_dataset = darcyDataset('/data5/store1/dlt/rectified_flow/data/piececonst_r421_N1024_smooth1.mat', mode='train',target_size=target_size)
    test_dataset = darcyDataset('/data5/store1/dlt/rectified_flow/data/piececonst_r421_N1024_smooth2.mat', mode='train',target_size=target_size)
    
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
    scorenet_model = globals()[scorenet_model_class](dim = target_size)
    scorenet_model = scorenet_model.to(device)
    logging.info(f"Model loaded on {device}, class: {scorenet_model_class}")

    # 初始化数据并移至设备
    sample_a, sample_x = test_dataset.get_full_data()
    a = sample_a.to(device).to(dtype=torch.float64)
    x = sample_x.to(device).to(dtype=torch.float64)
    
    # 应用标准化
    a = (a - a_mean) / a_std
    x = (x - x_mean) / x_std
    
    logging.info(f"Data loaded to {device}, shapes: a={a.shape}, x={x.shape}")

    # 初始化模型
    score_net = scorenet_model
    logging.info(f"Single GPU mode on {device}")

    if train:
        opt = optim.Adam(params=score_net.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.MSELoss()

        training_loss = [None for _ in range(niter)]
        mape_records1 = []
        mape_records2 = []
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
                
                t = torch.rand(batch_size, 1, device=device).to(dtype=torch.float64)
                t = t.view(batch_size, *([1] * (len(a_.shape) - 1)))
                opt.zero_grad()
                
                xt_ = rf.straight_process(a_, x_, t)
                exact_score = x_ - a_
                score = score_net(a_, xt_, t)
                loss = criterion(exact_score, score)

                loss.backward()
                opt.step()
                
                # 释放显存
                del a_, x_, t, xt_, exact_score, score
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            training_loss[it] = loss.item()

            # 记录MAPE
            if (it + 1) % check_interval == 0 or it == niter - 1:
                score_net.eval()
                with torch.no_grad():
                    xt = [a]
                    for t_val in np.arange(start=0.0, stop=T, step=rf_dt):
                        t_tensor = torch.ones(len(xt[0]), 1, 1, device=device) * t_val
                        score = score_net(a, xt[-1], t_tensor)
                        xt_ = rf.forward_process(xt=xt[-1], score=score, dt=rf_dt)
                        xt.append(xt_)
                    
                    # 计算误差前先还原数据
                    xt_last = xt[-1] * x_std + x_mean  # 还原预测值
                    x_true = x * x_std + x_mean        # 还原真实值
                    
                    mape1, mape2 = calculate_mape(xt_last, x_true)
                    mape_records1.append(mape1)
                    mape_records2.append(mape2)
                    mape_iterations.append(it + 1)
                    
                    logging.info(f"Iteration {it + 1}/{niter}, MAPE1: {mape1:.4f}%, MAPE2: {mape2:.4f}%")
                
                score_net.train()

            # 学习率衰减
            if (it+1) % (niter//4) == 0:
                new_lr = opt.param_groups[0]['lr'] / 4
                logging.info(f"LR reduced from {opt.param_groups[0]['lr']} to {new_lr}")
                opt.param_groups[0]['lr'] = new_lr

            # 进度日志
            if (it + 1) % 5 == 0:
                iter_end_time = time.time()
                elapsed = iter_end_time - iter_start_time
                iter_start_time = iter_end_time
                remaining = (niter - it - 1) * (elapsed / 5)
                logging.info(f"Iter {it+1}/{niter}, Loss: {loss.item():.8f}")
                logging.info(f"Last 5 iters: {elapsed:.2f}s, Remaining: {remaining/60:.2f}min")

        # 保存模型
        torch.save(score_net.state_dict(), f"{para_path}{model_name}")

        # 绘制训练曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(range(1, niter+1), training_loss, 'b-')
        ax1.set_ylabel("Loss", fontsize=14)
        ax1.set_yscale("log")
        ax1.set_title("Training Loss", fontsize=16)
        ax1.grid(True)
        
        ax2.plot(mape_iterations, mape_records2, 'r-o', label='Relative Error')
        ax2.set_xlabel("Iteration", fontsize=14)
        ax2.set_ylabel("Error (%)", fontsize=14)
        ax2.set_title("Relative Error", fontsize=16)
        ax2.grid(True)
        ax2.legend()
        
        plt.savefig(f"{save_path}{scorenet_model_class.lower()}_{target_size}_metrics.png", dpi=300)
        plt.close()

    # 评估模型
    state_dict = torch.load(f"{para_path}{model_name}", map_location=device)
    score_net.load_state_dict(state_dict)
    score_net.eval()
    with torch.no_grad():
        xt = [a]
        for t_val in np.arange(0.0, T, rf_dt):
            t = torch.ones(len(xt[0]), 1, 1, device=device) * t_val
            score = score_net(a, xt[-1], t)
            xt_ = rf.forward_process(xt[-1], score, rf_dt)
            xt.append(xt_)

        # 还原数据用于绘图
        xt_last = xt[-1] * x_std + x_mean  # 还原预测值
        x_true = x * x_std + x_mean        # 还原真实值
    
    # 绘制结果（使用还原后的数据）
    plot_2d_results(
        xt_last.cpu(), x_true.cpu(),
        ['xt (2D)', 'x (exact)'],
        "Operator Learning",
        f"{save_path}{scorenet_model_class.lower()}_{target_size}_2d.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config file path')
    args = parser.parse_args()

    config_path = args.config
    config = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config)

    main(config)