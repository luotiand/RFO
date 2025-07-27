import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import subprocess
from script.plot import plot_2d_results, plot_results, hinton, plot_3d_compare_with_diff
from script.ode_data import Eq1, WaveEquation, PoissonEquation, HeatEquation
from rectified.rectified_flow import RectFlow
import time
import matplotlib.pyplot as plt
from script.dataset import MyDataset_ns
from torch.utils.data import DataLoader
from scorenet.scorenet import MLP1d, MLP2d, CNN, MLP2d_ns, CNN_add, CNN_ns, FNO3d
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
        # 先对批次维度求平均
        diff_t = abs((pred - true).mean(axis=0))
        data_t = abs(true.mean(axis=0))
        
        # 避免除以零
        mask = data_t > 1e-10
        
        # MAPE1: 每个位置计算相对误差，然后求平均
        if mask.sum() == 0:
            mape1 = torch.tensor(0.0)
        else:
            mape1 = (diff_t[mask] / data_t[mask]).mean() * 100
        
        # MAPE2: 先求平均差异和平均真值，再计算相对误差
        if data_t.mean() < 1e-10:
            mape2 = torch.tensor(0.0)
        else:
            mape2 = (diff_t.mean() / data_t.mean()) * 100
        
        return mape1.item(), mape2.item()

def squared_absolute_error_loss(output, target):
    return torch.mean((torch.abs(output - target)) ** 2)

def main(config):
    # 从配置字典中提取参数
    save_path = config['save_path']
    para_path = config['para_path']
    setup_logger(save_path)
    logging.info("Initializing training process...")
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
    
    # 强制使用CUDA 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 加载数据集
    train_dataset = MyDataset_ns(
        file_path='/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat',
        input_steps=10,
        pred_steps=10,
        train=True
    )
    
    test_dataset = MyDataset_ns(
        file_path='/data5/store1/dlt/rectified_flow/data/ns_V1e-3_N5000_T50.mat',
        input_steps=10,
        pred_steps=10,
        train=False
    )
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    # 动态加载模型类并移至设备
    scorenet_model = globals()[scorenet_model_class]()
    scorenet_model = scorenet_model.to(device)
    logging.info(f"Model file will be saved as: {model_name}")
    logging.info(f"Model architecture: {scorenet_model}")

    # 初始化数据并移至设备
    sample_a, sample_x = test_dataset.get_full_data()
    a = sample_a.to(device).to(dtype=torch.float64)
    x = sample_x.to(device).to(dtype=torch.float64)

    # 确保模型在单GPU上运行
    score_net = scorenet_model
    logging.info("Using single GPU (CUDA:0) for computation")

    if train:
        opt = optim.Adam(params=score_net.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.MSELoss()

        # 训练
        training_loss = [None for _ in range(niter)]
        mape_records1 = []
        mape_records2 = []
        mape_iterations = []
        
        start_time = time.time()
        iter_start_time = start_time
        check_interval = max(1, niter // 10)

        for it in range(niter):
            for a_, x_ in dataloader:
                # 确保数据移至指定设备
                a_ = a_.to(device).to(dtype=torch.float64)
                x_ = x_.to(device).to(dtype=torch.float64)

                t = torch.rand(batch_size, 1).to(device).to(dtype=torch.float64)
                t = t.view(batch_size, *([1] * (len(a_.shape) - 1)))
                opt.zero_grad()
                xt_ = rf.straight_process(a_, x_, t)
                exact_score = x_ - a_
                
                # 前向传播
                score = score_net(a_, xt_, t)
                loss = criterion(exact_score, score)

                loss.backward()
                opt.step()
                
                # 释放显存
                del a_, x_, t, xt_, exact_score, score
                torch.cuda.empty_cache()
            
            training_loss[it] = loss.item()

            # 每经过10%的迭代，计算并记录MAPE
            if (it + 1) % check_interval == 0 or it == niter - 1:
                score_net.eval()
                with torch.no_grad():
                    # 生成预测结果并确保在正确设备
                    xt = [a]
                    for t_val in np.arange(start=0.0, stop=T, step=rf_dt):
                        t_tensor = torch.ones(len(xt[0]), 1, 1, 1, device=device) * t_val
                        score = score_net(a, xt[-1], t_tensor)
                        xt_ = rf.forward_process(xt=xt[-1], score=score, dt=rf_dt)
                        xt.append(xt_)
                    
                    # 计算两种MAPE
                    mape1, mape2 = calculate_mape(xt[-1], x)
                    mape_records1.append(mape1)
                    mape_records2.append(mape2)
                    mape_iterations.append(it + 1)
                    
                    logging.info(f"Iteration {it + 1}/{niter}, MAPE1: {mape1:.4f}%, MAPE2: {mape2:.4f}%")
                
                score_net.train()

            if (it+1) % (niter//4) == 0:
                new_lr = opt.param_groups[0]['lr'] / 4
                logging.info(f"Reducing learning rate from {opt.param_groups[0]['lr']} to {new_lr}")
                opt.param_groups[0]['lr'] = new_lr

            if (it + 1) % 5 == 0:
                iter_end_time = time.time()
                elapsed_time = iter_end_time - iter_start_time
                iter_start_time = iter_end_time
                estimated_remaining_time = (niter - it - 1) * (elapsed_time / 5)
                logging.info(f"Iteration {it + 1}/{niter}, Loss: {loss.item():.8f}")
                logging.info(f"Elapsed time for last 5 iterations: {elapsed_time:.2f} seconds")
                logging.info(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes")

        torch.save(score_net.state_dict(), f"{para_path}{model_name}")

        # 绘制损失曲线和MAPE曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.set_tight_layout(True)
        
        ax1.plot(range(1, niter+1), training_loss, color='blue')
        ax1.set_ylabel("Training Loss", fontdict={"size": 14})
        ax1.set_yscale("log")
        ax1.set_title("Rectified Flow Training Loss", fontdict={"size": 16})
        ax1.grid(True)
        
        ax2.plot(mape_iterations, mape_records2, 'r-o', label='relative error')
        ax2.set_xlabel("Iteration", fontdict={"size": 14})
        ax2.set_ylabel("error (%)", fontdict={"size": 14})
        ax2.set_title("relative error", fontdict={"size": 16})
        ax2.grid(True)
        ax2.legend()
        
        plt.savefig(f"{save_path}{scorenet_model_class.lower()}_training_metrics.png", dpi=300)
        plt.close()

    # 加载模型并确保在正确设备
    state_dict = torch.load(f"{para_path}{model_name}", map_location=device)
    score_net.load_state_dict(state_dict)
    score_net.eval()
    
    with torch.no_grad():
        # 评估 Operator Learning 和逆问题
        xt = [a]
        for t_val in np.arange(start=0.0, stop=T, step=rf_dt):
            t = torch.ones(len(xt[0]), 1, 1, 1, device=device) * t_val
            # print(t.shape)
            score = score_net(a, xt[-1], t)
            xt_ = rf.forward_process(xt=xt[-1], score=score, dt=rf_dt)
            xt.append(xt_)
        # print(1)
        # print('\n')
        # yt = [x]
        # for t_val in np.arange(start=0.0, stop=T, step=rf_dt):
        #     t = torch.ones(len(xt[0]), 1, 1, 1, device=device) * t_val
        #     score = score_net(x, yt[-1], T - t_val)
        #     yt_ = rf.reverse_process(xt=yt[-1], score=score, dt=rf_dt)
        #     yt.append(yt_)
        # print(T.shape)
    # 绘制结果
    plot_2d_results(
        data1=xt[-1][100].cpu(),
        data2=x[100].cpu(),
        labels=['xt (2D)', 'x (exact 2D)'],
        title='Operator Learning: xt vs x (2D)',
        filename=f'{save_path}{scorenet_model_class.lower()}_operator_learning_2d.png'
    )
    # plot_3d_compare_with_diff(
    #     data1=xt[-1].cpu().detach().numpy(), 
    #     data2=x.cpu().detach().numpy(),
    #     titles='Operator Learning: xt vs x (3D)',
    #     cmap_main='plasma',
    #     cmap_diff='coolwarm',
    #     elev=40,
    #     azim=-90,
    #     filename=f'{save_path}{scorenet_model_class.lower()}_operator_learning_3d.png'       
    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    # 动态加载配置文件
    config_path = args.config
    config = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config)

    main(config)