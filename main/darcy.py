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
from scorenet.scorenet import  MLP2d_Darcy, CNN_add, CNN_ns,  MLP2d_Darcy,GNN_Darcy
from scorenet.FNO2d import FNO2d
import argparse
import logging
from Adam import Adam
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True
################################################################
# 改进的损失函数
################################################################
class LpLoss(object):
    def __init__(self, p=2, reduction='mean'):
        super(LpLoss, self).__init__()
        self.p = p
        self.reduction = reduction

    def rel(self, x, y):
        num_examples = x.size(0)
        x_flat = x.view(num_examples, -1)
        y_flat = y.view(num_examples, -1)
        
        diff_norm = torch.norm(x_flat - y_flat, p=self.p, dim=1)
        y_norm = torch.norm(y_flat, p=self.p, dim=1)
        y_norm_safe = torch.where(y_norm < 1e-10, torch.ones_like(y_norm) * 1e-10, y_norm)
        
        rel_error = diff_norm / y_norm_safe
        
        if self.reduction == 'mean':
            return torch.mean(rel_error)
        elif self.reduction == 'sum':
            return torch.sum(rel_error)
        return rel_error

    def abs(self, x, y):
        num_examples = x.size(0)
        x_flat = x.view(num_examples, -1)
        y_flat = y.view(num_examples, -1)
        diff_norm = torch.norm(x_flat - y_flat, p=self.p, dim=1)
        
        if self.reduction == 'mean':
            return torch.mean(diff_norm)
        elif self.reduction == 'sum':
            return torch.sum(diff_norm)
        return diff_norm

    def __call__(self, x, y, mode='rel'):
        if mode == 'rel':
            return self.rel(x, y)
        else:
            return self.abs(x, y)
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

################################################################
# 评估指标计算（仅保留L2相对误差用于绘图）
################################################################
def calculate_metrics(pred, true):
    with torch.no_grad():
        # 只计算L2相对误差（用于评估）
        lp_loss = LpLoss(p=2, reduction='mean')
        l2_rel = lp_loss(pred, true, mode='rel')
        return l2_rel.item()
def squared_absolute_error_loss(output, target):
    return torch.mean((torch.abs(output - target)) ** 2)

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
    test_dataset = darcyDataset('/data5/store1/dlt/rectified_flow/data/piececonst_r421_N1024_smooth2.mat', mode='test',target_size=target_size)
    
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
    
    logging.info(f" a_mean={a_mean:.6f}, a_std={a_std:.6f}, x_mean={x_mean:.6f}, x_std={x_std:.6f}")

    # 动态加载模型类并移至指定设备
    scorenet_model = globals()[scorenet_model_class](12,12,32)
    scorenet_model = scorenet_model.to(device)
    logging.info(f"Model loaded on {device}, class: {scorenet_model_class}")

    # 初始化数据并移至设备
    sample_a, sample_x = test_dataset.get_full_data()
    a = sample_a.to(device).to(dtype=torch.float32)
    x = sample_x.to(device).to(dtype=torch.float32)
    
    # 应用标准化
    a = (a - a_mean) / a_std
    x = (x - x_mean) / x_std
    
    logging.info(f"Data loaded to {device}, shapes: a={a.shape}, x={x.shape}")

    # 初始化模型
    score_net = scorenet_model
    logging.info(f"Single GPU mode on {device}")

    if train:
        optimizer = Adam(score_net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = LpLoss(p=2, reduction='mean')
        
        # 仅记录L2相关指标
        train_losses = []  # 训练L2相对损失
        test_l2rel_list = []  # 测试L2相对误差
        check_interval = max(1, niter // 10)
        start_time = time.time()
        logging.info(f"开始训练 - 总迭代次数: {niter}, 批次大小: {batch_size}")
        
        for epoch in range(niter):
            score_net.train()
            train_loss = 0.0
            t1 = time.time()            
            for a_batch, x_batch in dataloader:
                # 确保数据移至指定设备
                a_ = a_batch.to(device).to(dtype=torch.float32)
                x_ = x_batch.to(device).to(dtype=torch.float32)
                
                # 应用标准化
                a_ = (a_ - a_mean) / a_std
                x_ = (x_ - x_mean) / x_std
                
                t = torch.rand(batch_size, 1, device=device).to(dtype=torch.float32)
                t = t.view(batch_size, *([1] * (len(a_.shape) - 1)))
                t = t.repeat(1, len(a_[0]),len(a_[0]))
                optimizer.zero_grad()
                
                xt_ = rf.straight_process(a_, x_, t)
                exact_score = x_ - a_
                score = score_net(xt_, t)
                pred_score = score_net(xt_, t)
                
                loss = criterion(pred_score, exact_score)

                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                # 释放显存
                del a_, x_, t, xt_, exact_score, pred_score
                torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            scheduler.step()
            

            # 评估
            if (epoch + 1) % check_interval == 0 or epoch == niter - 1:
                score_net.eval()
                with torch.no_grad():
                    xt = [a]
                    for t_val in np.arange(0.0, T, rf_dt):
                        t_tensor = torch.ones(len(a), 1, device=device) * t_val
                        t_tensor = t_tensor.view(len(a), *([1] * (len(a.shape) - 1)))
                        t_tensor = t_tensor.repeat(1, len(a[0]),len(a[0]))
                        score = score_net(xt[-1], t_tensor)
                        xt_ = rf.forward_process(xt[-1], score, dt=rf_dt)
                        xt.append(xt_)
                    
                    xt_last = xt[-1] * x_std + x_mean
                    x_true = x * x_std + x_mean
                    l2_rel = calculate_metrics(xt_last, x_true)  # 仅L2相对误差
                    
                    test_l2rel_list.append(l2_rel)
                    logging.info(
                        f"Epoch {epoch+1}/{niter} - "
                        f"Train Loss: {avg_train_loss:.6f}, "
                        f"Test L2 Rel: {l2_rel:.6f}"
                    )
                score_net.train()

            if (epoch + 1) % 5 == 0:
                t2 = time.time()
                epoch_time = t2 - t1
                remaining_time = (niter - epoch - 1) * (t2 - start_time) / (epoch + 1)
                logging.info(
                    f"Epoch {epoch+1} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Time: {epoch_time:.2f}s, "
                    f"Estimated Remaining: {remaining_time/60:.2f}min"
                )
        
        # 保存模型（保持原始路径和名称）
        torch.save(score_net.state_dict(), f"{para_path}{model_name}")
        logging.info(f"模型保存至: {para_path}{model_name}")
        
        # 绘制训练曲线（仅保留L2相关曲线）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # 训练损失曲线
        ax1.plot(range(1, niter+1), train_losses, 'b-', label='Train L2 Relative Loss')
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_yscale("log")
        ax1.set_title("Training Loss", fontsize=14)
        ax1.legend()
        ax1.grid(True)
        
        # 测试L2相对误差曲线（移除MAPE）
        ax2.plot(
            np.arange(check_interval, niter+1, check_interval) if niter != check_interval else [niter],
            test_l2rel_list, 'g-s', label='Test L2 Relative Error'
        )
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Error", fontsize=12)
        ax2.set_title("Test L2 Relative Error", fontsize=14)  # 标题仅保留L2
        ax2.legend()
        ax2.grid(True)
        
        # 保存训练曲线（保持原始命名风格）
        plt.savefig(f"{save_path}{scorenet_model_class.lower()}_{target_size}_training_metrics.png", dpi=300)
        plt.close()
    # 评估模型
    score_net.load_state_dict(torch.load(f"{para_path}{model_name}", map_location=device))
    score_net.eval()
    logging.info("开始推理评估...")
    # import ipdb ;ipdb.set_trace()
    with torch.no_grad():
        # 正向推理
        xt = [a]
        for t_val in np.arange(0.0, T, rf_dt):
            t_tensor = torch.ones(len(a), 1, device=device) * t_val
            t_tensor = t_tensor.view(len(a), *([1] * (len(a.shape) - 1)))
            t_tensor = t_tensor.repeat(1, len(a[0]),len(a[0]))
            score = score_net(xt[-1], t_tensor)
            xt_ = rf.forward_process(xt[-1], score, dt=rf_dt)
            xt.append(xt_)
        
        # 反向推理
        yt = [x]
        for t_val in np.arange(0.0, T, rf_dt):
            t_tensor = torch.ones(len(xt[0]), 1, device=device) * t_val
            t_tensor = t_tensor.view(len(xt[0]), *([1] * (len(a.shape) - 1)))
            t_tensor = t_tensor.repeat(1, len(a[0]),len(a[0]))
            score = score_net(yt[-1], T - t_tensor)
            yt_ = rf.reverse_process(yt[-1], score, dt=rf_dt)
            yt.append(yt_)
        
        # 还原数据
        xt_last = xt[-1] * x_std + x_mean
        x_true = x * x_std + x_mean
        yt_last = yt[-1] * a_std + a_mean
        y_true = a * a_std + a_mean
        
        # 计算最终指标（仅L2）
        final_l2rel = calculate_metrics(xt_last, x_true)
        logging.info(f"最终评估 - L2 Relative Error: {final_l2rel:.6f}")
    # 绘制结果（使用还原后的数据）
    plot_2d_results(
        xt_last, x_true,
        ['Predicted', 'Ground Truth'],
        "Operator Learning",
        f"{save_path}{scorenet_model_class.lower()}_forward_{target_size}_2d.png"
    )
    plot_2d_results(
        yt_last.cpu(), y_true.cpu(),
        ['Predicted', 'Ground Truth'],
        "Reverse Learning",
        f"{save_path}{scorenet_model_class.lower()}_reverse_{target_size}_2d.png"
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