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
from scorenet.FNO1d import FNO1d
import argparse
import logging
from Adam import Adam

torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True

# 全局字体设置（英文）
import numpy as np
import matplotlib.pyplot as plt

def plot_forward_reverse_comparison(forward_gt, forward_pred, reverse_gt, reverse_pred, base_save_path):
    """
    随机选择5个不重复样本，每个样本生成正向/反向推理对比图（共10张图）
    forward_gt: 正向推理真实值 (Tensor/ndarray, 形状: [batch, len])
    forward_pred: 正向推理预测值 (Tensor/ndarray, 形状需与forward_gt一致)
    reverse_gt: 反向推理真实值 (Tensor/ndarray, 形状: [batch, len])
    reverse_pred: 反向推理预测值 (Tensor/ndarray, 形状需与reverse_gt一致)
    base_save_path: 基础保存路径（如"inference"），生成格式：
                   "inference_forward_rand1.png" ~ "inference_forward_rand5.png"
                   "inference_reverse_rand1.png" ~ "inference_reverse_rand5.png"
    """
    # 统一转换为numpy数组（适配Tensor/ndarray输入）
    def to_numpy(tensor):
        return tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    
    # 转换数据格式并确保为浮点型
    f_gt = to_numpy(forward_gt).astype(np.float32)
    f_pred = to_numpy(forward_pred).astype(np.float32)
    r_gt = to_numpy(reverse_gt).astype(np.float32)
    r_pred = to_numpy(reverse_pred).astype(np.float32)
    
    # 1. 随机选择5个不重复样本（校验批次大小）
    batch_size = f_gt.shape[0]
    if batch_size < 5:
        raise ValueError(f"批次大小({batch_size})小于5，无法随机选择5个样本")
    # 生成0~batch_size-1的随机排列，取前5个作为样本索引
    rand_indices = np.random.permutation(batch_size)[:5].tolist()
    print(f"随机选择的样本索引：{rand_indices}")
    
    # 生成空间坐标（适配数据长度，所有样本共用同一坐标）
    x = np.linspace(0, 1, f_gt.shape[1])
    
    # 2. 遍历每个随机样本，绘制并保存正向/反向图
    for idx_idx, sample_idx in enumerate(rand_indices, 1):  # idx_idx: 1~5（样本序号）
        ###########################################################################
        # 绘制当前样本的正向推理图
        ###########################################################################
        fig_forward, ax_forward = plt.subplots(1, 1, figsize=(10, 6))
        # 真实值（蓝实线）、预测值（红虚线）
        ax_forward.plot(x, f_gt[sample_idx], 'b-', linewidth=2.5, label='Ground Truth')
        ax_forward.plot(x, f_pred[sample_idx], 'r--', linewidth=2.5, label='Predicted')
        # 标题（含样本序号）、标签、图例
        ax_forward.set_title(f'Forward ', fontsize=36, bbox=None)
        ax_forward.set_xlabel('Position', fontsize=12)
        ax_forward.set_ylabel('Value', fontsize=12)
        ax_forward.legend(fontsize=20, frameon=False)
        ax_forward.grid(False)  # 无网格
        # 保存
        plt.tight_layout()
        forward_save_path = f"{base_save_path}_forward_rand{idx_idx}.png"
        plt.savefig(forward_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_forward)
        print(f"正向推理图（样本{idx_idx}）已保存：{forward_save_path}")
        
        ###########################################################################
        # 绘制当前样本的反向推理图（与正向用同一样本索引）
        ###########################################################################
        fig_reverse, ax_reverse = plt.subplots(1, 1, figsize=(10, 6))
        ax_reverse.plot(x, r_gt[sample_idx], 'b-', linewidth=2.5, label='Ground Truth')
        ax_reverse.plot(x, r_pred[sample_idx], 'r--', linewidth=2.5, label='Predicted')
        ax_reverse.set_title(f'Inverse', fontsize=36, bbox=None)
        ax_reverse.set_xlabel('Position', fontsize=12)
        ax_reverse.set_ylabel('Value', fontsize=12)
        ax_reverse.legend(fontsize=20, frameon=False)
        ax_reverse.grid(False)  # 无网格
        # 保存
        plt.tight_layout()
        reverse_save_path = f"{base_save_path}_reverse_rand{idx_idx}.png"
        plt.savefig(reverse_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_reverse)
        print(f"反向推理图（样本{idx_idx}）已保存：{reverse_save_path}")
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
        if mode == 'abs':
            return self.abs(x, y)
        else:
            return self.rel(x, y)


################################################################
# 评估指标计算（仅保留L2相对误差用于绘图）
################################################################
def calculate_metrics(pred, true):
    with torch.no_grad():
        # 只计算L2相对误差（用于评估）
        lp_loss = LpLoss(p=2, reduction='mean')
        l2_rel = lp_loss(pred, true, mode='rel')
        return l2_rel.item()


################################################################
# 日志配置
################################################################
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


################################################################
# 主函数
################################################################
def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logging.warning("No GPU available, using CPU")
    
    save_path = config['save_path']
    para_path = config['para_path']
    setup_logger(save_path)
    logging.info(f"Initializing training on device: {device}")
    
    # 核心参数
    niter = config['niter']
    batch_size = config['batch_size']
    lr = config['lr']
    T = config['T']
    rf_dt = config['rf_dt']
    train = config['train']
    scorenet_model_class = config['scorenet_model_class']
    model_name = config['model_name']
    rf = config['rf']
    target_len = config['target_len']
    
    ################################################################
    # 数据加载与标准化
    ################################################################
    train_dataset = BurgersDataset(
        '/data5/store1/dlt/rectified_flow/data/burgers_data_R10.mat',
        mode='train',
        target_len=target_len
    )
    test_dataset = BurgersDataset(
        '/data5/store1/dlt/rectified_flow/data/burgers_data_R10.mat',
        mode='test',
        target_len=target_len
    )
    # import ipdb; ipdb.set_trace()
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
    logging.info(f"Training dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    
    # 计算标准化参数
    a_all, x_all = [], []
    for a_batch, x_batch in train_loader:
        a_all.append(a_batch)
        x_all.append(x_batch)
    a_all = torch.cat(a_all, dim=0)
    x_all = torch.cat(x_all, dim=0)
    
    a_mean, a_std = a_all.mean(), a_all.std() + 1e-6
    x_mean, x_std = x_all.mean(), x_all.std() + 1e-6
    logging.info(f"标准化参数 - a: (mean={a_mean:.6f}, std={a_std:.6f}); x: (mean={x_mean:.6f}, std={x_std:.6f})")
    
    ################################################################
    # 模型初始化
    ################################################################
    scorenet_model = globals()[scorenet_model_class](16, 64)
    scorenet_model = scorenet_model.to(device)
    score_net = scorenet_model
    logging.info(f"Model initialized: {scorenet_model_class} on {device}")
    
    # 测试集数据初始化
    sample_a, sample_x = test_dataset.get_full_data()
    a = (sample_a.to(device) - a_mean) / a_std
    x = (sample_x.to(device) - x_mean) / x_std
    logging.info(f"Test data loaded - a shape: {a.shape}, x shape: {x.shape}")
    
    ################################################################
    # 训练流程
    ################################################################
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
            
            for a_batch, x_batch in train_loader:
                a_ = (a_batch.to(device) - a_mean) / a_std
                x_ = (x_batch.to(device) - x_mean) / x_std
                
                time_options = torch.tensor([0.0,  1.0], device=device, dtype=torch.float32)  # 定义可选时间值
                rand_indices = torch.randint(0, 2, (batch_size, 1), device=device)  # 生成0-2的随机索引
                t = time_options[rand_indices]  # 根据索引选取时间值（形状：(current_bs, 1)）
                t = t.view(batch_size, *([1] * (len(a_.shape) - 1)))
                t = t.repeat(1, len(a_[0]))
                
                optimizer.zero_grad()
                xt_ = rf.straight_process(a_, x_, t)
                exact_score = x_ - a_
                pred_score = score_net(xt_, t)
                
                loss = criterion(pred_score, exact_score)
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
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
                        t_tensor = torch.ones(len(xt[0]), 1, device=device) * t_val
                        t_tensor = t_tensor.repeat(1, len(a[0]))
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
            
            # 进度日志
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
        plt.savefig(f"{save_path}{scorenet_model_class.lower()}_{target_len}_training_metrics.png", dpi=300)
        plt.close()
    
    ################################################################
    # 推理与评估
    ################################################################
    score_net.load_state_dict(torch.load(f"{para_path}{model_name}", map_location=device))
    score_net.eval()
    logging.info("开始推理评估...")
    
    with torch.no_grad():
        # 正向推理
        xt = [a]
        for t_val in np.arange(0.0, T, rf_dt):
            t_tensor = torch.ones(len(xt[0]), 1, device=device) * t_val
            t_tensor = t_tensor.repeat(1, len(a[0]))
            score = score_net(xt[-1], t_tensor)
            xt_ = rf.forward_process(xt[-1], score, dt=rf_dt)
            xt.append(xt_)
        
        # 反向推理
        yt = [x]
        for t_val in np.arange(0.0, T, rf_dt):
            t_tensor = torch.ones(len(yt[0]), 1, device=device) * t_val
            t_tensor = t_tensor.repeat(1, len(a[0]))
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
        
        # 绘图（保持原始命名）
        plot_forward_reverse_comparison(
            forward_gt=x_true,          # 正向真实值
            forward_pred=xt_last,       # 正向预测值
            reverse_gt=y_true,          # 反向真实值
            reverse_pred=yt_last,       # 反向预测值
            base_save_path = f"{save_path}{scorenet_model_class.lower()}_{target_len}",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    config = {}
    with open(args.config, 'r') as f:
        exec(f.read(), config)
    
    main(config)