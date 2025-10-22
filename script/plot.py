import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
def plot_3d_compare_with_diff(data1, data2, titles, cmap_main='viridis', cmap_diff='seismic', 
                             elev=25, azim=-135, filename='operator_learning_3d.png' ,figsize=(24, 8)):
    """
    并排显示 data1、data2 和差异场的 3D 对比
    
    参数：
        data1: 数据1 (N_test, T, H, W)
        data2: 数据2 (N_test, T, H, W)
        titles: 标题列表 [data1_title, data2_title, diff_title]
        cmap_main: 主数据颜色映射
        cmap_diff: 差异颜色映射
        elev: 俯仰角
        azim: 方位角
        figsize: 画布尺寸
    """
    # 数据预处理
    abs_error = np.abs(data1 - data2)
    mae = np.mean(abs_error)
    data3 = data1-data2
    # 计算均方根误差
    rmse = np.sqrt(np.mean(np.square(data1 - data2)))
    amse = np.mean(abs(data3))/np.mean(abs(data2))
    # 计算更合理的相对误差
    # 方法1：相对于数据范围的归一化误差
    data_range = np.max(data1) - np.min(data1)
    nrmse = rmse / data_range * 100  # 归一化RMSE（百分比）
    
    # 方法2：使用更稳健的相对误差计算（考虑数据的典型尺度）
    data_scale = np.mean(np.abs(data1))  # 数据绝对值的平均数作为尺度
    scaled_error = np.mean(abs_error) / data_scale * 100 if data_scale > 0 else 0
    
    # 打印结果
    # print("\n" + "="*50)
    # print("误差统计 (取所有样本平均)")
    # print("="*50)
    # print(f"平均绝对误差 (MAE): {mae:.6f}")
    # print(f"均方根误差 (RMSE): {rmse:.6f}")
    # print(f"平均相对误差 (AMSE): {amse:.6f}")
    # print(f"归一化RMSE (NRMSE): {nrmse:.2f}%")
    # print(f"相对于数据尺度的误差: {scaled_error:.2f}%")
    
    # 添加额外的描述性统计
    # print("\n数据统计:")
    # print(f"参考数据范围: [{np.min(data1):.4f}, {np.max(data1):.4f}], 均值: {np.mean(data1):.4f}")
    # print(f"预测数据范围: [{np.min(data2):.4f}, {np.max(data2):.4f}], 均值: {np.mean(data2):.4f}")
    # print("="*50 + "\n")
    data1_avg = data1.mean(axis=0)  # (T, H, W)
    data2_avg = data2.mean(axis=0)
    diff = data1_avg - data2_avg
    
    # 创建画布和子图
    fig = plt.figure(figsize=figsize)
    axes = [
        fig.add_subplot(131, projection='3d'),
        fig.add_subplot(132, projection='3d'),
        fig.add_subplot(133, projection='3d')
    ]
    
    # 绘制各子图
    _plot_3d_field(axes[0], data1_avg, title=titles[0], cmap=cmap_main, elev=elev, azim=azim)
    print(1)
    _plot_3d_field(axes[1], data2_avg, title=titles[1], cmap=cmap_main, elev=elev, azim=azim)
    _plot_3d_diff(axes[2], diff, title=titles[2], cmap=cmap_diff, elev=elev, azim=azim)
    print(2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(3)
    plt.close()
def _plot_3d_field(ax, data, title, cmap, elev, azim):
    """绘制单个 3D 场"""
    T, H, W = data.shape
    
    # 修复：生成比数据多一个点的网格
    x, y, z = np.meshgrid(
        np.arange(W+1), 
        np.arange(H+1), 
        np.arange(T+1), 
        indexing='ij'
    )

    # 生成等值面 - 确保数据与网格匹配
    threshold = np.median(data)
    
    # 将布尔数组调整为与网格相同大小
    voxel_array = np.zeros((W+1, H+1, T+1), dtype=bool)
    voxel_array[:-1, :-1, :-1] = data.transpose(2, 1, 0) > threshold
    
    # 调整颜色数组大小
    colors = np.zeros((W+1, H+1, T+1, 4))
    normalized_data = data.transpose(2, 1, 0) / data.max()
    colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(normalized_data)
    
    # 绘制体素
    ax.voxels(voxel_array, facecolors=colors, edgecolor='none', alpha=0.3)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Width', labelpad=10)
    ax.set_ylabel('Height', labelpad=10)
    ax.set_zlabel('Time Step', labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    
    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(data)
    plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)

def _plot_3d_diff(ax, diff, title, cmap, elev, azim):
    """绘制差异场"""
    T, H, W = diff.shape
    
    # 修复：生成比数据多一个点的网格
    x, y, z = np.meshgrid(
        np.arange(W+1), 
        np.arange(H+1), 
        np.arange(T+1), 
        indexing='ij'
    )
    
    # 归一化差异值
    vmax = np.abs(diff).max()
    norm_diff = diff / vmax
    
    # 创建比数据大一维的布尔掩码
    pos_mask = np.zeros((W+1, H+1, T+1), dtype=bool)
    neg_mask = np.zeros((W+1, H+1, T+1), dtype=bool)
    
    # 填入数据
    pos_mask[:-1, :-1, :-1] = diff.transpose(2, 1, 0) > 0
    neg_mask[:-1, :-1, :-1] = diff.transpose(2, 1, 0) < 0
    
    # 创建颜色数组
    pos_colors = np.zeros((W+1, H+1, T+1, 4))
    neg_colors = np.zeros((W+1, H+1, T+1, 4))
    
    # 填入颜色数据
    pos_colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(0.5 + norm_diff.transpose(2, 1, 0)/2)
    neg_colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(0.5 + norm_diff.transpose(2, 1, 0)/2)
    
    # 绘制体素
    ax.voxels(pos_mask, facecolors=pos_colors, edgecolor='none', alpha=0.5)
    ax.voxels(neg_mask, facecolors=neg_colors, edgecolor='none', alpha=0.5)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Width', labelpad=10)
    ax.set_ylabel('Height', labelpad=10)
    ax.set_zlabel('Time Step', labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    plt.figure(figsize=(6, 6))
    ax = ax if ax is not None else plt.gca()

    if max_weight is None:
        max_weight = 2**np.ceil(np.log(np.max(np.abs(matrix)))/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('auto', 'box')
    ax.set_xticks([])
    ax.set_yticks([])

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    plt.title("Hinton Diagram of Model Weights")
    plt.savefig('hint', dpi=300)
    plt.show()

# Plot the Hinton diagram
def plot_2d_results(data1, data2, labels, base_filename):
    """
    随机选择5个样本，分别保存GT和预测图，移除横纵坐标数值（无刻度线和标签）
    标题字体调大至16，无边框
    """
    with torch.no_grad():
        batch_size = data1.shape[0]
        if batch_size < 5:
            raise ValueError(f"批次大小({batch_size})小于5，无法随机选择5个样本")
        rand_indices = torch.randperm(batch_size)[:5].tolist()

    for i, idx in enumerate(rand_indices, 1):
        with torch.no_grad():
            gt_sample = data1[idx].cpu().detach().numpy()
            predict_sample = data2[idx].cpu().detach().numpy()

        ###########################################################################
        # 1. 真实值(GT)图：无横纵坐标数值
        ###########################################################################
        fig_gt, ax_gt = plt.subplots(1, 1, figsize=(8, 6))
        im_gt = ax_gt.imshow(gt_sample, cmap='viridis', aspect='auto')
        
        # 核心修改：移除横纵坐标的刻度线和数值
        ax_gt.set_xticks([])  # 清除x轴刻度
        ax_gt.set_yticks([])  # 清除y轴刻度
        ax_gt.set_xticklabels([])  # 清除x轴标签（冗余保障）
        ax_gt.set_yticklabels([])  # 清除y轴标签（冗余保障）
        
        # 标题设置（字体调大+无边框）
        ax_gt.set_title(
            f"{labels[0]}", 
            fontsize=36,
            bbox=None
        )
        fig_gt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        plt.tight_layout()
        gt_save_path = f"{base_filename}_rand{i}_gt.png"
        plt.savefig(gt_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_gt)
        print(f"已保存样本{i} - {labels[0]}图：{gt_save_path}")

        ###########################################################################
        # 2. 预测值(Predict)图：无横纵坐标数值
        ###########################################################################
        fig_pred, ax_pred = plt.subplots(1, 1, figsize=(8, 6))
        im_pred = ax_pred.imshow(predict_sample, cmap='viridis', aspect='auto')
        
        # 同样移除横纵坐标数值
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.set_xticklabels([])
        ax_pred.set_yticklabels([])
        
        # 标题设置
        ax_pred.set_title(
            f"{labels[1]}", 
            fontsize=36,
            bbox=None
        )
        fig_pred.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        plt.tight_layout()
        pred_save_path = f"{base_filename}_rand{i}_predict.png"
        plt.savefig(pred_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_pred)
        print(f"已保存样本{i} - {labels[1]}图：{pred_save_path}")
def plot_1d_results(data1, data2, labels, title, filename):
    """
    绘制一维数据的比较图和差值图并保存，标题同时显示整体平均和当前样本的L2相对误差
    """
    with torch.no_grad():
        # 计算每个样本的L2相对误差
        batch_size = data1.shape[0]
        data1_flat = data1.view(batch_size, -1)
        data2_flat = data2.view(batch_size, -1)
        
        # 分子：预测与真实的L2范数差
        diff_norm = torch.norm(data1_flat - data2_flat, p=2, dim=1)
        # 分母：真实值的L2范数（添加小值保护）
        data2_norm = torch.norm(data2_flat, p=2, dim=1)
        data2_norm_safe = torch.where(
            data2_norm < 1e-10, 
            torch.ones_like(data2_norm) * 1e-10, 
            data2_norm
        )
        
        # 样本级和整体L2相对误差
        sample_l2_rel = diff_norm / data2_norm_safe
        overall_l2_rel = sample_l2_rel.mean().item()  # 整体平均L2误差
        
        # 找到L2相对误差最小的样本
        min_l2_idx = torch.argmin(sample_l2_rel).item()
        current_sample_l2 = sample_l2_rel[min_l2_idx].item()  # 当前展示样本的L2误差
        
        # 打印误差信息
        print(f"所有样本的平均L2相对误差: {overall_l2_rel:.6f}")
        print(f"当前展示样本的L2相对误差: {current_sample_l2:.6f}")
        print(f"样本L2相对误差范围: {sample_l2_rel.min().item():.6f} ~ {sample_l2_rel.max().item():.6f}")
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 绘制误差最小的样本
    axes[0].plot(data1[min_l2_idx].cpu().detach().numpy())
    axes[0].set_title(labels[0])
    axes[0].grid(True)
    
    axes[1].plot(data2[min_l2_idx].cpu().detach().numpy())
    axes[1].set_title(labels[1])
    axes[1].grid(True)
    
    # 绘制差值
    diff = data1[min_l2_idx].cpu().detach().numpy() - data2[min_l2_idx].cpu().detach().numpy()
    axes[2].plot(diff)
    axes[2].set_title(f"Difference ({labels[0]} - {labels[1]})")
    axes[2].grid(True)
    
    # 标题同时显示整体平均和当前样本的L2误差
    plt.suptitle(
        f"{title}"
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # 调整布局容纳两行标题
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_results(data1, data2, labels, title, xlabel, ylabel, filename):
    """
    绘制数据比较图，标题同时显示整体平均L2相对误差
    """
    with torch.no_grad():
        # 计算整体L2相对误差
        data1_flat = data1.view(data1.shape[0], -1)
        data2_flat = data2.view(data2.shape[0], -1)
        diff_norm = torch.norm(data1_flat - data2_flat, p=2, dim=1)
        data2_norm = torch.norm(data2_flat, p=2, dim=1)
        data2_norm_safe = torch.where(data2_norm < 1e-10, torch.ones_like(data2_norm)*1e-10, data2_norm)
        overall_l2_rel = (diff_norm / data2_norm_safe).mean().item()
    
    plt.figure()
    plt.plot(data1.mean(dim=1).cpu().detach().numpy(), label=labels[0].format(len(data1)))
    plt.plot(data2.mean(dim=1).cpu().detach().numpy(), label=labels[1].format(len(data2)))
    
    # 标题添加整体平均L2误差
    plt.title(f"{title}\noverral_L2error: {overall_l2_rel:.6f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # 为两行标题留空间
    plt.savefig(filename, dpi=300)
    plt.close()