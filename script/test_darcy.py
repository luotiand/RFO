import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
import scipy
# 假设MatReader是用于读取.mat文件的类
class MatReader:
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape)-1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

class darcyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, train_ratio=0.9, mode='train'):
        """简化版Darcy流数据集，直接返回原始数据"""
        self.file_path = file_path
        self.reader = MatReader(file_path)
        
        # 读取输入输出数据
        self.inputs = self.reader.read_field('coeff')   # [n, 421, 421]
        self.outputs = self.reader.read_field('sol')    # [n, 421, 421]
        
        # 数据集划分
        total_samples = len(self.inputs)
        indices = np.random.permutation(total_samples)
        split_idx = int(train_ratio * total_samples)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """返回单个样本的输入和输出"""
        original_idx = self.indices[idx]
        a = self.inputs[original_idx]   # 渗透率场 [421, 421]
        x = self.outputs[original_idx]  # 压力场 [421, 421]
        return a.float(), x.float()
    
    def get_full_data(self):
        """返回整个数据集"""
        x_full = self.inputs[self.indices]   # [n_samples, 421, 421]
        y_full = self.outputs[self.indices]  # [n_samples, 421, 421]
        return x_full, y_full

def main():
    # 替换为实际数据文件路径
    file_path = "/data5/store1/dlt/rectified_flow/data/piececonst_r421_N1024_smooth1.mat"
    
    # 创建数据集实例（训练集）
    train_dataset = darcyDataset(
        file_path=file_path,
        train_ratio=0.9,
        mode='train'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 获取一个批次的样本
    for a_batch, x_batch in train_loader:
        break
    
    # 打印数据集信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(darcyDataset(file_path, mode='valid'))}")
    
    # 打印输入输出形状
    print("\n数据形状:")
    print(f"输入 (渗透率场) 形状: {a_batch.shape}")
    print(f"输出 (压力场) 形状: {x_batch.shape}")
    
    # 验证维度
    print(f"\n维度验证:")
    print(f"输入维度: {a_batch.ndim} (期望3维: batch, height, width)")
    print(f"输出维度: {x_batch.ndim} (期望3维: batch, height, width)")
    print(f"空间分辨率: {a_batch.shape[1]}x{a_batch.shape[2]} (期望421x421)")

if __name__ == "__main__":
    main()