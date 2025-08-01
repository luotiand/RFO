import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
torch.set_default_dtype(torch.double)


class MLP2d_burger(nn.Module):
    def __init__(self, 
                 input_dim: int = 8192,    # 输入特征维度 32x32=1024
                 h_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        
        # 主网络结构
        self.main_net_1 = nn.Sequential(
            nn.Linear(input_dim, h_dim),  # 输入维度调整为3倍特征维度
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            # nn.Linear(4*h_dim, 4*h_dim),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4*h_dim, input_dim)   # 输出维度保持与输入一致
        )
        self.main_net_2 = nn.Sequential(
            nn.Linear(input_dim, h_dim),  # 输入维度调整为3倍特征维度
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            # nn.Linear(4*h_dim, 4*h_dim),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4*h_dim, input_dim)   # 输出维度保持与输入一致
        )

    def forward(self,  x: torch.Tensor):
        # 输入形状验证
        bs, D = x.shape  # [batch_size,  input_dim]        
        # 主网络处理
        output_1 = self.main_net_1(x)  # [bs, T, input_dim]
        x_1 = x + 0.5*output_1    
        output_2 = self.main_net_2(x_1)
        x_2 = x_1 + 0.5*output_2
        return x_2
class MLP2d_Darcy(nn.Module):
    def __init__(self, 
                 dim: int = 421,
                 h_dim: int = 256,
                 k_list: list = [1]):
        super().__init__()
        self.dim = dim
        self.k_list = k_list
        self.register_buffer('pos_enc_x', self._get_positional_encoding(dim, dim, 0))
        self.register_buffer('pos_enc_y', self._get_positional_encoding(dim, dim, 1)) 
        # 时间编码器增强稳定性
        
        self.time_kernel = nn.Linear(3, dim**2) 
        
        # 主网络 (输入维度调整为3*dim²)
        self.main_net = nn.Sequential(
            nn.Linear(4, h_dim),  # 新增x特征
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )
    def _get_positional_encoding(self, height, width, axis=0):
        """Generate 2D positional encoding for one axis"""
        if axis == 0:  # x-axis
            pos = torch.arange(height).float().unsqueeze(1).repeat(1, width)
        else:  # y-axis
            pos = torch.arange(width).float().unsqueeze(0).repeat(height, 1)
        
        # Normalize to [0, 1]
        pos = pos / max(height, width)
        return pos
    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """生成6通道时间特征"""
     
        bs = t.shape[0]
        
     
        t_expanded = t.unsqueeze(-1)  # [bs, 1]
        
        # 生成6个核特征
        features = []
        for k in self.k_list:
            # clip核
            features.append((t_expanded / k).clamp(0, 1))
            # exp核
            features.append(1 - torch.exp(-t_expanded / k))
            # sin核
            features.append(torch.sin(t_expanded * torch.pi / k))
        

        return torch.cat(features, dim=-1)
    def forward(self,  x: torch.Tensor, t: torch.Tensor):
        # 输入形状验证
        assert x.dim() == 3, "输入a应为3维张量"
        bs, H, W = x.shape
        # import ipdb;ipdb.set_trace()
        # 标准化x
        time_feat = self.generate_time_features(t)
        time_feat = self.time_kernel(time_feat) 
        time_feat = time_feat.view(bs, H, W)  
        pos_x = self.pos_enc_x.unsqueeze(0).repeat(bs, 1, 1)
        pos_y = self.pos_enc_y.unsqueeze(0).repeat(bs, 1, 1)   
        # 特征融合 (新增x)
        combined = torch.stack([
            x, 
            time_feat,
            pos_x,
            pos_y
        ], dim=-1)  
        # 主网络处理
        output = self.main_net(combined)  #
        
        # 反标准化恢复x尺度
        return output.squeeze(-1)



class CNN_add(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, h_dim=64, dim=85, k_list=[1]):
        super().__init__()
        self.width = h_dim
        self.dim = dim
        self.time_kernel = nn.Linear(3, dim**2)
        self.k_list = k_list
        
        # Create positional encodings
        self.register_buffer('pos_enc_x', self._get_positional_encoding(dim, dim, 0))
        self.register_buffer('pos_enc_y', self._get_positional_encoding(dim, dim, 1))
        
        # Down-sampling layers
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, self.width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.width, self.width * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Up-sampling layers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.width * 4, self.width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.width, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def _get_positional_encoding(self, height, width, axis=0):
        """Generate 2D positional encoding for one axis"""
        if axis == 0:  # x-axis
            pos = torch.arange(height).float().unsqueeze(1).repeat(1, width)
        else:  # y-axis
            pos = torch.arange(width).float().unsqueeze(0).repeat(height, 1)
        
        # Normalize to [0, 1]
        pos = pos / max(height, width)
        return pos
    
    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """Generate 3-channel time features"""
        bs = t.shape[0]
        t_expanded = t.unsqueeze(-1)  # [bs, 1]
        
        # Generate 3 kernel features
        features = []
        for k in self.k_list:
            # clip kernel
            features.append((t_expanded / k).clamp(0, 1))
            # exp kernel
            features.append(1 - torch.exp(-t_expanded / k))
            # sin kernel
            features.append(torch.sin(t_expanded * torch.pi / k))
        
        return torch.cat(features, dim=-1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        bs = x.shape[0]
        
        # Generate time features
        # t_features = self.generate_time_features(t)
        # t_kernel = self.time_kernel(t_features).view(bs, self.dim, self.dim)
        t_kernel = t.repeat(1,self.dim,self.dim)
        # Get positional encodings (broadcast to batch size)
        pos_x = self.pos_enc_x.unsqueeze(0).repeat(bs, 1, 1)
        pos_y = self.pos_enc_y.unsqueeze(0).repeat(bs, 1, 1)
        
        # Stack all input channels (a, x, pos_x, pos_y, t_kernel)
        x_input = torch.stack([x, pos_x, pos_y, t_kernel], dim=1)
        
        # Down-sampling + up-sampling
        x_down = self.down(x_input)
        x_up = self.up(x_down)
        z = self.out(x_up).squeeze(1)
        
        return z

class CNN_ns(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 out_channels=1,
                 width=16,
                 dim=64, 
                 time_steps=10,
                 k_list=[1, 2]):
        super().__init__()
        self.dim = dim
        self.time_steps = time_steps
        self.k_list = k_list
        self.width = width
        
        # 3D卷积结构 (保持时间维度)
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(width, width*4, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
        )
        
        self.up = nn.Sequential(
            nn.ConvTranspose3d(width*4, width, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(width, out_channels, kernel_size=(1,3,3), padding=(0,1,1)),
        )
        
        # 时间核参数化（单通道）
        self.time_kernel = nn.Sequential(
            nn.Linear(6, 1),  # 将6维特征压缩到1通道
            nn.Tanh()
        )

    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """生成单通道时间特征"""
        bs = t.shape[0]
        
        # 生成原始6维特征
        t_expanded = t.squeeze(-1).squeeze(-1)
        t_expanded = t_expanded.unsqueeze(1).expand(-1, self.time_steps, -1)
        
        features = []
        for k in self.k_list:
            features.append((t_expanded / k).clamp(0, 1))
            features.append(1 - torch.exp(-t_expanded / k))
            features.append(torch.sin(t_expanded * torch.pi / k))
        
        # 合并后通过核函数 [bs, T, 6] → [bs, T, 1]
        time_feat = self.time_kernel(torch.cat(features, dim=-1))
        return time_feat  # [bs, T, 1]

    def forward(self, a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        # 输入维度处理
        bs, T, H, W = a.shape
        
        # 标准化处理
        a_norm = a.unsqueeze(1)  # [bs, 1, T, H, W]
        x_norm = x.unsqueeze(1)  # [bs, 1, T, H, W]
        # print(f"初始显存: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # 生成单通道时间特征 [bs, T, 1] → [bs, 1, T, H, W]
        time_feat = self.generate_time_features(t)
        time_feat = time_feat.unsqueeze(-1).unsqueeze(-1)  # [bs, T, 1, 1, 1]
        time_feat = time_feat.expand(-1, -1, -1, H, W).permute(0,2,1,3,4)  # [bs, 1, T, H, W]
        # print(f"时间特征生成后: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # 特征拼接 (总通道数=1+1+1=3)
        x_in = torch.cat([a_norm, x_norm, time_feat], dim=1)  # [bs, 3, T, H, W]
        # print(x_in.shape)
        # 3D卷积处理
        x_2 = self.down(x_in)
        # print(f"下采样后: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        x_3 = self.up(x_2)  # [bs, 1, T, H, W]
        # print(f"上采样后: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # 输出处理
        return x_3.squeeze(1) + x  # [bs, T, H, W]


class GNN_Darcy(nn.Module):
    def __init__(self, 
                 dim: int = 421,
                 h_dim: int = 256,
                 k_list: list = [1]):
        super().__init__()
        self.dim = dim
        self.k_list = k_list
        
        # 创建2D网格的邻接关系（固定）
        self.edge_index = self._create_grid_edges(dim)
        
        # 时间编码器 - 改为映射到与输入相同的维度
        self.time_encoder = nn.Linear(3, dim**2)
        
        # GNN层
        self.conv1 = GCNConv(3, h_dim)  # 输入特征: a, x, time
        self.conv2 = GCNConv(h_dim, h_dim)
        self.conv3 = GCNConv(h_dim, h_dim)
        
        # 输出层
        self.out_proj = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)  # 每个节点一个输出值
        )

    def _create_grid_edges(self, dim):
        """创建2D网格的邻接关系"""
        edge_index = []
        
        # 为每个节点创建边（4-连通：上、下、左、右）
        for i in range(dim):
            for j in range(dim):
                node_id = i * dim + j
                
                # 右邻居
                if j < dim - 1:
                    right = node_id + 1
                    edge_index.append([node_id, right])
                    edge_index.append([right, node_id])
                
                # 下邻居
                if i < dim - 1:
                    down = (i + 1) * dim + j
                    edge_index.append([node_id, down])
                    edge_index.append([down, node_id])
        
        # 转换为PyTorch Geometric格式
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """生成时间特征"""
        bs = t.shape[0]
        t_expanded = t.unsqueeze(-1)  # [bs, 1]
        
        # 生成多种时间核特征
        features = []
        for k in self.k_list:
            features.append((t_expanded / k).clamp(0, 1))  # clip核
            features.append(1 - torch.exp(-t_expanded / k))  # exp核
            features.append(torch.sin(t_expanded * torch.pi / k))  # sin核
        
        return torch.cat(features, dim=-1)

    def forward(self, a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        bs, H, W = a.shape
        device = a.device
        
        # 确保edge_index在正确的设备上
        edge_index = self.edge_index.to(device)
        
        # 标准化输入
        a_mean = a.mean(dim=(1, 2), keepdim=True)
        a_std = a.std(dim=(1, 2), keepdim=True) + 1e-6
        a_norm = (a - a_mean) / a_std
        
        x_mean = x.mean(dim=(1, 2), keepdim=True)
        x_std = x.std(dim=(1, 2), keepdim=True) + 1e-6
        x_norm = (x - x_mean) / x_std
        
        # 生成时间特征并映射到与输入相同的维度
        time_feat = self.generate_time_features(t)  # [bs, 3]
        time_embedding = self.time_encoder(time_feat).view(bs, H, W)  # [bs, H, W]
        
        # 为每个样本创建图
        batch_list = []
        for i in range(bs):
            # 节点特征: [a_norm, x_norm, time_embedding]
            a_node = a_norm[i].view(-1, 1)  # [N, 1]
            x_node = x_norm[i].view(-1, 1)  # [N, 1]
            t_node = time_embedding[i].view(-1, 1)  # [N, 1]
            
            # 合并特征
            node_features = torch.cat([a_node, x_node, t_node], dim=1)  # [N, 3]
            
            # 创建图数据
            graph_data = Data(
                x=node_features,
                edge_index=edge_index
            )
            batch_list.append(graph_data)
        
        # 批处理图
        batch = Batch.from_data_list(batch_list).to(device)
        
        # GNN处理
        h = F.relu(self.conv1(batch.x, batch.edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        h = F.relu(self.conv2(h, batch.edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.conv3(h, batch.edge_index)
        
        # 输出层
        out = self.out_proj(h)  # [N, 1]
        
        # 重塑回2D网格
        out = out.view(bs, H, W)
        
        return out