import torch.nn as nn
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.double)
class MLP1d(nn.Module):
    def __init__(self, dim: int = 100, h_dim: int = 200) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(0.5),   
            nn.Tanh(),
            nn.Linear(h_dim, 4*h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = t.expand_as(x)
        x = torch.cat((x, t), dim=-1)
        return self.net(x)
class MLP2d(nn.Module):
    def __init__(self, dim: int = 100, h_dim: int = 1024) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(int(2 * dim**2), h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            nn.Linear(4*h_dim, int(dim**2))
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        n = x.shape[0]
        t = t.expand_as(x)
        x = torch.cat((x, t), dim=-1)

        x = x.view(n, -1)
        y = self.net(x)
        z = y.view(n, int(self.dim), int(self.dim)) 
        
        return z

class MLP2d_add(nn.Module):
    def __init__(self, dim: int = 100, h_dim: int = 1024) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(int(3 * dim**2), h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            nn.Linear(4*h_dim, int(dim**2))
        )

    def forward(self,a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        n = x.shape[0]
        a_mean = a.mean(dim=(1,2), keepdim=True)  # [bs,1,1]
        a_std = a.std(dim=(1,2), keepdim=True) + 1e-6
        a = (a - a_mean) / a_std
        x_mean = x.mean(dim=(1,2), keepdim=True)  # [bs,1,1]
        x_std = x.std(dim=(1,2), keepdim=True) + 1e-6
        x = (x - x_mean) / x_std
        t = t-0.5
        t = t.expand_as(x)
        x = torch.cat((x, a), dim=-1)
        x = torch.cat((x, t), dim=-1)
        x = x.view(n, -1)
        y = self.net(x)
        z = y.view(n, int(self.dim), int(self.dim)) 
        z = z*a_std+a_mean
        return z


class MLP2d_ns(nn.Module):
    def __init__(self, 
                 dim: int = 64, 
                 h_dim: int = 256,
                 time_steps: int = 10,
                 k_list: list = [1,2]):
        super().__init__()
        self.dim = dim
        self.time_steps = time_steps
        self.k_list = k_list
        
        # 时间核参数化
        self.time_kernel = nn.Linear(6, dim**2)  # 6个特征 → 空间维度
        
        # 主网络 (输入维度调整为3*dim²)
        self.main_net = nn.Sequential(
            nn.Linear(3*dim**2, h_dim),  # 新增x特征
            nn.ReLU(),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            nn.Linear(4*h_dim, dim**2)
        )

    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """生成6通道时间特征"""
        # 原始t形状: [bs, 1, 1, 1]
        bs = t.shape[0]
        
        # 扩展至 [bs, time_steps, 1]
        t_expanded = t.squeeze(-1).squeeze(-1)  # [bs, 1]
        t_expanded = t_expanded.unsqueeze(1).expand(-1, self.time_steps, -1)  # [bs, T, 1]
        
        # 生成6个核特征
        features = []
        for k in self.k_list:
            # clip核
            features.append((t_expanded / k).clamp(0, 1))
            # exp核
            features.append(1 - torch.exp(-t_expanded / k))
            # sin核
            features.append(torch.sin(t_expanded * torch.pi / k))
        
        # 合并特征 [bs, T, 6]
        return torch.cat(features, dim=-1)

    def forward(self, a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        # 输入形状验证
        assert a.dim() == 4, "输入a应为4维张量"
        bs, T, H, W = a.shape
        
        # 标准化x
        x_mean = x.mean(dim=(2,3), keepdim=True)
        x_std = x.std(dim=(2,3), keepdim=True) + 1e-6
        x_norm = (x - x_mean) / x_std
        a_mean = a.mean(dim=(2,3), keepdim=True)
        a_std = a.std(dim=(2,3), keepdim=True) + 1e-6
        a_norm = (a - a_mean) / a_std
        # 生成时间特征 [bs, T, 6]
        time_feat = self.generate_time_features(t)
        
        # 时间特征映射到空间维度 [bs, T, H*W]
        time_feat = self.time_kernel(time_feat)  # [bs, T, H*W]
        time_feat = time_feat.view(bs, T, H, W)  # [bs, T, H, W]
        
        # 特征融合 (新增x)
        combined = torch.cat([
            a_norm.view(bs, T, -1),    # 原始a特征 [bs, T, H*W]
            x_norm.view(bs, T, -1), # 标准化x特征 [bs, T, H*W]
            time_feat.view(bs, T, -1) # 时间特征 [bs, T, H*W]
        ], dim=-1)  # [bs, T, 3*H*W]
        
        # 主网络处理
        output = self.main_net(combined)  # [bs, T, H*W]
        
        # 反标准化恢复x尺度
        return output.view(bs, T, H, W) * x_std + x_mean

class MLP2d_bg(nn.Module):
    def __init__(self, 
                 input_dim: int = 1024,    # 输入特征维度 32x32=1024
                 h_dim: int = 256,
                 time_steps: int = 201,    # 对应扩展后的时间步
                 k_list: list = [1, 2]):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.k_list = k_list
        
        # 时间特征生成层 (6核特征)
        self.time_kernel = nn.Linear(6, input_dim)
        
        # 主网络结构
        self.main_net = nn.Sequential(
            nn.Linear(3*input_dim, h_dim),  # 输入维度调整为3倍特征维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*h_dim, input_dim)   # 输出维度保持与输入一致
        )

    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """生成时间特征矩阵"""
        bs = t.shape[0]
        
        # 扩展时间维度 [bs, time_steps, 1]
        t_expanded = t.view(bs, 1, 1).expand(-1, self.time_steps, -1)
        
        # 生成6种时间核特征
        features = []
        for k in self.k_list:
            features.extend([
                (t_expanded / k).clamp(0, 1),          # 截断核
                1 - torch.exp(-t_expanded / k),       # 指数核 
                torch.sin(t_expanded * torch.pi / k)  # 正弦核
            ])
        
        return torch.cat(features, dim=-1)  # [bs, T, 6]

    def forward(self, a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        # 输入形状验证
        assert a.shape == x.shape, "输入维度需一致"
        bs, T, D = a.shape  # [batch_size, time_steps, input_dim]
        
        # 标准化处理
        def normalize(tensor):
            mean = tensor.mean(dim=2, keepdim=True)
            std = tensor.std(dim=2, keepdim=True) + 1e-6
            return (tensor - mean) / std
        
        a_norm = normalize(a)
        x_norm = normalize(x)
        
        # 生成时间特征 [bs, T, 6]
        time_feat = self.generate_time_features(t)
        
        # 时间特征映射 [bs, T, input_dim]
        time_feat = self.time_kernel(time_feat)
        
        # 特征拼接
        combined = torch.cat([
            a_norm,        # 原始输入特征
            x_norm,        # 当前状态特征
            time_feat      # 时间编码特征
        ], dim=-1)         # [bs, T, 3*input_dim]
        
        # 主网络处理
        output = self.main_net(combined)  # [bs, T, input_dim]
        
        # 反标准化
        x_mean = x.mean(dim=2, keepdim=True)
        x_std = x.std(dim=2, keepdim=True) + 1e-6
        return output * x_std + x_mean

class MLP2d_burger(nn.Module):
    def __init__(self, 
                 input_dim: int = 8192,    # 输入特征维度 32x32=1024
                 h_dim: int = 512,
                 k_list: list = [1,2]):
        super().__init__()
        self.input_dim = input_dim
        self.k_list = k_list
        
        # 时间特征生成层 (6核特征)
        self.time_kernel = nn.Linear(6, input_dim)
        
        # 主网络结构
        self.main_net = nn.Sequential(
            nn.Linear(3*input_dim, h_dim),  # 输入维度调整为3倍特征维度
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            # nn.Linear(4*h_dim, 4*h_dim),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4*h_dim, input_dim)   # 输出维度保持与输入一致
        )

    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """生成时间特征矩阵"""
        # print(t.shape)
        bs = t.shape[0]
        t_expanded = t.view(bs, 1)
        
        # 生成6种时间核特征
        features = []
        for k in self.k_list:
            features.extend([
                (t_expanded / k).clamp(0, 1),          # 截断核
                1 - torch.exp(-t_expanded / k),       # 指数核 
                torch.sin(t_expanded * torch.pi / k)  # 正弦核
            ])

        return torch.cat(features, dim=-1)  # [bs, T, 6]

    def forward(self, a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        # 输入形状验证
        assert a.shape == x.shape, "输入维度需一致"
        bs, D = a.shape  # [batch_size,  input_dim]
        
        # # 标准化处理
        # def normalize(tensor):
        #     mean = tensor.mean(dim=-1, keepdim=True)
        #     std = tensor.std(dim=-1, keepdim=True) + 1e-6
        #     return (tensor - mean) / std
        
        
        # 生成时间特征 [bs,  6]
        time_feat = self.generate_time_features(t)
        
        # 时间特征映射 [bs, input_dim]
        time_feat = self.time_kernel(time_feat)
        # 特征拼接
        combined = torch.cat([
            a,        # 原始输入特征
            x,        # 当前状态特征
            time_feat      # 时间编码特征
        ], dim=-1)         # [bs, T, 3*input_dim]
        
        # 主网络处理
        output = self.main_net(combined)  # [bs, T, input_dim]
        
        return output
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
class CNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, width=4):
        super().__init__()
        self.width = width
        # Down-sampling layers
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(width),
            nn.Sigmoid(),
            nn.Conv2d(width, width * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.Sigmoid(),
        )
        # Up-sampling layers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(width),
            nn.Sigmoid(),
            nn.ConvTranspose2d(width, out_channels, kernel_size=4, stride=2, padding=1),
        )
        self.out = nn.Conv2d(out_channels,out_channels,kernel_size =1,stride = 1, padding=0)
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Expand time input and stack with spatial input
        t = t.expand_as(x)  # Ensure `t` is on the same device as `x`
        x_1 = torch.stack([x, t], dim=1)  # Ensure `x` is on the same device as `t`

        # Down-sampling + residual connection
        x_2 = self.down(x_1)
        x_3 = self.up(x_2)
        x_4 = self.out(x_3)
        z = x_4.squeeze(1)+x
        return z* x_std + x_mean


import torch
import torch.nn as nn
import math

class CNN_add(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, width=64, dim=85, k_list=[1]):
        super().__init__()
        self.width = width
        self.dim = dim
        self.time_kernel = nn.Linear(3, dim**2)
        self.k_list = k_list
        
        # Create positional encodings
        self.register_buffer('pos_enc_x', self._get_positional_encoding(dim, dim, 0))
        self.register_buffer('pos_enc_y', self._get_positional_encoding(dim, dim, 1))
        
        # Down-sampling layers
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Up-sampling layers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, out_channels, kernel_size=3, stride=1, padding=1),
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
    
    def forward(self, a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        bs = a.shape[0]
        
        # Generate time features
        t_features = self.generate_time_features(t)
        t_kernel = self.time_kernel(t_features).view(bs, self.dim, self.dim)
        
        # Get positional encodings (broadcast to batch size)
        pos_x = self.pos_enc_x.unsqueeze(0).repeat(bs, 1, 1)
        pos_y = self.pos_enc_y.unsqueeze(0).repeat(bs, 1, 1)
        
        # Stack all input channels (a, x, pos_x, pos_y, t_kernel)
        x_input = torch.stack([a, x, pos_x, pos_y, t_kernel], dim=1)
        
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
import torch
import torch.nn as nn
import torch.fft

class FourierBlock2D(nn.Module):
    def __init__(self, modes_h, modes_w, channels):
        super().__init__()
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.weights = nn.Parameter(
            torch.randn(channels, modes_h, modes_w, 2) * 0.02
        )
        self.time_conv = nn.Conv3d(
            channels, channels, 
            kernel_size=(3, 1, 1), 
            padding=(1, 0, 0)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        freq_w = W // 2 + 1
        
        # 傅里叶变换
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # [B,C,T,H,freq_w,2]
        
        # 动态截断
        trunc_h = min(self.modes_h, H)
        trunc_w = min(self.modes_w, freq_w)
        ft_trunc = x_ft[..., :trunc_h, :trunc_w, :]  # [B,C,T,trunc_h,trunc_w,2]
        
        # 频域卷积（使用字母下标）
        out_ft = torch.einsum('bcthwr,chwr->bcthwr', 
                            ft_trunc, 
                            self.weights[:, :trunc_h, :trunc_w, :])
        
        # 填充完整频域
        out_ft_full = torch.zeros_like(x_ft)
        out_ft_full[..., :trunc_h, :trunc_w, :] = out_ft
        
        # 逆变换恢复空间尺寸
        x_space = torch.fft.irfft2(
            torch.complex(out_ft_full[...,0], out_ft_full[...,1]),
            s=(H, W)
        )
        
        # 时间卷积
        x_time = self.time_conv(x)
        
        return torch.relu(x_space + x_time)

class FNO3d(nn.Module):
    def __init__(self, 
                 modes1=4, 
                 modes2=4, 
                 width=32,
                 time_steps=10, 
                 k_list=[1,2]):
        super().__init__()
        self.time_steps = time_steps
        self.k_list = k_list
        
        # 保持原始时间特征生成器
        self.time_kernel = nn.Linear(6, 1)
        
        # 输入投影层（通道数3 = a + x + time_feat）
        self.in_proj = nn.Conv3d(3, width, kernel_size=1)
        
        # 傅里叶块堆叠
        self.fourier_blocks = nn.Sequential(
            FourierBlock2D(modes1, modes2, width),
            FourierBlock2D(modes1, modes2, width),
            FourierBlock2D(modes1, modes2, width)
        )
        
        # 输出层保持原设计
        self.out_proj = nn.Sequential(
            nn.Conv3d(width, 128, 1),
            nn.GELU(),
            nn.Conv3d(128, 1, 1)
        )

    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        bs = t.shape[0]
        t_expanded = t.squeeze(-1).squeeze(-1)
        t_expanded = t_expanded.unsqueeze(1).expand(-1, self.time_steps, -1)
        
        features = []
        for k in self.k_list:
            features.append((t_expanded / k).clamp(0, 1))
            features.append(1 - torch.exp(-t_expanded / k))
            features.append(torch.sin(t_expanded * torch.pi / k))
        
        time_feat = self.time_kernel(torch.cat(features, dim=-1))
        return time_feat  # [bs, T, 1]

    def forward(self, a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        bs, T, H, W = a.shape
        
        # 生成时间特征（完全保持原流程）
        time_feat = self.generate_time_features(t)
        time_feat = time_feat.unsqueeze(-1).unsqueeze(-1)          # [bs, T, 1, 1, 1]
        time_feat = time_feat.expand(-1, -1, -1, H, W).permute(0,2,1,3,4)  # [bs, 1, T, H, W]
        
        # 输入拼接保持原方式
        a_5d = a.unsqueeze(1)  # [bs, 1, T, H, W]
        x_5d = x.unsqueeze(1)  # [bs, 1, T, H, W]
        x_in = torch.cat([a_5d, x_5d, time_feat], dim=1)  # [bs, 3, T, H, W]
        
        # 投影与傅里叶处理
        x = self.in_proj(x_in)
        x = self.fourier_blocks(x)
        
        # 输出处理
        return self.out_proj(x).squeeze(1)

# if __name__ == "__main__":
#     # 验证测试
#     B, T, H, W = 2, 10, 64, 64
#     model = OriginalTimeFNO()
#     a = torch.randn(B, T, H, W)
#     x = torch.randn(B, T, H, W)
#     t = torch.randn(B, 1, 1, 1)
    
#     out = model(a, x, t)
#     print(f"输出形状: {out.shape}")  # 应得到 [2, 10, 64, 64]
    
#     # 参数量统计
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"总参数量: {total_params/1e6:.2f} M")  # 约1.24M (原版12.67M)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
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