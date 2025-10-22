import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io

# 数据加载器（保持用户原有实现）
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

# 自定义数据集



class BurgersDataset(Dataset):
    def __init__(self, 
                 file_path, 
                 train_ratio=0.9, 
                 mode='train',
                 target_len=None):  # 新增参数：目标长度（如4096,2048,512等，None表示不采样）
        # 加载原始数据
        reader = MatReader(file_path, to_cuda=False)
        
        # 读取输入输出数据（根据实际字段名调整）
        self.inputs = reader.read_field('a')    # 原始形状 [4, 8192]
        self.outputs = reader.read_field('u')   # 假设原始形状 [4, 201, 8192]（根据实际调整）
        # import ipdb;ipdb.set_trace()
        # 下采样处理（如果指定了目标长度）
        self.target_len = target_len
        if target_len is not None:
            self.inputs = self.downsample_by_step(self.inputs, target_len)
            # 对outputs的空间维度下采样（假设outputs形状为[样本数, 时间步, 空间长度]）
            self.outputs = self.downsample_by_step(self.outputs, target_len, dim=-1)
        
        # 数据集划分
        total_samples = len(self.inputs)
        indices = np.random.permutation(total_samples)
        split_idx = int(train_ratio * total_samples)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
    
    def downsample_by_step(self, x, target_len, dim=1):
        """
        按间隔采样降低分辨率
        
        参数:
            x: 原始张量，形状 [样本数, (时间步), 原始长度]
            target_len: 目标空间长度（如4096, 2048等）
            dim: 需要下采样的维度（输入是1D用dim=1，输出含时间步用dim=2）
        
        返回:
            下采样后的张量
        """
        original_len = x.shape[dim]
        step = original_len // target_len  # 计算采样间隔
        # 生成采样索引（从0开始，每隔step取一个点）
        indices = torch.arange(0, original_len, step, device=x.device)[:target_len]
        # 按索引采样（保持其他维度不变）
        return torch.index_select(x, dim=dim, index=indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取原始索引
        original_idx = self.indices[idx]
        
        # 输入数据（已下采样，形状 [target_len]）
        a = self.inputs[original_idx]
        
        x = self.outputs[original_idx]
        
        return a.float(), x.float()  # 确保数据类型一致
    
    def get_full_data(self):
        """返回整个数据集（测试时使用，已下采样）"""
        x_full = self.inputs[self.indices]  # 形状 [N_test, target_len]
        y_full = self.outputs[self.indices]  # 形状 [N_test, target_len]
        return x_full, y_full


class AdvectionDataset(Dataset):
    def __init__(self, 
                 file_path, 
                 train_ratio=0.9, 
                 mode='train',
                 target_len=1024):  # 新增参数：目标长度（如4096,2048,512等，None表示不采样）
        # 加载原始数据
        with h5py.File(file_path, 'r') as f:
            u = np.array(f['tensor'][:], dtype=np.float32)  # (50, 64, 64, 5000)
        # 读取输入输出数据（根据实际字段名调整）
        self.inputs = torch.from_numpy(u[0])    # 原始形状 [4, 8192]
        self.outputs = torch.from_numpy(u[-1])   # 假设原始形状 [4, 201, 8192]（根据实际调整）
        
        # 下采样处理（如果指定了目标长度）
        self.target_len = target_len
        if target_len is not None:
            self.inputs = self.downsample_by_step(self.inputs, target_len)
            # 对outputs的空间维度下采样（假设outputs形状为[样本数, 时间步, 空间长度]）
            self.outputs = self.downsample_by_step(self.outputs, target_len, dim=-1)
        
        # 数据集划分
        total_samples = len(self.inputs)
        indices = np.random.permutation(total_samples)
        split_idx = int(train_ratio * total_samples)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
    
    def downsample_by_step(self, x, target_len, dim=1):
        """
        按间隔采样降低分辨率
        
        参数:
            x: 原始张量，形状 [样本数, (时间步), 原始长度]
            target_len: 目标空间长度（如4096, 2048等）
            dim: 需要下采样的维度（输入是1D用dim=1，输出含时间步用dim=2）
        
        返回:
            下采样后的张量
        """
        original_len = x.shape[dim]
        step = original_len // target_len  # 计算采样间隔
        # 生成采样索引（从0开始，每隔step取一个点）
        indices = torch.arange(0, original_len, step, device=x.device)[:target_len]
        # 按索引采样（保持其他维度不变）
        return torch.index_select(x, dim=dim, index=indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取原始索引
        original_idx = self.indices[idx]
        
        # 输入数据（已下采样，形状 [target_len]）
        a = self.inputs[original_idx]
        
        # 输出数据（已下采样，形状 [201, target_len]）
        x = self.outputs[original_idx]
        
        return a.float(), x.float()  # 确保数据类型一致
    
    def get_full_data(self):
        """返回整个数据集（测试时使用，已下采样）"""
        x_full = self.inputs[self.indices]  # 形状 [N_test, target_len]
        y_full = self.outputs[self.indices]  # 形状 [N_test, 201, target_len]
        return x_full, y_full


class darcyDataset(Dataset):
    def __init__(self, file_path, train_ratio=1.0, test_ratio = 0.05,mode='train', target_size=None):
        # 加载原始数据
        reader = MatReader(file_path, to_cuda=False)
        
        # 读取输入输出数据（假设形状为 [样本数, 原始分辨率, 原始分辨率]）
        self.inputs = reader.read_field('coeff')   # 形状 [n, 421, 421]
        self.outputs = reader.read_field('sol')    # 形状 [n, 421, 421]
        
        # 下采样处理（如果指定了目标分辨率）
        self.target_size = target_size
        if target_size is not None:
            # 对输入和输出的两个空间维度同时下采样
            self.inputs = self.downsample_2d(self.inputs, target_size)
            self.outputs = self.downsample_2d(self.outputs, target_size)
        
        # 数据集划分
        total_samples = len(self.inputs)
        indices = np.random.permutation(total_samples)
        split_idx = int(train_ratio * total_samples)
        test_idx = int(test_ratio * total_samples)
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[:test_idx]

    def downsample_2d(self, x, target_size):
        """
        对二维数据的两个空间维度同时下采样，处理非整除情况
        
        参数:
            x: 原始张量，形状 [样本数, original_size, original_size]
            target_size: 目标分辨率
        
        返回:
            下采样后的张量，形状 [样本数, target_size, target_size]
        """
        original_size = x.shape[1]
        
        # 计算采样步长（使用浮点除法以获得更精确的间隔）
        step = (original_size - 1) / (target_size - 1)
        
        # 生成均匀分布的采样索引（确保覆盖整个范围）
        indices = torch.linspace(0, original_size - 1, target_size, dtype=torch.long, device=x.device)
        
        # 按索引采样（先对行，再对列）
        x_downsampled = torch.index_select(x, dim=1, index=indices)
        x_downsampled = torch.index_select(x_downsampled, dim=2, index=indices)
        
        return x_downsampled

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取原始索引
        original_idx = self.indices[idx]
        
        # 输入数据（已下采样）
        a = self.inputs[original_idx]
        
        # 输出数据（已下采样）
        x = self.outputs[original_idx]
        
        return a.float(), x.float()

    def get_full_data(self):
        """返回整个数据集（测试时使用，已下采样）"""
        x_full = self.inputs[self.indices]
        y_full = self.outputs[self.indices]
        return x_full, y_full


class DiffusionDataset(Dataset):
    def __init__(self, 
                 file_path, 
                 train_ratio=0.9, 
                 mode='train'):
        # 加载HDF5文件中的数据
        with h5py.File(file_path, 'r') as f:
            # 获取所有数据键（假设为'0000'到'0999'等格式）
            self.keys = sorted([key for key in f.keys() if key.isdigit()])
            # 验证数据数量
            assert len(self.keys) > 0, "未找到有效的数据项"
            
            # 读取第一个数据项来确定形状
            sample_key = self.keys[0]
            sample_data = f[sample_key]['data'][:]  # shape (101, 128, 128, 2)
            
            # 初始化存储输入和输出的列表
            self.inputs = []  # t=0时刻的数据
            self.outputs = []  # t=-1（最后一个时刻，即t=100）的数据
            
            # 遍历所有数据项，提取t=0和t=-1时刻的数据
            for key in self.keys:
                data = np.array(f[key]['data'][:], dtype=np.float32)  # (101, 128, 128, 2)
                
                # 输入：t=0时刻的数据
                input_t0 = data[0]  # (128, 128, 2)
                self.inputs.append(input_t0)
                
                # 输出：t=-1（最后一个时刻）的数据
                output_t_last = data[-1]  # (128, 128, 2)
                self.outputs.append(output_t_last)
        
        self.inputs = torch.from_numpy(np.array(self.inputs))
        self.outputs = torch.from_numpy(np.array(self.outputs))
        
        # 数据集划分
        total_samples = len(self.keys)
        indices = np.random.permutation(total_samples)
        split_idx = int(train_ratio * total_samples)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:  # 验证或测试模式
            self.indices = indices[split_idx:]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取原始索引
        original_idx = self.indices[idx]
        
        # 输入数据：t=0时刻 (2, 128, 128)
        input_data = self.inputs[original_idx]
        
        # 输出数据：t=-1时刻 (2, 128, 128)
        output_data = self.outputs[original_idx]
        
        return input_data.float(), output_data.float()
    
    def get_full_data(self):
        """返回整个数据集（测试时使用）"""
        input_full = self.inputs[self.indices]  # 形状 [N, 2, 128, 128]
        output_full = self.outputs[self.indices]  # 形状 [N, 2, 128, 128]
        return input_full, output_full
class MyDataset(Dataset):
    def __init__(self, equation, dt=0.01, N=10000):
        """
        初始化数据集
        :param equation: 用于生成数据的 Poisson 方程实例
        :param dt: 时间步长
        :param N: 样本数
        """
        self.eq = equation
        self.dt = dt
        self.N = N
        self.a, self.x = self.eq.sample(dt=self.dt, N=self.N)  # 生成样本数据

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        根据索引获取样本
        :param idx: 样本索引
        :return: (a, x) 对
        """
        a_sample = self.a[idx]
        x_sample = self.x[idx]
        return torch.from_numpy(a_sample), torch.from_numpy(x_sample)
class MyDataset2(Dataset):
    def __init__(self, a,x):
        """
        初始化数据集
        :param equation: 用于生成数据的 Poisson 方程实例
        :param dt: 时间步长
        :param N: 样本数
        """
        self.a, self.x = a,x  # 生成样本数据

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        """
        根据索引获取样本
        :param idx: 样本索引
        :return: (a, x) 对
        """
        a_sample = self.a[idx]
        x_sample = self.x[idx]
        return a_sample, x_sample
    


class MyDataset_ns(Dataset):
    def __init__(self, 
                 file_path: str,
                 input_steps: int = 10,
                 pred_steps: int = 10,
                 train: bool = True,
                 train_ratio: float = 0.9,
                 seed: int = 42):
        """
        Navier-Stokes 时间序列预测数据集
        参数说明:
            file_path: HDF5文件路径
            input_steps: 输入时间步数
            pred_steps: 预测时间步数
            train: 是否为训练集
            train_ratio: 训练集比例
            seed: 随机种子
        """
        # 读取原始数据
        with h5py.File(file_path, 'r') as f:
            u = np.array(f['u'][()], dtype=np.float32)  # (50, 64, 64, 5000)
        
        # 调整维度顺序 (样本数, 时间步, 空间X, 空间Y)
        u_processed = u.transpose(3, 0, 1, 2)  # (5000, 50, 64, 64)
        
        # 分割输入输出时间窗口
        self.x = u_processed[:, :input_steps, :, :]          # (5000, 10, 64, 64)
        self.y = u_processed[:, input_steps:input_steps+pred_steps, :, :]  # (5000, 10, 64, 64)
        
        # 数据集分割
        np.random.seed(seed)
        indices = np.random.permutation(self.x.shape[0])
        split_idx = int(len(indices) * train_ratio)
        
        self.indices = indices[:split_idx] if train else indices[split_idx:]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_sample = torch.from_numpy(self.x[real_idx])  # (10, 64, 64)
        y_sample = torch.from_numpy(self.y[real_idx])  # (10, 64, 64)
        
        # 添加通道维度 (适配CNN输入)
        return x_sample, y_sample # (10, 1, 64, 64)
    def get_full_data(self):
        """返回整个数据集（测试时使用）"""
        x_full = torch.from_numpy(self.x[self.indices])  # (N_test, 10, 64, 64)
        y_full = torch.from_numpy(self.y[self.indices])  # (N_test, 10, 64, 64)
        return x_full, y_full

# 使用示例 -------------------------------------------------------------------
if __name__ == "__main__":
    # 初始化数据集
    train_dataset = MyDataset_ns(
        file_path='/home/luotian.ding/myproject/rectified_flow/data/ns/ns_V1e-3_N5000_T50.mat',
        input_steps=10,
        pred_steps=10,
        train=True
    )
    
    test_dataset = MyDataset_ns(
        file_path='/home/luotian.ding/myproject/rectified_flow/data/ns/ns_V1e-3_N5000_T50.mat',
        input_steps=10,
        pred_steps=10,
        train=False
    )
    
    # 创建DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 验证数据形状
    sample_x, sample_y = next(iter(train_loader))
    print("输入数据维度:", sample_x.shape)   # 预期: [32, 10, 1, 64, 64]
    print("输出数据维度:", sample_y.shape)   # 预期: [32, 10, 1, 64, 64]




class DarcyDataset2(Dataset):
    def __init__(self,
                 data_path=None,
                 inverse_problem=False,
                 normalizer_x=None,
                 normalization=True,
                 renormalization=False,
                 subsample_attn: int = 15,
                 subsample_nodes: int = 1,
                 subsample_inverse: int = 1,
                 subsample_method='nearest',
                 subsample_method_inverse='average',
                 n_krylov: int = 3,
                 uniform: bool = True,
                 train_data=True,
                 train_len=0.9,
                 valid_len=0.0,
                 online_features=False,
                 sparse_edge=False,
                 return_edge=False,
                 return_lap_only=True,
                 return_boundary=True,
                 noise=0,
                 random_state=1127802):
        '''
        PyTorch dataset overhauled for the Darcy flow data from Li et al 2020
        https://github.com/zongyi-li/fourier_neural_operator

        FNO2d network size: 2368001
        original grid size = 421*421
        Laplacian size = (421//subsample) * (421//subsample)
        subsample = 2, 3, 5, 6, 7, 10, 12, 15

        Uniform (update Apr 2021): 
        node: node features, coefficient a, (N, n, n, 1)
        pos: x, y coords, (n_s*n_s, 2)
        grid: fine grid, x- and y- coords (n, n, 2)
        targets: solution u_h, (N, n, n, 1)
        targets_grad: grad_h u_h, (N, n, n, 2)
        edge: Laplacian and krylov, (S, n_sub, n_sub, n_krylov) stored as list of sparse matrices

        '''
        self.data_path = data_path
        self.n_grid_fine = 421  # finest resolution along x-, y- dim
        self.subsample_attn = subsample_attn  # subsampling for attn
        self.subsample_nodes = subsample_nodes  # subsampling for input and output
        self.subsample_inverse = subsample_inverse  # subsample for inverse output
        self.subsample_method = subsample_method  # 'interp' or 'nearest'
        self.subsample_method_inverse = subsample_method_inverse  # 'interp' or 'average'
        # sampling resolution for nodes subsampling along x-, y-
        self.n_grid = int(((self.n_grid_fine - 1)/self.subsample_attn) + 1)
        self.h = 1/self.n_grid_fine
        self.uniform = uniform  # not used
        self.train_data = train_data
        self.train_len = train_len
        self.valid_len = valid_len
        self.n_krylov = n_krylov
        self.return_edge = return_edge
        self.sparse_edge = sparse_edge
        self.normalization = normalization
        self.normalizer_x = normalizer_x  # the normalizer for data
        self.renormalization = renormalization  # whether to add I/h to edge
        self.inverse_problem = inverse_problem  # if True, target is coeff
        self.return_boundary = return_boundary
        # if True, boundary nodes included in grid (-1, n, n) else, (-1, n-2, n-2)
        self.return_lap_only = return_lap_only  # do not return diffusion matrix
        # whether to generate edge features on the go for __getitem__()
        self.online_features = online_features
        self.edge_features = None
        self.mass_features = None
        self.random_state = random_state
        self.eps = 1e-8
        self.noise = noise
        if self.data_path is not None:
            self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        
        get_seed(self.random_state, printout=False)
        with timer(f"Loading {self.data_path.split('/')[-1]}"):
            #try:
            data = loadmat(self.data_path)
            a = data['coeff']  # (N, n, n)
            u = data['sol']  # (N, n, n)
            del data  
            gc.collect()
#             except FileNotFoundError as e:
#                 print(r"Please download the dataset from https://github.com/zongyi-li/fourier_neural_operator and put untar them in the data folder.")

        data_len = self.get_data_len(len(a))

        if self.train_data:
            a, u = a[:data_len], u[:data_len]
        else:
            a, u = a[-data_len:], u[-data_len:]
        self.n_samples = len(a)

        nodes, targets, targets_grad = self.get_data(a, u)
        

        self.coeff = nodes  # un-transformed coeffs
        # pos and elem are already downsampled
        self.pos, self.elem = self.get_grid(self.n_grid)
        self.pos_fine = self.get_grid(self.n_grid_fine,
                                      subsample=self.subsample_nodes,
                                      return_elem=False,
                                      return_boundary=self.return_boundary)

        if self.return_edge and not self.online_features:
            self.edge_features, self.mass_features = self.get_edge(a)
            # (n_s*n_s, 2),  list of a list of csr matrices, list of csr matrix

        # TODO: if processing inverse before normalizer, shapes will be unmatched
        if self.inverse_problem:
            nodes, targets = targets, nodes
            if self.subsample_inverse is not None and self.subsample_inverse > 1:
                n_grid = int(((self.n_grid_fine - 1)/self.subsample_nodes) + 1)
                n_grid_inv = int(
                    ((self.n_grid_fine - 1)/self.subsample_inverse) + 1)
                pos_inv = self.get_grid(n_grid_inv,
                                        return_elem=False,
                                        return_boundary=self.return_boundary)
                if self.subsample_method_inverse == 'average':
                    s_inv = self.subsample_inverse//self.subsample_nodes
                    
                    targets = pooling_2d(targets.squeeze(),
                                         kernel_size=(s_inv, s_inv), padding=True)
                elif self.subsample_method_inverse == 'interp':
                    targets = self.get_interp2d(targets.squeeze(),
                                                n_grid,
                                                n_grid_inv)
                elif self.subsample_method_inverse is None:
                    targets = targets.squeeze()
                self.pos_fine = pos_inv
                targets = targets[..., None]

        if self.train_data and self.normalization:
            self.normalizer_x = UnitGaussianNormalizer()
            self.normalizer_y = UnitGaussianNormalizer()
            nodes = self.normalizer_x.fit_transform(nodes)

            if self.return_boundary:
                _ = self.normalizer_y.fit_transform(x=targets)
            else:
                _ = self.normalizer_y.fit_transform(
                    x=targets[:, 1:-1, 1:-1, :])
        elif self.normalization:
            nodes = self.normalizer_x.transform(nodes)

        if self.noise > 0:
            nodes += self.noise*np.random.randn(*nodes.shape)

        self.node_features = nodes  # (N, n, n, 1)
        self.target = targets  # (N, n, n, 1) of (N, n_s, n_s, 1) if inverse
        self.target_grad = targets_grad  # (N, n, n, 2)

    def get_data_len(self, len_data):
        if self.train_data:
            if self.train_len <= 1:
                train_len = int(self.train_len*len_data)
            elif 1 < self.train_len <= len_data:
                train_len = self.train_len
            else:
                train_len = int(0.8*len_data)
            return train_len
        else:
            if self.valid_len <= 1:
                valid_len = int(self.valid_len*len_data)
            elif 1 < self.valid_len <= len_data:
                valid_len = self.valid_len
            else:
                valid_len = int(0.1*len_data)
            return valid_len

    def get_data(self, a, u):
        # get full resolution data
        # input is (N, 421, 421)
        batch_size = a.shape[0]
        n_grid_fine = self.n_grid_fine
        s = self.subsample_nodes
        n = int(((n_grid_fine - 1)/s) + 1)
                   
                   
        targets = u  # (N, 421, 421)
        if not self.inverse_problem:
            targets_gradx, targets_grady = self.central_diff(
                targets, self.h)  # (N, n_f, n_f)
            targets_gradx = targets_gradx[:, ::s, ::s]
            targets_grady = targets_grady[:, ::s, ::s]

            # targets_gradx = targets_gradx.unsqueeze(-1)
            # targets_grady = targets_grady.unsqueeze(-1)
            # cat = torch.cat if torch.is_tensor(targets_gradx) else np.concatenate
            # grad = cat([targets_gradx, targets_grady], -1) # (N, 2*(n_grid-2)**2)
            targets_grad = np.stack(
                [targets_gradx, targets_grady], axis=-1)  # (N, n, n, 2)
        else:
            targets_grad = np.zeros((batch_size, 1, 1, 2))

        targets = targets[:, ::s, ::s].reshape(batch_size, n, n, 1)

        if s > 1 and self.subsample_method == 'nearest':
            nodes = a[:, ::s, ::s].reshape(batch_size, n, n, 1)
        elif s > 1 and self.subsample_method in ['interp', 'linear', 'average']:
            nodes = pooling_2d(a,
                               kernel_size=(s, s),
                               padding=True).reshape(batch_size, n, n, 1)
        else:
            nodes = a.reshape(batch_size, n, n, 1)

        return nodes, targets, targets_grad

    @staticmethod
    def central_diff(x, h, padding=True):
        # x: (batch, n, n)
        # b = x.shape[0]
        if padding:
            x = np.pad(x, ((0, 0), (1, 1), (1, 1)),
                       'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (x[:, d:, s:-s] - x[:, :-d, s:-s])/d  # (N, S_x, S_y)
        grad_y = (x[:, s:-s, d:] - x[:, s:-s, :-d])/d  # (N, S_x, S_y)

        return grad_x/h, grad_y/h

    @staticmethod
    def get_grid(n_grid, subsample=1, return_elem=True, return_boundary=True):
        x = np.linspace(0, 1, n_grid)
        y = np.linspace(0, 1, n_grid)
        x, y = np.meshgrid(x, y)
        nx = ny = n_grid  # uniform grid
        s = subsample

        if return_elem:
            grid = np.c_[x.ravel(), y.ravel()]  # (n_node, 2)
            elem = []
            for j in range(ny-1):
                for i in range(nx-1):
                    a = i + j*nx
                    b = (i+1) + j*nx
                    d = i + (j+1)*nx
                    c = (i+1) + (j+1)*nx
                    elem += [[a, c, d], [b, c, a]]

            elem = np.asarray(elem, dtype=np.int32)
            return grid, elem
        else:
            if return_boundary:
                x = x[::s, ::s]
                y = y[::s, ::s]
            else:
                x = x[::s, ::s][1:-1, 1:-1]
                y = y[::s, ::s][1:-1, 1:-1]
            grid = np.stack([x, y], axis=-1)
            return grid

    @staticmethod
    def get_grad_tri(grid, elem):
        ve1 = grid[elem[:, 2]]-grid[elem[:, 1]]
        ve2 = grid[elem[:, 0]]-grid[elem[:, 2]]
        ve3 = grid[elem[:, 1]]-grid[elem[:, 0]]
        area = 0.5*(-ve3[:, 0]*ve2[:, 1] + ve3[:, 1]*ve2[:, 0])
        # (# elem, 2-dim vector, 3 vertices)
        Dlambda = np.zeros((len(elem), 2, 3))

        Dlambda[..., 2] = np.c_[-ve3[:, 1]/(2*area), ve3[:, 0]/(2*area)]
        Dlambda[..., 0] = np.c_[-ve1[:, 1]/(2*area), ve1[:, 0]/(2*area)]
        Dlambda[..., 1] = np.c_[-ve2[:, 1]/(2*area), ve2[:, 0]/(2*area)]
        return Dlambda, area

    @staticmethod
    def get_norm_matrix(A, weight=None):
        '''
        A has to be csr
        '''
        if weight is not None:
            A += diags(weight)
        D = diags(A.diagonal()**(-0.5))
        A = (D.dot(A)).dot(D)
        return A

    @staticmethod
    def get_scaler_sizes(n_f, n_c, scale_factor=True):
        factor = np.sqrt(n_c/n_f)
        factor = np.round(factor, 4)
        last_digit = float(str(factor)[-1])
        factor = np.round(factor, 3)
        if last_digit < 5:
            factor += 5e-3
        factor = int(factor/5e-3 + 5e-1 ) * 5e-3
        down_factor = (factor, factor)
        n_m = round(n_f*factor)-1
        up_size = ((n_m, n_m), (n_f, n_f))
        down_size = ((n_m, n_m), (n_c, n_c))
        if scale_factor:
            return down_factor, up_size
        else:
            return down_size, up_size

    @staticmethod
    def get_interp2d(x, n_f, n_c):
        '''
        interpolate (N, n_f, n_f) to (N, n_c, n_c)
        '''
        x_f, y_f = np.linspace(0, 1, n_f), np.linspace(0, 1, n_f)
        x_c, y_c = np.linspace(0, 1, n_c), np.linspace(0, 1, n_c)
        x_interp = []
        for i in range(len(x)):
            xi_interp = interpolate.interp2d(x_f, y_f, x[i])
            x_interp.append(xi_interp(x_c, y_c))
        return np.stack(x_interp, axis=0)

    def get_edge(self, a):
        '''
        Modified from Long Chen's iFEM routine in 2D
        https://github.com/lyc102/ifem
        a: diffusion constant for all, not downsampled
        (x,y), elements downsampled if applicable
        '''
        grid, elem = self.pos, self.elem
        Dphi, area = self.get_grad_tri(grid, elem)
        ks = self.subsample_attn//self.subsample_nodes
        n_samples = len(a)
        online_features = self.online_features
        # (-1, n, n) -> (-1, n_s, n_s)
        a = pooling_2d(a, kernel_size=(ks, ks), padding=True)
        n = len(grid)
        edges = []
        mass = []

        with tqdm(total=n_samples, disable=online_features) as pbar:
            for i in range(n_samples):
                # diffusion coeff
                K = a[i].reshape(-1)  # (n, n) -> (n^2, )
                K_to_elem = K[elem].mean(axis=1)
                # stiffness matrix
                A = csr_matrix((n, n))
                M = csr_matrix((n, n))
                Lap = csr_matrix((n, n))
                for i in range(3):
                    for j in range(3):
                        # $A_{ij}|_{\tau} = \int_{\tau}K\nabla \phi_i\cdot \nabla \phi_j dxdy$
                        Lapij = area*(Dphi[..., i]*Dphi[..., j]).sum(axis=-1)
                        Aij = K_to_elem*Lapij
                        Mij = area*((i == j)+1)/12
                        A += csr_matrix((Aij,
                                        (elem[:, i], elem[:, j])), shape=(n, n))
                        Lap += csr_matrix((Lapij,
                                          (elem[:, i], elem[:, j])), shape=(n, n))
                        M += csr_matrix((Mij,
                                        (elem[:, i], elem[:, j])), shape=(n, n))

                w = np.asarray(
                    M.sum(axis=-1))*self.n_grid**2 if self.renormalization else None
                A, Lap = [self.get_norm_matrix(m, weight=w) for m in (A, Lap)]
                # A, Lap = [self.get_norm_matrix(m) for m in (A, Lap)]
                edge = [A]
                Laps = [Lap]
                if self.n_krylov > 1:
                    for i in range(1, self.n_krylov):
                        edge.append(A.dot(edge[i-1]))
                        Laps.append(Lap.dot(Laps[i-1]))

                edge = Laps if self.return_lap_only else edge + Laps

                edges.append(edge)
                mass.append(M)
                pbar.update()

        return edges, mass

    def __getitem__(self, index):
        '''
        Outputs:
            - pos: x-, y- coords
            - a: diffusion coeff
            - target: solution
            - target_grad, gradient of solution
        '''
        pos_dim = 2
        pos = self.pos[:, :pos_dim]
        if self.return_edge and not self.online_features:
            edges = self.edge_features[index]

            if self.sparse_edge:
                edges = [csr_to_sparse(m) for m in edges]
            else:
                edges = np.asarray(
                    [m.toarray().astype(np.float32) for m in edges])

            edge_features = torch.from_numpy(edges.transpose(1, 2, 0))

            mass = self.mass_features[index].toarray().astype(np.float32)
            mass_features = torch.from_numpy(mass)
        elif self.return_edge and self.online_features:
            a = self.node_features[index].reshape(
                1, self.n_grid_fine, self.n_grid_fine, -1)
            if self.normalization:
                a = self.normalizer_x.inverse_transform(a)
            a = a[..., 0]
            edges, mass = self.get_edge(a)
            edges = np.asarray([m.toarray().astype(np.float32)
                               for m in edges[0]])
            edge_features = torch.from_numpy(edges.transpose(1, 2, 0))

            mass = mass[0].toarray().astype(np.float32)
            mass_features = torch.from_numpy(mass)
        else:
            edge_features = torch.tensor([1.0])
            mass_features = torch.tensor([1.0])
        if self.subsample_attn < 5:
            pos = torch.tensor([1.0])
        else:
            pos = torch.from_numpy(pos)

        grid = torch.from_numpy(self.pos_fine)
        node_features = torch.from_numpy(self.node_features[index])
        coeff = torch.from_numpy(self.coeff[index])
        target = torch.from_numpy(self.target[index])
        target_grad = torch.from_numpy(self.target_grad[index])

        return dict(node=node_features.float(),
                    coeff=coeff.float(),
                    pos=pos.float(),
                    grid=grid.float(),
                    edge=edge_features.float(),
                    mass=mass_features.float(),
                    target=target.float(),
                    target_grad=target_grad.float())


class BurgersDataset2(Dataset):
    def __init__(self, subsample: int,
                 n_grid_fine=2**13,
                 viscosity: float = 0.1,
                 n_krylov: int = 2,
                 smoother=None,
                 uniform: bool = True,
                 train_data=True,
                 train_portion=0.9,
                 valid_portion=0.1,
                 super_resolution: int = 1,
                 data_path=None,
                 online_features=False,
                 return_edge=False,
                 renormalization=False,
                 return_distance_features=True,
                 return_mass_features=False,
                 return_downsample_grid: bool = True,
                 random_sampling=False,
                 random_state=1127802,
                 debug=False):
        r'''
        PyTorch dataset overhauled for Burger's data from Li et al 2020
        https://github.com/zongyi-li/fourier_neural_operator

        FNO1d network size n_hidden = 64, 16 modes: 549569
        Benchmark: error \approx 1e-2 after 100 epochs after subsampling to 512
        subsampling = 2**3 #subsampling rate
        h = 2**13 // sub #total grid size divided by the subsampling rate

        Periodic BC
        Uniform: 
            - node: f sampled at uniform grid
            - pos: uniform grid, pos encoding 
            - targets: [targets, targets derivative]
        '''
        self.data_path = data_path
        if subsample > 1:
            assert subsample % 2 == 0
        self.subsample = subsample
        self.super_resolution = super_resolution
        self.supsample = subsample//super_resolution
        self.n_grid_fine = n_grid_fine  # finest resolution
        self.n_grid = n_grid_fine // subsample
        self.h = 1/n_grid_fine
        self.uniform = uniform
        self.return_downsample_grid = return_downsample_grid
        self.train_data = train_data
        self.train_portion = train_portion
        self.valid_portion = valid_portion
        self.n_krylov = n_krylov
        self.viscosity = viscosity
        self.smoother = smoother
        self.random_sampling = random_sampling
        self.random_state = random_state
        self.online_features = online_features
        self.edge_features = None  # failsafe
        self.mass_features = None
        self.return_edge = return_edge
        # whether to do Kipf-Weiling renormalization A += I
        self.renormalization = renormalization
        self.return_mass_features = return_mass_features
        self.return_distance_features = return_distance_features
        self.debug = debug
        self._set_seed()
        self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        data = 'train' if self.train_data else 'valid'
        with timer(f"Loading {self.data_path.split('/')[-1]} for {data}."):
            data = loadmat(self.data_path)
            x_data = data['a']
            y_data = data['u']
            del data
            gc.collect()

        train_len, valid_len = self.train_test_split(len(x_data))

        if self.train_data:
            x_data, y_data = x_data[:train_len], y_data[:train_len]
        else:
            x_data, y_data = x_data[-valid_len:], y_data[-valid_len:]

        self.n_samples = len(x_data)

        get_data = self.get_uniform_data if self.uniform else self.get_nonuniform_data
        grid, grid_fine, nodes, targets = get_data(x_data, y_data)

        if not self.online_features and self.return_edge:
            edge_features = []
            mass_features = []
            for i in tqdm(range(self.n_samples)):
                edge, mass = self.get_edge(grid)
                edge_features.append(edge)
                mass_features.append(mass)
            self.edge_features = np.asarray(edge_features, dtype=np.float32)
            self.mass_features = np.asarray(mass_features, dtype=np.float32)
         ##PINN  do not expand dimension
        self.node_features = nodes[..., None] if nodes.ndim == 2 else nodes
        # self.pos = np.c_[grid[...,None], grid[...,None]] #(N, S, 2)
        self.n_features = self.node_features.shape[-1]
        self.pos = grid[..., None] if self.uniform else grid
        self.pos_fine = grid_fine[..., None]
        self.target = targets[..., None] if targets.ndim == 2 else targets

    def _set_seed(self):
        s = self.random_state
        os.environ['PYTHONHASHSEED'] = str(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def get_uniform_data(self, x_data, y_data):
        targets = y_data
        targets_diff = self.central_diff(targets, self.h)

        if self.super_resolution >= 2:
            nodes = x_data[:, ::self.supsample]
            targets = targets[:, ::self.supsample]
            targets_diff = targets_diff[:, ::self.supsample]
        else:
            nodes = x_data[:, ::self.subsample]
            targets = targets[:, ::self.subsample]
            targets_diff = targets_diff[:, ::self.subsample]

        targets = np.stack([targets, targets_diff], axis=2)
        grid = np.linspace(0, 1, self.n_grid)  # subsampled
        # grids = np.asarray([grid for _ in range(self.n_samples)])
        grid_fine = np.linspace(0, 1, self.n_grid_fine//self.supsample)

        return grid, grid_fine, nodes, targets

    @staticmethod
    def central_diff(x, h):
        # x: (batch, seq_len, feats)
        # x: (seq_len, feats)
        # x: (seq_len, )
        # padding is one before feeding to this function
        # TODO: change the pad to u (u padded with the other end b/c periodicity)
        if x.ndim == 1:
            pad_0, pad_1 = x[-2], x[1]  # pad periodic
            x = np.c_[pad_0, x, pad_1]  # left center right
            x_diff = (x[2:] - x[:-2])/2
        elif x.ndim == 2:
            pad_0, pad_1 = x[:, -2], x[:, 1]
            x = np.c_[pad_0, x, pad_1]
            x_diff = (x[:, 2:] - x[:, :-2])/2
            # pad = np.zeros(x_diff.shape[0])

        # return np.c_[pad, x_diff/h, pad]
        return x_diff/h

    @staticmethod
    def laplacian_1d(x, h):
        # standard [1, -2, 1] stencil
        x_lap = (x[1:-1] - x[:-2]) - (x[2:] - x[1:-1])
        # return np.r_[0, x_lap/h**2, 0]
        return x_lap/h**2

    def train_test_split(self, len_data):
        # TODO: change this to random split
        if self.train_portion <= 1:
            train_len = int(self.train_portion*len_data)
        elif 1 < self.train_portion <= len_data:
            train_len = self.train_portion
        else:
            train_len = int(0.8*len_data)

        if self.valid_portion <= 1:
            valid_len = int(self.valid_portion*len_data)
        elif 1 < self.valid_portion <= len_data:
            valid_len = self.valid_portion
        else:
            valid_len = int(0.1*len_data)
        try:
            assert train_len <= len_data - valid_len, \
                f"train len {train_len} be non-overlapping with test len {valid_len}"
        except AssertionError as err:
            print(err)
        return train_len, valid_len

    def get_nonuniform_data(self, x_data, y_data):
        '''
        generate non-uniform data for each sample
        same number of points, but uniformly random chosen
        out:
            - x_data assimilated by u first.
        deprecated
        '''
        targets_u = y_data
        x0, xn = 0, 1
        n_nodes = self.n_grid
        h = self.h
        grid_u = np.linspace(x0, xn, n_nodes)
        grids_u = np.asarray([grid_u for _ in range(self.n_samples)])
        pad_0, pad_n = y_data[:, 0], y_data[:, -1]
        targets_u_diff = self.central_diff(np.c_[pad_0, y_data, pad_n], h)

        grids, nodes, targets, targets_diff = [], [], [], []
        # mesh, function, [solution, solution uniform], derivative
        for i in tqdm(range(self.n_samples)):
            node_fine = x_data[i]  # all 8192 function value f
            _node_fine = np.r_[0, node_fine, 0]  # padding
            node_fine_diff = self.central_diff(_node_fine, h)
            node_fine_lap = self.laplacian_1d(_node_fine, h)
            sampling_density = np.sqrt(
                node_fine_diff**2 + self.viscosity*node_fine_lap**2)
            # sampling_density = np.sqrt(node_fine_lap**2)
            sampling_density = sampling_density[1:-1]
            sampling_density /= sampling_density.sum()

            grid, ix, ix_fine = self.get_grid(sampling_density)

            grids.append(grid)
            node = x_data[i, ix]  # coarse grid
            target, target_diff = y_data[i,
                                         ix_fine], targets_u_diff[i, ix_fine]

            nodes.append(node)
            targets.append(target)
            targets_diff.append(target_diff)

        if self.super_resolution >= 2:
            nodes_u = x_data[:, ::self.supsample]
            targets_u = y_data[:, ::self.supsample]
            targets_u_diff = targets_u_diff[:, ::self.supsample]
        else:
            nodes_u = x_data[:, ::self.subsample]
            targets_u = y_data[:, ::self.subsample]
            targets_u_diff = targets_u_diff[:, ::self.subsample]

        grids = np.asarray(grids)
        nodes = np.asarray(nodes)
        targets = np.asarray(targets)
        targets_diff = np.asarray(targets_diff)
        '''
        target[...,0] = target_u: u on uniform grid
        target[...,1] = targets_u_diff: Du on uniform grid
        target[...,2] = target: u
        target[...,3] = targets_diff: Du

        last dim is for reference
        target[...,4] = nodes_u: f on uniform grid
        '''
        targets = np.stack(
            [targets_u, targets_u_diff, targets, targets_diff, nodes_u], axis=2)

        return grids, grids_u, nodes, targets

    def get_grid(self, sampling_density):
        x0, xn = 0, 1
        sampling_density = None if self.random_sampling else sampling_density
        ix_fine = np.sort(np.random.choice(range(1, self.n_grid_fine-1),
                                           size=self.super_resolution*self.n_grid-2,
                                           replace=False, p=sampling_density))
        ix_fine = np.r_[0, ix_fine, self.n_grid_fine-1]
        ix = ix_fine[::self.super_resolution]
        grid = self.h*ix[1:-1]
        grid = np.r_[x0, grid, xn]
        ix = np.r_[0, ix[1:-1], self.n_grid_fine-1]

        return grid, ix, ix_fine

    def get_edge(self, grid):
        '''
        generate edge features
        '''
        # mass lumping
        weight = np.asarray([self.n_grid for _ in range(
            self.n_grid)]) if self.renormalization else None
        edge = get_laplacian_1d(grid,
                                normalize=True,
                                weight=weight,
                                smoother=self.smoother).toarray().astype(np.float32)
        if self.n_krylov > 1:
            dim = edge.shape  # (S, S)
            dim += (self.n_krylov, )  # (S, S, N_krylov)
            edges = np.zeros(dim)
            edges[..., 0] = edge
            for i in range(1, self.n_krylov):
                edges[..., i] = edge.dot(edges[..., i-1])
        else:
            edges = edge[..., None]

        distance = get_distance_matrix(grid, graph=False)
        mass = get_mass_1d(grid, normalize=False).toarray().astype(np.float32)

        if self.return_mass_features and self.return_distance_features:
            mass = mass[..., None]
            edges = np.concatenate([edges, distance, mass], axis=2)
        elif self.return_distance_features:
            edges = np.concatenate([edges, distance], axis=2)
        return edges, mass

    def __getitem__(self, index):
        '''
        Outputs:
            - pos: coords
            - x: rhs
            - target: solution
        '''
        pos_dim = 1 if self.uniform else 2
        if self.uniform:
            pos = self.pos[:, :pos_dim]
        else:
            pos = self.pos[index, :, :pos_dim]
        grid = pos[..., 0]
        if self.online_features:
            edge = get_laplacian_1d(
                grid, normalize=True).toarray().astype(np.float32)
            if self.n_krylov > 1:
                dim = edge.shape
                dim += (self.n_krylov, )
                edges = np.zeros(dim)
                edges[..., 0] = edge
                for i in range(1, self.n_krylov):
                    edges[..., i] = edge.dot(edges[..., i-1])
            else:
                edges = edge[..., np.newaxis]

            distance = get_distance_matrix(grid, graph=False)
            mass = get_mass_1d(
                grid, normalize=False).toarray().astype(np.float32)
            if self.return_distance_features:
                edges = np.concatenate([edges, distance], axis=2)
            edge_features = torch.from_numpy(edges)
            mass = torch.from_numpy(mass)
        elif not self.online_features and self.return_edge:
            edge_features = torch.from_numpy(self.edge_features[index])
            mass = torch.from_numpy(self.mass_features[index])
        else:
            edge_features = torch.tensor([1.0])
            mass = torch.tensor([1.0])

        if self.return_downsample_grid:
            self.pos_fine = grid[..., None]
        pos_fine = torch.from_numpy(self.pos_fine)
        pos = torch.from_numpy(grid[..., None])
        node_features = torch.from_numpy(self.node_features[index])
        target = torch.from_numpy(self.target[index])
        return dict(node=node_features.float(),
                    pos=pos.float(),
                    grid=pos_fine.float(),
                    edge=edge_features.float(),
                    mass=mass.float(),
                    target=target.float(),)