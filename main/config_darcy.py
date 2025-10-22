import torch
from script.ode_data import Eq1, WaveEquation, PoissonEquation, HeatEquation
from rectified.rectified_flow import RectFlow
from scorenet.scorenet import CNN_add,CNN_ns,MLP2d_Darcy,GNN_Darcy
# 参数设置
para_path = "/data5/store1/dlt/rectified_flow/modelpara/"
save_path = "/data5/store1/dlt/rectified_flow/result/darcyfinal1/"
eq_T = 1.0 # 方程采样总长
N = 10000  # a数量
niter = 500
target_len=85
lr = 1e-3
batch_size = 4 # 减少批处理大小
T = 1.0
eq_dt = 0.01
rf_dt = 1.0
h_dim = 64
train = 0
rf = RectFlow()
eq = PoissonEquation()
# 模型相关
# scorenet_model_class = "MLP2d_Darcy"  #  "MLP1d", "MLP2d", "FNO", "CNN"
scorenet_model_class = "FNO2d"  #  "MLP1d", "MLP2d", "FNO", "CNN"

# GPU 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 文件名自动生成
model_name = scorenet_model_class.lower() + ".pth"
