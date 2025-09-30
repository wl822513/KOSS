import torch.nn as nn
from KOSS.KOSS2_0426AS import KOSS, KOSSConfig


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class KOSSModel(nn.Module):
    def __init__(self, input_features, KOSS_hidden_size, KOSS_num_layers,
                 output_time_steps, output_features):
        super(KOSSModel, self).__init__()
        self.output_time_steps = output_time_steps  # 96
        self.output_features = output_features  # D'

        # Mamba (KOSS) 层配置
        self.config = KOSSConfig(d_model=KOSS_hidden_size, n_layers=KOSS_num_layers)

        # 输入映射层，将输入特征映射到 KOSS 隐藏维度
        self.input_linear = nn.Linear(input_features, KOSS_hidden_size)

        # KOSS 层
        self.KOSS = KOSS(self.config)

        # 残差连接
        self.residual = nn.Linear(KOSS_hidden_size, KOSS_hidden_size)

        # 输出映射层，将 KOSS 隐藏维度映射到目标特征维度
        self.output_linear = nn.Linear(KOSS_hidden_size, output_features)

    def forward(self, x):
        """
        x: (B, 96, D)  ->  KOSS  ->  (B, 96, D')
        """
        # 线性变换输入
        x = self.input_linear(x)  # (B, 96, KOSS_hidden_size)

        # KOSS 层处理整个序列
        KOSS_out = self.KOSS(x)  # (B, 96, KOSS_hidden_size)

        # 线性变换到目标特征维度
        out = self.output_linear(KOSS_out)  # (B, 96, D')

        return out  # (B, 96, D')


# 参数配置 LSTF
paramsKOSS = {
    "KOSS_hidden_size": 64,               # Mamba 隐藏层大小
    "KOSS_num_layers": 2,                 # Mamba 层数
    "input_time_steps": 720,                 # 输入时间步数
    "output_time_steps": 720,                # 输出时间步数
    "input_features": 5,                   # 输入特征的数量
    "output_features": 2,                  # 输出特征的数量
    "num_epochs": 100,                      # 训练的总 epochs 数
    "period": 4.5,                         # 训练周期
    "batch_size": 32,                      # 批大小
    "learning_rate": 0.001,                # 学习率
}




