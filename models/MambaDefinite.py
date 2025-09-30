import torch
import torch.nn as nn
import numpy as np
from filterpy.kalman import KalmanFilter
from Mamba.mamba4 import Mamba, MambaConfig
from matplot_func import plot_kf_output_comparison
from functions import extract_xData, calculate_2d_msd, calculate_2d_fourier

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MambaModel(nn.Module):
    def __init__(self, input_features, mamba_hidden_size, mamba_num_layers,
                 output_time_steps, output_features, period):
        super(MambaModel, self).__init__()
        self.output_time_steps = output_time_steps
        self.output_features = output_features
        self.input_features = input_features
        self.period = period

        # Mamba layer configuration
        self.config = MambaConfig(d_model=mamba_hidden_size, n_layers=mamba_num_layers)
        self.linear = nn.Linear(input_features, mamba_hidden_size)
        # Mamba layer
        self.mamba = Mamba(self.config)

        # Add residual connection layer
        self.residual = nn.Linear(mamba_hidden_size, mamba_hidden_size)

        # Output layer to predict future time steps data
        self.fc = nn.Linear(mamba_hidden_size, output_features)

    def forward(self, x):

        # Process Mamba layer
        mamba_out = self.mamba(self.linear(x))

        # Generate future time step predictions
        out = self.fc(mamba_out)  # Use only the last time step's output

        return out


# 参数配置
paramsMamba = {
    "mamba_hidden_size": 64,               # Mamba 隐藏层大小
    "mamba_num_layers": 2,                 # Mamba 层数
    "input_time_steps": 720,                 # 输入时间步数
    "output_time_steps": 720,                # 输出时间步数
    "input_features": 11,                   # 输入特征的数量
    "output_features": 11,                  # 输出特征的数量
    "num_epochs": 100,                      # 训练的总 epochs 数
    "period": 4.5,                         # 训练周期
    "batch_size": 32,                      # 批大小
    "learning_rate": 0.0001,                # 学习率
}




