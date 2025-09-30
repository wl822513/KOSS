import torch.nn as nn
from KOSS.KOSS import KOSS, KOSSConfig


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class KOSSModel(nn.Module):
    def __init__(self, input_features, KOSS_hidden_size, KOSS_num_layers,
                 output_time_steps, output_features, period):
        super(KOSSModel, self).__init__()
        self.output_time_steps = output_time_steps
        self.output_features = output_features
        self.period = period

        # Mamba layer configuration
        self.config = KOSSConfig(d_model=KOSS_hidden_size, n_layers=KOSS_num_layers)

        self.linear = nn.Linear(input_features, KOSS_hidden_size)

        # Mamba layer
        self.KOSS = KOSS(self.config)

        # Add residual connection layer
        self.residual = nn.Linear(KOSS_hidden_size, KOSS_hidden_size)

        # Output layer to predict future time steps data
        self.fc = nn.Linear(KOSS_hidden_size, output_time_steps * output_features)

    def forward(self, x):
        # Process Mamba layer
        KOSS_out = self.KOSS(self.linear(x))

        # Add residual connection
        residual_out = self.residual(KOSS_out)
        KOSS_out = KOSS_out + residual_out

        # Generate future time step predictions  (B, L, N)
        out = self.fc(KOSS_out[:, -1, :])  # Use only the last time step's output
        out = out.view(-1, self.output_time_steps, self.output_features)

        return out


# 参数配置
paramsKOSS = {
    "KOSS_hidden_size": 64,               # Mamba 隐藏层大小
    "KOSS_num_layers": 2,                 # Mamba 层数
    "input_time_steps": 8,                 # 输入时间步数
    "output_time_steps": 4,                # 输出时间步数
    "input_features": 5,                   # 输入特征的数量
    "output_features": 2,                  # 输出特征的数量
    "num_epochs": 10,                      # 训练的总 epochs 数
    "period": 4.5,                         # 训练周期
    "batch_size": 32,                      # 批大小
    "learning_rate": 0.001,                # 学习率
}




