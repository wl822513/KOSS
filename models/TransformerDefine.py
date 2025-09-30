import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class TransformerModel(nn.Module):
    def __init__(self, input_features, transformer_hidden_size, num_transformer_layers, num_heads,
                 input_time_steps, output_time_steps, output_features):
        super(TransformerModel, self).__init__()
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps
        self.output_features = output_features

        # 线性映射层，将输入特征映射到 transformer_hidden_size 维度
        self.input_fc = nn.Linear(input_features, transformer_hidden_size)

        # Transformer 编码层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_transformer_layers)

        # 输出层
        self.fc = nn.Linear(transformer_hidden_size, output_time_steps * output_features)

    def forward(self, x):
        # 调用处理函数
        # 线性映射层
        transformed_input = self.input_fc(x)
        # Transformer 层
        transformer_out = self.transformer_encoder(transformed_input)

        # 取最后一个时间步的输出
        out = self.fc(transformer_out[:, -1, :])
        out = out.view(-1, self.output_time_steps, self.output_features)
        return out

    @staticmethod
    def Cumulative_Sequence(lstm_out, x):
        """
        处理 LSTM 输出，计算累积序列并构建累积矩阵。

        参数:
        lstm_out (torch.Tensor): LSTM 的输出，形状为 (batch_size, time_steps, features)
        x (torch.Tensor): 原始输入数据，形状为 (batch_size, time_steps, features)

        返回:
        torch.Tensor: 处理后的累积矩阵
        """
        # 获取输入的形状
        batch_size, time_steps, features = lstm_out.shape

        # 获取所有样本的最后一个时间步输出 (跳过第一个样本)
        last_sequences = lstm_out[1:, -1, :]  # 形状 (batch_size - 1, features)

        # 获取 lstm_out 的第一个样本
        first_sample = lstm_out[0]  # 形状 (time_steps, features)

        # 将第一个样本和 last_sequences 组合
        combined_sequence = torch.cat((first_sample, last_sequences),
                                      dim=0)  # shape: (time_steps + batch_size - 1, features)
        # print("combined_sequence：\n", combined_sequence)

        # 复制 batch_size 次，扩展成 (batch_size, time_steps + batch_size - 1, features)
        expanded_combined_sequence = (combined_sequence.unsqueeze(0).expand(batch_size, -1, -1)
                                      .to(lstm_out.device))  # 确保在同一设备
        # print("expanded_combined_sequence：\n", expanded_combined_sequence)

        # 创建一个形状为 (batch_size, time_steps + batch_size - 1, features) 的零矩阵
        cumulative_matrix = torch.zeros((batch_size, time_steps + batch_size - 1, features), dtype=lstm_out.dtype,
                                        device=lstm_out.device)
        # 生成索引，用于填充前面的行
        for i in range(batch_size):
            cumulative_matrix[i, :time_steps + i, :] = 1  # 将前 time_steps + i 行填充为 1

        # 逐元素相乘
        result = cumulative_matrix * expanded_combined_sequence  # 按元素相乘

        return result


paramsTransformer = {
    "transformer_hidden_size": 64,      # Transformer 编码器的隐藏层大小
    "num_transformer_layers": 2,         # Transformer 编码器的层数
    "num_heads": 8,                      # Transformer 编码器的多头注意力头数
    "input_time_steps": 720,               # 输入时间步数
    "output_time_steps": 720,              # 输出时间步数
    "input_features": 6,                 # 输入特征的数量
    "output_features": 5,                # 输出特征的数量
    "num_epochs": 20,                    # 训练的总 epochs 数
    "period": 4.5,                      # 训练周期
    "batch_size": 1,                    # 批大小
    "learning_rate": 0.001,             # 学习率
}
