import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils


class LSTMModel(nn.Module):
    def __init__(self, input_features, lstm_hidden_size, lstm_num_layers,
                 input_time_steps, output_time_steps, output_features, period):
        super(LSTMModel, self).__init__()
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps
        self.output_features = output_features
        self.period = period

        # LSTM 层
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                            batch_first=True)
        # 输出层
        self.fc = nn.Linear(lstm_hidden_size, output_time_steps * output_features)

    def forward(self, x):
        # LSTM 层
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])  # 只用最后一个时间步的输出
        out = out.view(-1, self.output_time_steps, self.output_features)
        return out


paramsLSTM = {
    "input_features": 6,
    "lstm_hidden_size": 64,
    "lstm_num_layers": 2,
    "input_time_steps": 336,
    "output_time_steps": 336,
    "output_features": 5,
    "num_epochs": 20,
    "period": 4.5,
    "batch_size": 32,
    "learning_rate": 0.001,
}


