import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pscan import pscan


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 3  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 2

    dt_min: float = 0.0001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1
    dt_init_floor = 1e-4

    alpha: float = 0.4  # 控制 LSTM 在输出中的比例

    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True  # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


# LSTMGate类
class LSTMGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # 定义门控组件的参数
        self.W_i = nn.Linear(input_dim, input_dim)  # 输入门权重
        self.W_f = nn.Linear(input_dim, input_dim)  # 遗忘门权重
        self.W_o = nn.Linear(input_dim, input_dim)  # 输出门权重
        self.W_g = nn.Linear(input_dim, input_dim)  # 候选值权重

        # 定义偏置项
        self.b_i = nn.Parameter(torch.zeros(input_dim))  # 输入门偏置
        self.b_f = nn.Parameter(torch.zeros(input_dim))  # 遗忘门偏置
        self.b_o = nn.Parameter(torch.zeros(input_dim))  # 输出门偏置
        self.b_g = nn.Parameter(torch.zeros(input_dim))  # 候选值偏置

    def forward(self, x, h):
        h_prev, c_prev = h

        # 计算门控值
        i = torch.sigmoid(self.W_i(x) + self.b_i)  # 输入门
        f = torch.sigmoid(self.W_f(x) + self.b_f)  # 遗忘门
        o = torch.sigmoid(self.W_o(x) + self.b_o)  # 输出门
        g = torch.tanh(self.W_g(x) + self.b_g)  # 候选值

        # 更新状态
        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, (h, c)


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


# MambaBlock类
class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.alpha = config.alpha  # 使用 alpha 来控制 LSTM 输出的比例

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # 初始化其他参数
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # # 定义 LSTM gate 组件
        self.lstm_gate = LSTMGate(config.d_inner)

    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = F.silu(x)
        y = self.ssm(x)

        # # 检查并初始化 c_prev，区分训练和测试阶段
        # if not hasattr(self, 'c_prev') or self.c_prev.shape[0] != z.shape[0]:
        #     self.c_prev = torch.zeros_like(z)
        #
        # # 如果是训练模式，允许 c_prev 跟踪梯度
        # if self.training:
        #     self.c_prev = self.c_prev.detach()
        #
        # # 传递 LSTM 门并更新 c_prev 状态
        # _, (z, self.c_prev) = self.lstm_gate(x, (_, self.c_prev))
        z = F.silu(z)

        # output = self.alpha * (y * z) + (1 - self.alpha) * z

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x

        return y

    def step(self, x, cache):
        h, inputs = cache
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1]

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        z = F.silu(z)
        x, z = self.lstm_gate(x, z)

        output = y * z
        output = self.out_proj(output)

        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)

        BX = deltaB * (x.unsqueeze(-1))

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)

        h = deltaA * h + BX
        y = (h @ C.unsqueeze(-1)).squeeze(2)
        y = y + D * x

        return y, h.squeeze(1)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(mean + self.eps)
        return self.scale * x / std
