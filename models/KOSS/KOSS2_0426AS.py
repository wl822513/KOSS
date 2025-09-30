import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pscan import pscan
from .S4 import FiLMBlock  # 适用于包内相对导入
from .S4 import HiPPO_LegT



"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison.

- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""

# 配置类 这个类设置了KOSS模型的配置参数，包括维度、初始化方法和是否使用偏置。
@dataclass
class KOSSConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 2  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 2

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True  # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


# Mamba模型
class KOSS(nn.Module):
    def __init__(self, config: KOSSConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        # self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):  # 将输入通过所有层进行处理。
        # x : (B, L, D)
        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)
        # x = self.norm_f(x)

        return x

    def step(self, x, caches):  # 在每一层中处理输入并更新缓存
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)
        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


# 残差快
class ResidualBlock(nn.Module):
    def __init__(self, config: KOSSConfig):  # 创建 KOSSBlock 和归一化层 (RMSNorm)。
        super().__init__()

        self.mixer = KOSSBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):  # 应用归一化、MambaBlock 和残差连接。
        # x : (B, L, D)
        # output : (B, L, D)
        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):  # 在 MambaBlock 中应用缓存和残差连接。
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)
        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


# KOSS块
class KOSSBlock(nn.Module):
    def __init__(self, config: KOSSConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)
        self.conv2d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        # projects x to input-dependent Δ, C，K
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state, bias=False)
        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # self.dt_proj.bias._no_reinit = True
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        self.K_proj = nn.Sequential(
            nn.Linear(config.d_inner, config.d_inner * 3, bias=True),
            nn.ReLU(),
            nn.Linear(config.d_inner * 3, config.d_inner, bias=False),
            nn.ReLU(),
            nn.Linear(config.d_inner, config.d_inner, bias=False),
            nn.LayerNorm(config.d_inner)  # ⭐加在最后面，归一化输出
        )

        # self.k_alpha = nn.Parameter(torch.tensor(0.5))  # 初始化成0.5，代表一开始old和new各占一半
        self.k_alpha = nn.Parameter(torch.zeros(1, 1, config.d_inner))  # 初始是0

        # 初始化FDU层
        self.FDU = FDU(sigma=0.5)
        self.SSOL = SSOL()
        self.segment_num = 8

        # 使用 MLP 替代 FDU
        self.MLP = nn.Sequential(
            nn.Linear(config.d_inner, config.d_inner),
            nn.ReLU()
        )

        # 使用 LSTM 替代 FDU
        self.LSTM = nn.LSTM(
            input_size=config.d_inner,
            hidden_size=config.d_inner,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.Attention = nn.MultiheadAttention(
            embed_dim=config.d_inner,
            num_heads=4,
            batch_first=True
        )

        self.S4SS = HiPPO_LegT(config.d_inner)

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)
        _, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  # depthwise convolution over time, with a short filter
        x = self.conv2d(x)[:, :, :L]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z  #(32,5,128)
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x):
        # x : (B, L, ED)
        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        # TODO remove .float()

        deltaCK = self.x_proj(x)  # (B, L, dt_rank+ED*N+N)
        # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, C = torch.split(deltaCK, [self.config.dt_rank,  self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, ED)

        if self.config.pscan:
            y = self.seg_selective_scan(x, delta, A, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, C, D)
        return y
    def seg_selective_scan(self, x, delta, A, C, D):
        """
        :param x: Input (B, L, ED)
        :param delta: Zero-order hold discretization time interval (B, L, ED)
        :param A: State transition matrix (ED, N)
        :param C: Observation matrix (B, L, N)
        :param D: Residual modulation (ED,)
        :return: Output (B, L, ED)
        """
        B, L, ED = x.shape
        # segment_size = self.segment_size  # You can make this a class attribute if needed
        segment_size = max(8, round(L / self.segment_num))

        # Precompute deltaA
        # deltaA = torch.exp((delta.unsqueeze(-1) * A))  # (B, L, ED, N)

        hs = []
        yh = torch.zeros(B, segment_size, ED, device=x.device, dtype=x.dtype)  # Initial hidden state
        K = torch.zeros(B, segment_size, ED, device=x.device, dtype=x.dtype)  # Initial Kalman gain

        for start in range(0, L, segment_size):
            end = min(start + segment_size, L)
            seg_len = end - start

            x_seg = x[:, start:end, :]  # (B, seg_len, ED)
            delta_seg = delta[:, start:end, :]  # (B, seg_len, ED)
            # deltaA_seg = deltaA[:, start:end, :]  # (B, seg_len, ED, N)
            C_seg = C[:, start:end, :]  # (B, seg_len, N)

            # Update Kalman gain
            K_error = F.softsign((x_seg - yh[:, :seg_len, :]) ** 2)  # Error-based modulation
            K_new = self.K_proj(K_error)  # (B, seg_len, ED)
            alpha = self.k_alpha.clamp(0.01, 0.99)
            K = (1 - alpha) * K[:, :seg_len, :] + alpha * K_new

            # Compute K * d(x)/dt
            deltax = self.FDU(x_seg, delta_seg)
            # deltax = self.MLP(x_seg)
            # deltax, _ = self.LSTM(x_seg)
            # deltax, _ = self.Attention(x_seg, x_seg, x_seg)
            Kdy_dt = K * deltax  # (B, seg_len, ED)

            # Compute A_k and B_k
            A_k, B_k = self.SSOL(A, C_seg, K)  # (B, seg_len, ED, N), (B, seg_len, ED, N)
            deltaA_k = torch.exp(delta_seg.unsqueeze(-1) * A_k)  # (B, seg_len, ED, N) discrete parameter A_k
            deltaB_k = delta_seg.unsqueeze(-1) * B_k  # (B, seg_len, ED, N) discrete parameter B_k

            # Compute B_k * x + K * d(x)/dt
            deltaB_kx = deltaB_k * x_seg.unsqueeze(-1) + Kdy_dt.unsqueeze(-1)  # (B, seg_len, ED, N)
            # deltaB_kx = deltaB_k * x_seg.unsqueeze(-1)

            # pscan update
            h = pscan(deltaA_k, deltaB_kx)  # (B, seg_len, ED, N)
            yh = (h @ C_seg.unsqueeze(-1)).squeeze(-1)  # (B, L, ED)
            yh = yh + D.unsqueeze(0).unsqueeze(0) * x_seg  # Add residual connection
            hs.append(h)

        hs = torch.cat(hs, dim=1)  # (B, L, ED)
        # Final output
        y = (hs @ C.unsqueeze(-1)).squeeze(-1)  # (B, L, ED)
        y = y + D.unsqueeze(0).unsqueeze(0) * x  # Add residual connection

        return y

class SSOL(nn.Module):
    """
    Selective State Optimization Layer
    Implements:
        A_k = (A - K C A)(I + K C)
        B_k = - (A - K C A) K
    """

    def __init__(self, activation=None):
        super(SSOL, self).__init__()
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1)
        }
        self.activation = activations.get(activation, None)

    def forward(self, A, C, K):
        """
        A: (ED, N)
        C: (B, L, N)
        K: (B, L, ED)
        Returns:
            A_k: (B, L, ED, N)
            B_k: (B, L, ED, N)
        """
        # Expand dims to match for broadcasting
        K_ = K.unsqueeze(-1)        # (B, L, ED, 1)
        C_ = C.unsqueeze(2)         # (B, L, 1, N)

        # Compute KC and KCA
        KC = K_ * C_                # (B, L, ED, N)
        KCA = KC * A                # (B, L, ED, N)

        A_minus_KCA = A - KCA       # (B, L, ED, N)
        A_k = A_minus_KCA * (1 + KC)  # (B, L, ED, N)
        B_k = -A_minus_KCA * K_       # (B, L, ED, N)

        # Optional activation (usually not recommended here)
        # if self.activation:
        #     B_k = self.activation(B_k)

        return A_k, B_k


class FDU(nn.Module):
    def __init__(self, sigma=0.5, eps=1e-5):
        """
        Fourier Differentiation Unit (FDU): 用频域卷积与频率加权计算连续导数。
        :param sigma: 控制平滑程度的高斯滤波器参数（可学习）
        :param eps: 防止除0或NaN出现的小数值
        """
        super(FDU, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))  # 可学习的高斯滤波器宽度
        self.eps = eps

    def forward(self, x, delta_t):
        """
        :param x: 输入信号 (B, T, D)
        :param delta_t: 采样间隔 (B, T, D) or scalar-like
        :return: 物理导数估计 dy/dt (B, T, D)
        """
        B, T, D = x.shape  # (batch, time, dim)

        # ---- reshape to (B, D, T) for FFT ----
        x = x.permute(0, 2, 1)  # (B, D, T)
        delta_t = delta_t.permute(0, 2, 1) if delta_t.ndim == 3 else delta_t.view(1, 1, T)

        # ---- FFT ----
        X_f = torch.fft.fft(x, dim=-1)  # 复数频谱 (B, D, T)

        # ---- 构造频率基础 ----
        freqs = torch.fft.fftfreq(T, d=1.0).to(x.device)  # (T,)
        freqs = freqs.view(1, 1, T)  # (1, 1, T)

        # ---- 高斯滤波器 in frequency domain ----
        gaussian_filter = torch.exp(- (freqs ** 2) * (self.sigma ** 2))  # (1, 1, T)

        # ---- 平滑频谱 ----
        X_smooth_f = X_f * gaussian_filter  # (B, D, T)

        # ---- ω = 2πf / Δt ----
        omega = 2j * torch.pi * freqs / (delta_t + self.eps)  # 复频率 (B, D, T)

        # ---- 频域求导 ----
        dX_f = omega * X_smooth_f  # (B, D, T)

        # ---- iFFT 回时域 ----
        dx_dt = torch.fft.ifft(dX_f, dim=-1).real  # 只取实部 (B, D, T)

        # ---- 转回原来的形状 ----
        dx_dt = dx_dt.permute(0, 2, 1)  # (B, T, D)

        # ---- 数值检查（可选）----
        if torch.isnan(dx_dt).any():
            print("[Warning] NaN detected in FDU output!")

        return dx_dt


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output