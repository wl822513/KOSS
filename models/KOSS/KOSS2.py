import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pscan import pscan
from .S4 import FiLMBlock  # 适用于包内相对导入



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
    expand_factor: int = 1  # E in paper/comments
    d_conv: int = 4

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
        # self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state + config.d_state, bias=False)
        self.x_proj = nn.Sequential(
            nn.Linear(config.d_inner, config.d_inner*3),  # 第一层，全连接
            nn.ReLU(),  # 激活函数
            nn.Linear(config.d_inner*3, config.d_inner),  # 第二层，全连接
            nn.ReLU(),
            nn.Linear(config.d_inner, config.dt_rank + config.d_state + config.d_inner)  # 输出层
        )

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
            nn.Linear(config.d_inner, config.d_inner*3),  # 第一层，全连接
            nn.ReLU(),  # 激活函数
            nn.Linear(config.d_inner*3, config.d_inner),  # 第二层，全连接
            nn.ReLU(),
            nn.Linear(config.d_inner, config.d_inner)  # 输出层
        )

        # 初始化FFD层
        # self.FFD = FFDEU(kernel_size=self.config.d_conv)
        self.FDU = FDU(kernel_size=self.config.d_conv)
        self.SSOL = SSOL()

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
        delta, C, K = torch.split(deltaCK, [self.config.dt_rank,  self.config.d_state,  self.config.d_inner], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, ED)

        if self.config.pscan:
            y = self.seg_selective_scan(x, delta, A, C, D, K)
        else:
            y = self.selective_scan_seq(x, delta, Aprime, Bprime, C, D)
        return y

    def selective_scan(self, x, delta, A, B, C, D, Kdy_dt):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        # y : (B, L, ED)
        _, L, _ = x.shape
        segment_size = 20

        for start in range(0, L, segment_size):
            hs = []

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B32, L5, ED128, N16)

        deltaB = delta.unsqueeze(-1) * B  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan(deltaA, BX) + Kdy_dt.unsqueeze(-1)  #+ 0.5 * FiLMBlock(x)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def seg_selective_scan(self, x, delta, A, C, D, K):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        # y : (B, L, ED)

        _, L, _ = x.shape
        segment_size = 20
        error_K = 0

        # Calculate deltaA
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)

        # To store all the intermediate results for each segment
        hs = []
        # h = K[:, :segment_size, :]
        h = torch.zeros_like(x[:, :segment_size, :])
        # Loop over the segments
        for start in range(0, L, segment_size):
            end = min(start + segment_size, L)  # Ensure the last segment does not exceed L

            # Slice the input x and other matrices within the current segment
            x_segment = x[:, start:end, :]  # (B, segment_size, ED)
            delta_segment = delta[:, start:end, :]  # (B, segment_size, ED)
            deltaA_segment = deltaA[:, start:end, :]  # (B, segment_size, ED, N)
            C_segment = C[:, start:end, :]  # (B, segment_size, N)
            K_segment = K[:, start:end, :]  # (B, segment_size, N)
            if start:
              h = (h[:, :x_segment.shape[1], :] @ C_segment.unsqueeze(-1)).squeeze()

            # Calculate the error term for the Kalman gain
            error_K = self.K_proj(F.softsign((x_segment - h) ** 2)) + error_K    # Compute Kalman gain
            Kdy_dt = error_K * self.FDU(x_segment, delta_segment)  # Corrected Kalman term

            # Calculate Aprime, Bprime via SSOL function
            Aprime, Bprime = self.SSOL(deltaA_segment, C_segment, error_K)
            Bprimex = Bprime * x_segment.unsqueeze(-1) + Kdy_dt.unsqueeze(-1)
            # Update h using pscan
            h = pscan(Aprime, Bprimex)  # pscan process and Kalman update

            # Append h_segment to hs
            hs.append(h)

        # After the loop, stack all segments into one tensor (B, L, ED, N)
        hs = torch.cat(hs, dim=1)  # (B, L, ED, N) concatenation
        # Update y based on hs
        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class SSOL(nn.Module):
    """
    具有选择性的状态空间最优化层 (Selective State Optimization Layer)
    """

    def __init__(self, activation="leaky_relu"):
        super(SSOL, self).__init__()
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)  # 负值保持一定比例
        else:
            self.activation = None  # 不使用激活函数

    def forward(self, A, C, K):
        """
        :param A: 状态转移矩阵 (B, L, D)
        :param C: 观测矩阵 (B, L, D)
        :param K: 卡尔曼增益 (B, L, D)
        :return: Aprime (优化后的状态转移矩阵), Bprime (优化后的控制矩阵)
        """
        KCA = (K.unsqueeze(-1) * C.unsqueeze(-2)) * A  # 计算 KCA
        A_KCA = A - KCA  # 计算 A - KCA
        Aprime = A_KCA + A_KCA * (K.unsqueeze(-1) * C.unsqueeze(-2))  # 计算 Aprime
        Bprime = KCA - A  # 计算 Bprime

        # 应用激活函数
        # if self.activation:
        #     Bprime = self.activation(Bprime)

        return Aprime, Bprime


class FFDEU(nn.Module):
    def __init__(self, kernel_size=3):
        """
        频率有限差分增强层 (Frequency Finite Difference Enhanced Unit)
        :param kernel_size: 平滑滤波器的窗口大小
        """
        super(FFDEU, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.ones(1, 1, kernel_size) / kernel_size  # 生成均匀平滑滤波器

    def forward(self, x):
        """
        前向传播：先平滑信号，再计算有限差分（数值导数）
        :param x: 输入信号, 形状 (batch_size, len, feature)
        :return:  数值差分 `dy`
        """
        batch_size, seq_len, feature_dim = x.shape  # 解析输入形状

        # **Step 1: 信号平滑**
        x = x.permute(0, 2, 1)  # (batch_size, feature_dim, seq_len) 适配 Conv1d
        x_padded = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode='reflect')  # 反射填充
        x_smooth = F.conv1d(x_padded, self.kernel.to(x.device).expand(feature_dim, 1, -1), groups=feature_dim)[:, :, :seq_len]
        x_smooth = x_smooth.permute(0, 2, 1)  # (batch_size, seq_len, feature_dim)

        # **Step 2: 计算有限差分**
        dy = x_smooth[:, 1:] - x_smooth[:, :-1]  # (batch_size, seq_len-1, feature_dim)
        dy = torch.cat([dy, dy[:, -1:].clone()], dim=1)  # 维度对齐 (batch_size, seq_len, feature_dim)

        return dy  # 返回平滑信号 + 数值导数


class FDU(nn.Module):
    def __init__(self, kernel_size=3, sigma=0.5):
        """
        傅里叶求导层 (Fourier Differentiation Unit)
        :param kernel_size: 高斯平滑滤波器窗口大小
        :param sigma: 高斯滤波器标准差
        """
        super(FDU, self).__init__()
        self.kernel_size = nn.Parameter(torch.tensor(kernel_size, dtype=torch.float32),
                                        requires_grad=True)  # 使kernel_size可学习
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=True)  # 使sigma可学习

    def gaussian_kernel_1d(self, kernel_size, sigma):
        """ 生成 1D 高斯核 """
        x = torch.arange(kernel_size) - kernel_size // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()  # 归一化
        return kernel.view(1, 1, -1)  # 调整形状 (1, 1, kernel_size)

    def forward(self, x, delta_t):
        """
        前向传播：在频域计算平滑滤波 + 物理导数 dY/dt
        :param x: 输入信号, 形状 (batch_size, seq_len, feature_dim)
        :param delta_t: 采样时间间隔, 形状 (batch_size, seq_len, feature_dim)
        :return:  真实物理导数 dY/dt
        """
        batch_size, seq_len, feature_dim = x.shape  # 获取输入形状
        x = x.permute(0, 2, 1)  # 调整为 (batch_size, feature_dim, seq_len)
        delta_t = delta_t.permute(0, 2, 1)  # (batch_size, feature_dim, seq_len)

        # **Step 1: 计算傅里叶变换**
        X_f = torch.fft.fft(x, dim=-1)  # 计算 FFT (batch_size, feature_dim, seq_len)

        # **Step 2: 重新计算高斯平滑滤波器的傅里叶变换**
        kernel_size_int = int(torch.round(self.kernel_size).item())  # 确保核大小为整数
        kernel = self.gaussian_kernel_1d(kernel_size_int, self.sigma.item())

        kernel_padded = torch.zeros(seq_len, device=x.device)  # 先填充为0
        kernel_padded[:kernel_size_int] = kernel.to(x.device).squeeze()  # 将高斯核填入
        kernel_f = torch.fft.fft(kernel_padded, dim=-1)  # 计算 FFT 变换

        X_smooth_f = X_f * kernel_f  # 频域平滑

        # **Step 3: 计算角频率 ω**
        freq_base = torch.fft.fftfreq(seq_len, d=1.0, device=x.device).view(1, 1, -1)  # 频率索引
        omega = 2j * torch.pi * freq_base / (delta_t + 1e-6)  # 计算 ω，防止除 0

        # **Step 4: 频域求导**
        dX_f = omega * X_smooth_f  # 频域求导

        # **Step 5: 逆傅里叶变换**
        dy_dt = torch.fft.ifft(dX_f, dim=-1).real  # 取实部

        dy_dt = dy_dt.permute(0, 2, 1)  # 调整回 (batch_size, seq_len, feature_dim)

        return dy_dt   # 返回 dy/dt（真实物理导数）