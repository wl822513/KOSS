# coding=utf-8

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import special as ss
from .utils.op import transition
import opt_einsum as oe
contract = oe.contract

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT, self).__init__()
        self.N = N
        A, B = transition('lmu', N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device))
        self.register_buffer('B', torch.Tensor(B).to(device))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))

    def forward(self):  # torch.Size([128, 1, 1]) -
        return self.A, self.B

    def reconstruct(self, c):
        a = (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


################################################################


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes1, compression=0, ratio=0.5, mode_type=0):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.compression = compression
        self.ratio = ratio
        self.mode_type = mode_type
        if self.mode_type == 1:
            # modes2=modes1-10000
            modes2 = modes1
            self.modes2 = min(modes2, seq_len // 2)
            self.index0 = list(range(0, int(ratio * min(seq_len // 2, modes2))))
            self.index1 = list(range(len(self.index0), self.modes2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[:min(seq_len // 2, self.modes2) - int(ratio * min(seq_len // 2, modes2))]
            self.index = self.index0 + self.index1
            self.index.sort()
        elif self.mode_type > 1:
            # modes2=modes1-1000
            modes2 = modes1
            self.modes2 = min(modes2, seq_len // 2)
            self.index = list(range(0, seq_len // 2))
            np.random.shuffle(self.index)
            self.index = self.index[:self.modes2]
        else:
            self.modes2 = min(modes1, seq_len // 2)
            self.index = list(range(0, self.modes2))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.cfloat))
        if self.compression > 0:
            # print('compressed version')
            self.weights0 = nn.Parameter(self.scale * torch.rand(in_channels, self.compression, dtype=torch.cfloat))
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(self.compression, self.compression, len(self.index), dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(self.compression, out_channels, dtype=torch.cfloat))
        # print(self.modes2)

    def forward(self, x):
        B, H, E, N = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        # pdb.set_trace()
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        # a = x_ft[:, :,:, :self.modes1]
        # out_ft[:, :,:, :self.modes1] = torch.einsum("bjix,iox->bjox", a, self.weights1)
        if self.compression == 0:
            if self.modes1 > 1000:
                for wi, i in enumerate(self.index):
                    # print(self.index)
                    # print(out_ft.shape)
                    out_ft[:, :, :, i] = torch.einsum('bji,io->bjo', (x_ft[:, :, :, i], self.weights1[:, :, wi]))
            else:
                a = x_ft[:, :, :, :self.modes2]
                out_ft[:, :, :, :self.modes2] = torch.einsum("bjix,iox->bjox", a, self.weights1)
        elif self.compression > 0:
            a = x_ft[:, :, :, :self.modes2]
            a = torch.einsum("bjix,ih->bjhx", a, self.weights0)
            a = torch.einsum("bjhx,hkx->bjkx", a, self.weights1)
            out_ft[:, :, :, :self.modes2] = torch.einsum("bjkx,ko->bjox", a, self.weights2)
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs, N=512, N2=32):
        super(Model, self).__init__()
        self.configs = configs
        # self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.seq_len_all = self.seq_len + self.label_len

        self.output_attention = configs.output_attention
        self.layers = configs.e_layers
        self.modes1 = min(configs.modes1, self.pred_len // 2)
        # self.modes1 = 32
        self.enc_in = configs.enc_in
        self.proj = False
        self.e_layers = configs.e_layers
        self.mode_type = configs.mode_type
        if self.configs.ours:
            # b, s, f means b, f
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

        if configs.enc_in > 1000:
            self.proj = True
            self.conv1 = nn.Conv1d(configs.enc_in, configs.d_model, 1)
            self.conv2 = nn.Conv1d(configs.d_model, configs.dec_in, 1)
            self.d_model = configs.d_model
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.d_model))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.d_model))
        if self.configs.ab == 2:
            self.multiscale = [1, 2, 4]
            # self.multiscale = [1]
            self.window_size = [2]  # 第4个维度
            self.legts = nn.ModuleList(
                [HiPPO_LegT(N=n, dt=1. / self.pred_len / i) for n in self.window_size for i in self.multiscale])
            self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n,
                                                             seq_len=min(self.pred_len, self.seq_len),
                                                             modes1=configs.modes1, compression=configs.version,
                                                             ratio=configs.ratio, mode_type=self.mode_type) for n in
                                              self.window_size for _ in range(len(self.multiscale))])
            self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

    def forward(self, x_enc):
        # decomp init

        if self.configs.ab == 2:
            return_data = [x_enc]
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1, 2))
                x_enc = x_enc.transpose(1, 2)
            if self.configs.ours:
                means = x_enc.mean(1, keepdim=True).detach()
                # mean
                x_enc = x_enc - means
                # var
                stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                x_enc /= stdev
                # affine
                x_enc = x_enc * self.affine_weight + self.affine_bias
            B, L, E = x_enc.shape
            seq_len = self.seq_len
            x_decs = []
            jump_dist = 0
            for i in range(0, len(self.multiscale) * len(self.window_size)):
                x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
                x_in = x_enc[:, -x_in_len:]
                legt = self.legts[i]
                x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
                out1 = self.spec_conv_1[i](x_in_c)
                if self.seq_len >= self.pred_len:
                    x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
                else:
                    x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
                x_dec = x_dec_c @ (legt.eval_matrix[-self.pred_len:, :].T)
                x_decs += [x_dec]
            return_data.append(x_in_c)
            return_data.append(out1)
            x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0, 2, 1)
            if self.configs.ours:
                x_dec = x_dec - self.affine_bias
                x_dec = x_dec / (self.affine_weight + 1e-10)
                x_dec = x_dec * stdev
                x_dec = x_dec + means
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1, 2))
                x_dec = x_dec.transpose(1, 2)
            return_data.append(x_dec)
        if self.output_attention:
            # return x_dec, return_data
          return x_dec, out1.permute(0, 3, 1, 2)
        else:
            return x_dec, None  # [B, L, D]



class Configs(object):
    ab = 2
    modes1 = 8
    seq_len = 192
    label_len = 0
    pred_len = 192
    output_attention = True
    enc_in = 128
    dec_in = 8
    d_model = 4
    embed = 'timeF'
    dropout = 0.05
    freq = 'h'
    factor = 1
    n_heads = 8
    d_ff = 16
    e_layers = 2
    d_layers = 1
    moving_avg = 25
    c_out = 1
    activation = 'gelu'
    wavelet = 0
    ours = False
    version = 16
    ratio = 1
    mode_type = 0


def FiLMBlock(enc):
    configs = Configs()
    model = Model(configs).to(device)
    _, out1 = model.forward(enc)
    return out1


if __name__ == '__main__':
    configs = Configs()
    model = Model(configs).to(device)

    enc = torch.randn([32, configs.seq_len, configs.enc_in]).cuda()
    out, out1 = model.forward(enc)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print('model size', count_parameters(model) / (1024 * 1024))
    print('input shape', enc.shape)
    print('output shape', out.shape)
