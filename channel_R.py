import torch
import torch.nn as nn

class Channel(nn.Module):
    def __init__(self, channel_type='Rayleigh', snr=10.0, eps=1e-9, rx_ant=1):
        super().__init__()
        self.channel_type = channel_type
        self.snr = snr
        self.eps = eps
        self.rx_ant = rx_ant

    def forward(self, x):
        # x: [B, C, H, W]
        # 1. 如果没有 SNR，直接透传 (无噪声)
        if self.snr is None:
            return x, torch.ones_like(x)

        # 2. 计算信号功率
        # p_signal: [B, 1, 1, 1]
        p_signal = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        snr_lin = 10.0 ** (self.snr / 10.0)
        p_noise = p_signal / snr_lin
        
        # noise_std: [B, 1, 1, 1]
        noise_std = p_noise.sqrt()

        if self.channel_type == 'AWGN':
            # AWGN 模式下不需要多天线逻辑，保持原样
            noise = torch.randn_like(x) * noise_std
            return x + noise, torch.ones_like(x)

        elif self.channel_type == 'Rayleigh':
            B, C, H, W = x.shape
            
            # === 修复点 ===
            # 将 noise_std 从 [B, 1, 1, 1] 变成 [B, 1, 1, 1, 1]
            # 这样才能和 5维的 h 正确广播
            noise_std_5d = noise_std.unsqueeze(1) 

            # h: [B, rx_ant, C, H, W]
            h = torch.randn(B, self.rx_ant, C, H, W, device=x.device)
            
            # noise: [B, rx_ant, C, H, W]
            # 现在维度对齐了: [B, rx, C, H, W] * [B, 1, 1, 1, 1] -> OK
            noise = torch.randn(B, self.rx_ant, C, H, W, device=x.device) * noise_std_5d

            # x_expanded: [B, 1, C, H, W]
            x_expanded = x.unsqueeze(1)

            # y: [B, rx_ant, C, H, W]
            y = h * x_expanded + noise

            # === MRC 合并 ===
            # 分子: sum(y * h) -> [B, C, H, W]
            numerator = (y * h).sum(dim=1)
            # 分母: sum(h^2) -> [B, C, H, W]
            denominator = h.pow(2).sum(dim=1).clamp_min(self.eps)
            
            y_mrc = numerator / denominator
            h_eff = denominator

            return y_mrc, h_eff

        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")