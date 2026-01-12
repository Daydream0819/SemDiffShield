
import torch
import torch.nn as nn
from channel_R import Channel  # 假设这是你之前的 Channel 类


# === 基础组件：残差块 (保持特征提取能力) ===
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(),  # SiLU (Swish) 在扩散模型相关任务中表现最好
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)  # 残差连接


# === 1. 编码器 (Tx): 4 -> channel_ch ===
class LatentEncoder(nn.Module):
    def __init__(self, hidden=128, channel_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            # 1. 升维提取特征
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.SiLU(),

            # 2. 深层特征处理 (加深网络可以更好地理解语义)
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),

            # 3. 压缩/映射到传输通道
            # 注意：最后不要加 Activation，因为模拟信号可以是正负任意值
            nn.Conv2d(hidden, channel_ch, 3, padding=1)
        )

    def forward(self, x):
        # x: [B, 4, 64, 64]
        # out: [B, channel_ch, 64, 64]
        return self.net(x)


# === 2. 解码器 (Rx): channel_ch -> 4 ===
class LatentDecoder(nn.Module):
    def __init__(self, hidden=128, channel_ch=16, use_csi=True):
        super().__init__()

        # 如果使用 CSI (信道状态信息)，输入通道数翻倍
        # 因为我们会把接收到的信号 y 和信道 h 拼在一起
        in_dim = channel_ch * 2 if use_csi else channel_ch

        self.net = nn.Sequential(
            # 1. 融合 信号与CSI
            nn.Conv2d(in_dim, hidden, 3, padding=1),
            nn.SiLU(),

            # 2. 强力去噪与恢复
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),  # 多加一层，解码通常比编码难

            # 3. 映射回 Latent 空间
            # 注意：SD Latent 的值域大概在 -5 到 5 之间，不要用 Sigmoid/Tanh
            nn.Conv2d(hidden, 4, 3, padding=1)
        )

    def forward(self, x):
        # x: [B, in_dim, 64, 64]
        # out: [B, 4, 64, 64]
        return self.net(x)

class DeepJSCC(nn.Module):
    def __init__(self, hidden = 256,channel_ch=16, snr=10.0, rx_ant=1,channel_type="AWGN"):
        super().__init__()
        self.channel_ch = channel_ch
        self.encoder = LatentEncoder(hidden=hidden,channel_ch=channel_ch)
        # 解码器输入通道 = 接收信号通道 + 信道状态信息(CSI)通道
        # 假设 h 和 y 维度一致 (每个feature map对应一个衰落系数)
        self.decoder = LatentDecoder(hidden=256, channel_ch=channel_ch,use_csi=True)

        # 使用修正后的 Channel，确保它返回 y 和 h
        self.channel = Channel(channel_type="AWGN", snr=snr, rx_ant=rx_ant)
        self.eps = 1e-6
        
    def forward(self, z):
        # z: [B, 4, 64, 64] (Latent)

        # ==========================
        # 1. 编码 (Source Coding)
        # ==========================
        tx_features = self.encoder(z)  # [B, 16, 64, 64]

        # ==========================
        # 2. 功率归一化 (Power Constraint)
        # ==========================
        # 计算每个样本的平均功率
        # p: [B, 1, 1, 1]
        p = tx_features.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        scale = torch.sqrt(p)

        # 发送信号 (满足功率约束 E[x^2]=1)
        tx_sig = tx_features / scale

        # ==========================
        # 3. 无线信道 (Channel)
        # ==========================
        # 这一步非常关键：
        # 我们希望 Channel 返回原始的 y 和 h，不要在 Channel 里做 MRC
        # y: [B, 16, 64, 64], h: [B, 16, 64, 64] (实数模拟复数衰落)
        rx_sig, h = self.channel(tx_sig)

        # ==========================
        # 4. 接收端预处理 (Receiver)
        # ==========================

        # --- 方案 A: 纯神经网络方法 (推荐) ---
        # 直接把 y 和 h 拼起来喂给网络。
        # 网络会自己学会 "y / h" 或者 "y * h / (h^2 + n)" 这种操作。
        # 这样可以避免人为除以 0 的风险，也能处理非线性。

        # 确保 h 的维度和 rx_sig 一致 (如果 h 是 [B,1,1,1] 需要 expand)
        if h.shape[1] != rx_sig.shape[1]:
            h = h.expand_as(rx_sig)

        # 拼接: [B, 16+16, 64, 64]
        decoder_in = torch.cat([rx_sig, h], dim=1)

        # --- 方案 B: 显式 MMSE (如果你非要用) ---
        # snr_lin = 10 ** (self.channel.snr / 10.0)
        # noise_var = 1.0 / snr_lin
        # # MMSE = (h* y) / (|h|^2 + sigma^2)
        # mmse_sig = (rx_sig * h) / (h.pow(2) + noise_var)
        # decoder_in = torch.cat([mmse_sig, h], dim=1) # 依然建议拼上 h 给网络参考

        # ==========================
        # 5. 解码 (Joint Source-Channel Decoding)
        # ==========================
        z_hat = self.decoder(decoder_in)

        # ==========================
        # 6. 幅度恢复 (Scale Recovery)
        # ==========================
        # 在实际 DeepJSCC 中，我们通常不传 scale，让 Decoder 自己学映射。
        # 但在 Latent 空间，数值范围很重要。
        # 两种选择：
        # 1. 训练时：不乘 scale，让 Loss (MSE) 强迫 Decoder 学会放大。
        # 2. 仿真时(Cheat)：直接 z_hat = z_hat * scale

        # 这里我们保持原汁原味的 DeepJSCC，不乘 scale。
        # *注意*：这意味着刚开始训练时 Loss 会很大，网络前几层权值会迅速变大。

        return z_hat
    # 在 model_R.py 的 DeepJSCC 类中添加：

    def change_channel(self, channel_type, snr):
        # 切换时也要保持 rx_ant 设置 (可以存个 self.rx_ant)
        # 或者简化处理，暂时只改 snr
        self.channel = Channel(channel_type=channel_type, snr=snr, rx_ant=self.channel.rx_ant)