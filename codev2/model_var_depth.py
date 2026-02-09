import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class HybridSTFT_LIRA_VarDepth(nn.Module):
    """
    U-Net đối xứng: N encoder blocks + N decoder blocks (N = n_blocks).
    - Down/Up theo trục thời gian: Conv/ConvT kernel=(1,4), stride=(1,2)
    - Giữ số kênh cố định (default 64) như model gốc để so sánh công bằng
    """
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        n_blocks: int = 3,          # 2 / 3 / 4 / 5
        channels: int = 64,
        expected_t: int = 1024,     # chỉnh nếu T khác
    ):
        super().__init__()
        assert n_blocks >= 1
        self.n_blocks = n_blocks
        self.expected_t = expected_t
        c = channels

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # Encoder blocks + downs
        self.enc_blocks = nn.ModuleList([ResidualBlock(c, c) for _ in range(n_blocks)])
        self.downs = nn.ModuleList([
            nn.Conv2d(c, c, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False)
            for _ in range(n_blocks)
        ])

        # Bottleneck (giữ giống bản gốc)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Ups + decoder blocks (đối xứng)
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(
                c, c, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), output_padding=(0, 1), bias=False
            )
            for _ in range(n_blocks)
        ])
        self.dec_blocks = nn.ModuleList([ResidualBlock(c, c) for _ in range(n_blocks)])

        self.output_layer = nn.Conv2d(c, output_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.initial(x)

        # Encoder
        for blk, down in zip(self.enc_blocks, self.downs):
            x = blk(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, blk in zip(self.ups, self.dec_blocks):
            x = up(x)
            x = blk(x)

        x = self.output_layer(x)

        # match T
        if x.shape[-1] != self.expected_t:
            x = F.interpolate(x, size=(x.shape[-2], self.expected_t), mode="bilinear", align_corners=False)
        return x


def count_params(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable
