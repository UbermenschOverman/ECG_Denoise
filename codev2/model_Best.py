import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class HybridSTFT_LIRA(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(HybridSTFT_LIRA, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ----- Encoder -----
        self.encoder_block1 = ResidualBlock(64, 64)
        self.down1 = nn.Conv2d(64, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))  # T: 1024 → 512

        self.encoder_block2 = ResidualBlock(64, 64)
        self.down2 = nn.Conv2d(64, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))  # T: 512 → 256

        self.encoder_block3 = ResidualBlock(64, 64)
        self.down3 = nn.Conv2d(64, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))  # T: 256 → 128

        # ----- Bottleneck -----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ----- Decoder -----
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 4), stride=(1, 2),
                                      padding=(0, 1), output_padding=(0, 1))  # 128 → 256
        self.decoder_block1 = ResidualBlock(64, 64)

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 4), stride=(1, 2),
                                      padding=(0, 1), output_padding=(0, 1))  # 256 → 512
        self.decoder_block2 = ResidualBlock(64, 64)

        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 4), stride=(1, 2),
                                      padding=(0, 1), output_padding=(0, 1))  # 512 → 1024
        self.decoder_block3 = ResidualBlock(64, 64)

        # ----- Output -----
        self.output_layer = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial(x)

        # Encoder
        x = self.encoder_block1(x)
        x = self.down1(x)

        x = self.encoder_block2(x)
        x = self.down2(x)

        x = self.encoder_block3(x)
        x = self.down3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x)
        x = self.decoder_block1(x)

        x = self.up2(x)
        x = self.decoder_block2(x)

        x = self.up3(x)
        x = self.decoder_block3(x)

        # Output
        x = self.output_layer(x)

        # Đảm bảo chiều thời gian khớp (phòng trường hợp output_padding không khớp)
        expected_t = 1024
        if x.shape[-1] != expected_t:
            x = F.interpolate(x, size=(x.shape[-2], expected_t), mode='bilinear', align_corners=False)

        return x
    
def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def sizeof_model_bytes(m):
    # tham số + buffer, giả định FP32
    n_param = sum(p.numel() for p in m.parameters())
    n_buffer = sum(b.numel() for b in m.buffers())
    return 4 * (n_param + n_buffer)

if __name__ == "__main__":
    model = HybridSTFT_LIRA(input_channels=1, output_channels=1)

    tot, train = count_params(model)
    sz_mb = sizeof_model_bytes(model) / (4096**2)
    print(f"Total params:     {tot:,}")
    print(f"Trainable params: {train:,}")
    print(f"Approx size:      {sz_mb:.2f} MB (FP32)")

    # In nhanh tham số theo từng module con cấp 1 (tùy chọn)
    print("\nPer-submodule trainable params:")
    for name, mod in model.named_children():
        p = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  {name:20s}: {p:,}")

    # sanity run (H=10 là 2F, T=4096; output sẽ nội suy về T=1024)
    x = torch.randn(1, 1, 10, 4096)
    y = model(x)
    print(f"\nInput  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
