import torch
import torch.nn as nn
import torch.nn.functional as F
input_channels, output_channels = 1, 1
# ====== building blocks ======
def sep_conv3x3(in_ch, out_ch, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=dilation, dilation=dilation,
                  groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class BottleneckDS(nn.Module):
    def __init__(self, ch, r=4, dilation=1):
        super().__init__()
        mid = max(8, ch // r)
        self.reduce = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.sep = sep_conv3x3(mid, mid, dilation=dilation)
        self.expand = nn.Sequential(
            nn.Conv2d(mid, ch, 1, bias=False), nn.BatchNorm2d(ch)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        idn = x
        out = self.reduce(x)
        out = self.sep(out)
        out = self.expand(out)
        return self.act(out + idn)

class UpBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.post = sep_conv3x3(ch, ch)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2), mode='bilinear', align_corners=False)
        return self.post(x)

# ====== model S ======
class HybridSTFT_LIRA_S(nn.Module):
    """
    Input:  (B,1,H=2F,T=1024)
    Output: (B,1,H,T=1024)
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        C = 64
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C), nn.ReLU(inplace=True)
        )
        # Encoder 3 mức
        self.e1 = BottleneckDS(C)
        self.down1 = nn.Conv2d(C, C, kernel_size=(1,4), stride=(1,2), padding=(0,1), bias=False)
        self.bn_d1 = nn.BatchNorm2d(C)

        self.e2 = BottleneckDS(C)
        self.down2 = nn.Conv2d(C, C, kernel_size=(1,4), stride=(1,2), padding=(0,1), bias=False)
        self.bn_d2 = nn.BatchNorm2d(C)

        self.e3 = BottleneckDS(C)
        self.down3 = nn.Conv2d(C, C, kernel_size=(1,4), stride=(1,2), padding=(0,1), bias=False)
        self.bn_d3 = nn.BatchNorm2d(C)

        # Bottleneck
        self.bott = BottleneckDS(C)

        # Decoder 3 mức
        self.up1 = UpBlock(C); self.d1 = BottleneckDS(C)
        self.up2 = UpBlock(C); self.d2 = BottleneckDS(C)
        self.up3 = UpBlock(C); self.d3 = BottleneckDS(C)

        self.head = nn.Conv2d(C, input_channels, 3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.e1(x); x = self.bn_d1(self.down1(x))
        x = self.e2(x); x = self.bn_d2(self.down2(x))
        x = self.e3(x); x = self.bn_d3(self.down3(x))
        x = self.bott(x)
        x = self.d1(self.up1(x))
        x = self.d2(self.up2(x))
        x = self.d3(self.up3(x))
        x = self.head(x)
        if x.shape[-1] != 4096:
            x = F.interpolate(x, size=(x.shape[-2], 4096), mode='bilinear', align_corners=False)
        return x

# ====== run to print params ======
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = HybridSTFT_LIRA_S(input_channels, output_channels)
    n_params = count_params(model)
    print(f"Params: {n_params}")
    # sanity input: H arbitrary (e.g., 34), T=1024
    x = torch.randn(1, 1, 10, 4096)
    y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
