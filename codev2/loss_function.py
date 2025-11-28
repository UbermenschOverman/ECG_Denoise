import torch
import torch.nn as nn
from pytorch_msssim import ssim

class NoiseReductionLoss(nn.Module):
    def __init__(self, alpha=1, beta=0, gamma=0):
        """
        alpha: trọng số MSE (giữ điểm chính xác)
        beta:  trọng số L1 (giảm sai số tuyệt đối)
        gamma: trọng số SSIM (giữ hình dạng)
        """
        super(NoiseReductionLoss, self).__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # L1 weight
        self.gamma = gamma  # SSIM weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.win_size = 5

    def forward(self, output, target):
        # Đảm bảo cùng kích thước
        if output.shape != target.shape:
            min_len = min(output.shape[-1], target.shape[-1])
            output = output[..., :min_len]
            target = target[..., :min_len]

        # 1. MSE Loss
        loss_mse = self.mse(output, target)

        # 2. L1 Loss
        loss_l1 = self.l1(output, target)

        # 3. SSIM Loss
        loss_ssim = 1 - ssim(output, target, data_range=1.0, size_average=True, win_size=self.win_size)

        return self.alpha * loss_mse + self.beta * loss_l1 + self.gamma * loss_ssim
