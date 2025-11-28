"""
train_mitdb.py
│
├── đọc dữ liệu STFT từ .pt → ./datasetv2/
│   ├── train.pt
│   └── val.pt
│
├── huấn luyện model → HybridSTFT_LIRA (model.py)
│
├── lưu model vào → ./codev2/checkpoints/
│   └── mit34.pth
│
└── sinh ảnh visualize → ./output_images_after_training.png
TRAIN_PATH = "./datasetv2/train.pt"
VAL_PATH   = "./datasetv2/val.pt"
Hai file .pt này là dataset đã được chuẩn bị sẵn ở dạng STFT tensor,
mỗi phần tử có: {"input": noisy_STFT, "target": clean_STFT}
Thư mục datasetv2/ nằm cùng cấp với thư mục codev2/
Nếu đổi vị trí folder, chỉ cần sửa hai dòng này cho khớp.
Ví dụ:
TRAIN_PATH = "../preprocessed_mitdb/train.pt"
VAL_PATH   = "../preprocessed_mitdb/val.pt"
CHECKPOINT_PATH = "./codev2/checkpoints/mit34.pth"
(Nếu folder checkpoints/ chưa có, script sẽ tự động tạo bằng:os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True))
"""

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

# ===============================
#         LOSS FUNCTION
# ===============================
class NoiseReductionLoss(nn.Module):
    def __init__(self, alpha=1, beta=0, gamma=0):
        """
        alpha: trọng số MSE (giữ điểm chính xác)
        beta:  trọng số L1 (giảm sai số tuyệt đối)
        gamma: trọng số SSIM (giữ hình dạng)
        """
        super(NoiseReductionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.win_size = 5

    def forward(self, output, target):
        # Đảm bảo cùng kích thước
        if output.shape != target.shape:
            min_len = min(output.shape[-1], target.shape[-1])
            output = output[..., :min_len]
            target = target[..., :min_len]

        # 1️⃣ MSE Loss
        loss_mse = self.mse(output, target)
        # 2️⃣ L1 Loss
        loss_l1 = self.l1(output, target)
        # 3️⃣ SSIM Loss
        loss_ssim = 1 - ssim(output, target, data_range=1.0, size_average=True, win_size=self.win_size)

        return self.alpha * loss_mse + self.beta * loss_l1 + self.gamma * loss_ssim

# ===============================
#           TRAINING
# ===============================
from stft_dataset import STFTDataset
from model import HybridSTFT_LIRA

# ==== SEED ====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==== CONFIG ====
TRAIN_PATH = "./datasetv2/train.pt"
VAL_PATH   = "./datasetv2/val.pt"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
CHECKPOINT_PATH = "./codev2/checkpoints/mit34.pth"
PATIENCE = 10

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {DEVICE}")

# ==== DATASETS & LOADERS ====
train_set = STFTDataset(TRAIN_PATH)
val_set   = STFTDataset(VAL_PATH)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ==== MODEL + LOSS + OPTIMIZER ====
model = HybridSTFT_LIRA(input_channels=1, output_channels=1).to(DEVICE)
criterion = NoiseReductionLoss(alpha=1, beta=0, gamma=0)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==== TRAINING LOOP ====
best_val = float("inf")
stale = 0
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        x = batch["input"].to(DEVICE).float()
        y = batch["target"].to(DEVICE).float()

        optimizer.zero_grad()
        out = model(x)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()

    tr_loss /= max(1, len(train_loader))

    # === Validation ===
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["input"].to(DEVICE).float()
            y = batch["target"].to(DEVICE).float()
            out = model(x)
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
            va_loss += criterion(out, y).item()
    va_loss /= max(1, len(val_loader))

    print(f"[Epoch {epoch}/{EPOCHS}] Train: {tr_loss:.6f} | Val: {va_loss:.6f}")

    # === Checkpointing ===
    if va_loss < best_val:
        best_val = va_loss
        stale = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✅ Saved best model: {CHECKPOINT_PATH}")
    else:
        stale += 1
        if stale >= PATIENCE:
            print(f"🚨 Early stopping at epoch {epoch}.")
            break

# ===============================
#      VISUALIZATION EXAMPLE
# ===============================
def resize_to_length(img, target_len):
    h, w = img.shape
    if w == target_len:
        return img
    from skimage.transform import resize
    return resize(img, (h, target_len), preserve_range=True, anti_aliasing=True)

model.eval()
with torch.no_grad():
    sample_batch = next(iter(val_loader))
    x = sample_batch["input"].to(DEVICE).float()
    y = sample_batch["target"].to(DEVICE).float()
    pred = model(x)
    if pred.shape[-2:] != x.shape[-2:]:
        pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)

    inp = x[0].detach().cpu().numpy().squeeze()
    tg  = y[0].detach().cpu().numpy().squeeze()
    out = pred[0].detach().cpu().numpy().squeeze()

    target_len = tg.shape[1]
    if inp.ndim == 3: inp = inp[0]
    if tg.ndim  == 3: tg  = tg[0]
    if out.ndim == 3: out = out[0]
    inp = resize_to_length(inp, target_len)
    out = resize_to_length(out, target_len)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(inp, aspect="auto", cmap="jet"); axs[0].set_title("Input (Noisy)");  axs[0].axis("off")
    axs[1].imshow(out, aspect="auto", cmap="jet"); axs[1].set_title("Output (Pred)");  axs[1].axis("off")
    axs[2].imshow(tg,  aspect="auto", cmap="jet"); axs[2].set_title("Target (Clean)"); axs[2].axis("off")
    plt.tight_layout()
    plt.savefig("output_images_after_training.png", dpi=200)
    plt.close()
    print("✅ Saved visualization: output_images_after_training.png")

# ==== Final save ====
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"✅ Final model saved to {CHECKPOINT_PATH}")
