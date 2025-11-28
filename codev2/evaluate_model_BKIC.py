import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.signal import istft
from skimage.metrics import structural_similarity as ssim

from stft_dataset import STFTDataset
from model import HybridSTFT_LIRA
import torch.nn.functional as F

# ===== Config =====
DATA_PATH   = "./dataset/datasetBKIC/training_dataset.pt"       # hoặc file test/*.pt
MODEL_PATH  = "codev2/checkpointsv2/mit22.pth"
SAVE_DIR    = "evaluation_outputs"
FS          = 360
SEG_LEN     = 4096
NPERSEG     = 8
NOVERLAP    = 7
WINDOW      = "boxcar"

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===== Helpers =====
def split_ri(stft_img_2F_T: np.ndarray):
    """ stft_img shape (2F, T) -> (F,T) real, (F,T) imag """
    assert stft_img_2F_T.ndim == 2
    Fdim = stft_img_2F_T.shape[0] // 2
    real = stft_img_2F_T[:Fdim, :]
    imag = stft_img_2F_T[Fdim:, :]
    return real, imag

def istft_from_ri(real_FT: np.ndarray, imag_FT: np.ndarray):
    Zxx = real_FT + 1j * imag_FT
    _, x = istft(Zxx, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW)
    # cắt/đệm về SEG_LEN để so sánh ổn định
    if len(x) < SEG_LEN:
        x = np.pad(x, (0, SEG_LEN - len(x)), mode="edge")
    else:
        x = x[:SEG_LEN]
    return x

def mse(a, b):  # a,b 1D
    return float(np.mean((np.asarray(a) - np.asarray(b))**2))

def rmse(a, b):
    return float(np.sqrt(mse(a, b)))

def snr(clean, test):
    clean = np.asarray(clean); test = np.asarray(test)
    n = test - clean
    Ps = np.mean(clean**2) + 1e-12
    Pn = np.mean(n**2) + 1e-12
    return 10.0 * np.log10(Ps/Pn)

def snri(noisy, denoised, clean):
    return snr(clean, denoised) - snr(clean, noisy)

def cosine_sim(a, b):
    a = np.asarray(a); b = np.asarray(b)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den

def prd(clean, rec):
    clean = np.asarray(clean); rec = np.asarray(rec)
    return float(100.0 * np.linalg.norm(clean - rec) / (np.linalg.norm(clean) + 1e-12))

def ssim_2d(img1, img2):
    # kỳ vọng cùng shape (2F,T). Không cần resize trong pipeline này.
    img1 = np.asarray(img1); img2 = np.asarray(img2)
    assert img1.shape == img2.shape
    # SSIM cần win_size lẻ và <= min(H,W)
    win = max(3, min(img1.shape) - (1 - min(img1.shape) % 2))
    return float(ssim(img1, img2, data_range=img1.max() - img1.min() + 1e-12, win_size=win))

def plot_example(idx, noisy_td, deno_td, clean_td, save_dir):
    t = np.arange(SEG_LEN) / FS
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(t, noisy_td, label="Noisy")
    
    plt.ylabel("Amplitude"); plt.legend(); plt.title("Noisy vs Denoised")
    plt.subplot(2,1,2)
    
    plt.plot(t, deno_td,  label="Denoised")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend(); plt.title("Clean vs Denoised")
    plt.tight_layout()
    path = os.path.join(save_dir, f"sample_{idx:03d}_combined.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path

# ===== Load data/model =====
dataset = STFTDataset(DATA_PATH)
loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
model   = HybridSTFT_LIRA(input_channels=1, output_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== Evaluate =====
metrics = {"mse": [], "rmse": [], "snri": [], "cosine": [], "prd": [], "ssim": []}

for i, batch in enumerate(loader, start=1):
    # shapes: (B,1,2F,T)
    x = batch["input"].to(DEVICE).float()
    y = batch["target"].to(DEVICE).float()

    with torch.no_grad():
        y_pred = model(x)
        # bảo toàn kích thước (2F,T)
        if y_pred.shape[-2:] != x.shape[-2:]:
            y_pred = F.interpolate(y_pred, size=x.shape[-2:], mode="bilinear", align_corners=False)

    x_np      = x[0,0].cpu().numpy()       # (2F,T) noisy STFT (Re;Im xếp chồng)
    y_np      = y[0,0].cpu().numpy()       # (2F,T) clean  STFT
    ypred_np  = y_pred[0,0].cpu().numpy()  # (2F,T) pred   STFT

    # time-domain reconstructions
    xr, xi        = split_ri(x_np)
    yr, yi        = split_ri(y_np)
    ypr, ypi      = split_ri(ypred_np)

    noisy_td      = istft_from_ri(xr,   xi)
    clean_td      = istft_from_ri(yr,   yi)
    denoised_td   = istft_from_ri(ypr,  ypi)

    # metrics
    mse_v   = mse(clean_td, denoised_td)
    rmse_v  = rmse(clean_td, denoised_td)
    snri_v  = snri(noisy_td, denoised_td, clean_td)
    cos_v   = cosine_sim(clean_td, denoised_td)
    prd_v   = prd(clean_td, denoised_td)
    ssim_v  = ssim_2d(y_np, ypred_np)  # SSIM trên ảnh STFT (2F,T)

    metrics["mse"].append(mse_v)
    metrics["rmse"].append(rmse_v)
    metrics["snri"].append(snri_v)
    metrics["cosine"].append(cos_v)
    metrics["prd"].append(prd_v)
    metrics["ssim"].append(ssim_v)

    img_path = plot_example(i, noisy_td, denoised_td, clean_td, SAVE_DIR)
    print(f"[{i}] saved {img_path}")
    print(f"    MSE={mse_v:.6f} RMSE={rmse_v:.6f} SNRi={snri_v:.6f} "
          f"Cos={cos_v:.6f} SSIM={ssim_v:.6f} PRD={prd_v:.6f}")

# ===== Summary =====
def avg(x): return float(np.mean(x)) if len(x) else float("nan")
print("\n== Averages ==")
print(f"MSE={avg(metrics['mse']):.6f} RMSE={avg(metrics['rmse']):.6f} "
      f"SNRi={avg(metrics['snri']):.6f} Cos={avg(metrics['cosine']):.6f} "
      f"SSIM={avg(metrics['ssim']):.6f} PRD={avg(metrics['prd']):.6f}")
