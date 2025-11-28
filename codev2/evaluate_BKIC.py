import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import istft
from model import HybridSTFT_LIRA
import shutil # Import thư viện shutil để xóa cây thư mục

# ========================
# CONFIG
# ========================
FS = 500            # Sampling rate of raw BKIC
NPERSEG = 8
NOVERLAP = 7
WINDOW = "boxcar"

# ========================
# Utility
# ========================
def get_ecg_root():
    """Tự động xác định thư mục gốc ECG."""
    try:
        this_file = os.path.abspath(__file__)
        codev2_dir = os.path.dirname(this_file)
        ecg_root = os.path.dirname(codev2_dir)
        return ecg_root
    except NameError:
        cwd = os.path.abspath(os.getcwd())
        parts = cwd.split(os.sep)
        if "ECG" in parts:
            idx = parts.index("ECG")
            return os.sep.join(parts[: idx + 1])
        return os.path.join(cwd, "ECG")

def split_ri(arr_2F_T):
    """(2F, T) → real(F,T), imag(F,T)"""
    F = arr_2F_T.shape[0] // 2
    return arr_2F_T[:F], arr_2F_T[F:]

def istft_from_ri(real_FT, imag_FT):
    Zxx = real_FT + 1j * imag_FT
    _, x = istft(Zxx, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW)
    return x

def read_ecg_txt(path):
    """Read BKIC txt → column 9 (index 8)."""
    arr = np.loadtxt(path)
    if arr.ndim == 1 or arr.shape[1] < 9:
        raise ValueError(f"Invalid TXT (must have >=9 columns): {path}")
    return arr[:, 8]

def load_segments(seg_dir):
    """Load *.npy STFT BKIC segments."""
    files = sorted([f for f in os.listdir(seg_dir) if f.endswith(".npy")])
    segs = []
    for f in files:
        arr = np.load(os.path.join(seg_dir, f))

        # valid shapes:
        # (10, T)
        # (1, 10, T)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]                     # → (10,T)
        if arr.ndim != 2 or arr.shape[0] % 2 != 0:
            raise ValueError(f"Invalid segment shape {arr.shape} in {f}")

        segs.append(arr)
    return segs, files

def clear_directory(dir_path):
    """Xóa tất cả file và folder trong một thư mục."""
    if not os.path.exists(dir_path):
        return
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path) # Xóa file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # Xóa thư mục
        except Exception as e:
            print(f'Thất bại khi xóa {file_path}. Lý do: {e}')

# ========================
# Setup paths
# ========================
ECG_ROOT = get_ecg_root()
DATA_DIR = os.path.join(ECG_ROOT, "datasetBKIC_preprocessed")
RAW_DIR  = os.path.join(ECG_ROOT, "datasetBKIC")
MODEL_DIR = os.path.join(ECG_ROOT, "trained_models")
SAVE_DIR = os.path.join(ECG_ROOT, "evaluation_outputs_BKIC")

# Xóa nội dung cũ trong thư mục trước khi tạo mới
print(f"🔄 Đang xóa nội dung cũ trong thư mục: {SAVE_DIR}")
clear_directory(SAVE_DIR)

os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "mit22.pth")

# ========================
# Load model
# ========================
print(f"🔹 Loading model: {MODEL_PATH}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridSTFT_LIRA(input_channels=1, output_channels=1).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ========================
# Evaluate subjects
# ========================
subjects = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"🔸 Found {len(subjects)} subjects")

for subj in subjects:
    print(f"\n🔹 Processing {subj} ...")

    seg_dir = os.path.join(DATA_DIR, subj)
    segs, seg_files = load_segments(seg_dir)

    # RAW ECG
    raw_path = os.path.join(RAW_DIR, f"{subj}.txt")
    raw_sig = read_ecg_txt(raw_path)

    # Denoise reconstruction
    reconstructed = []

    for arr in segs:
        # arr → (10, T)
        x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        # x: (1,1,10,T)

        with torch.no_grad():
            pred = model(x)          # (1,1,10,1024)
            pred = pred.squeeze().cpu().numpy()   # (10,1024)

        real_FT, imag_FT = split_ri(pred)
        wave = istft_from_ri(real_FT, imag_FT)
        reconstructed.append(wave)

    denoised = np.concatenate(reconstructed)

    # Align length with raw
    min_len = min(len(raw_sig), len(denoised))
    raw_sig = raw_sig[:min_len]
    denoised = denoised[:min_len]

    # ========================
    # Save comparison plot
    # ========================
    t = np.arange(min_len) / FS
    plt.figure(figsize=(14,5))
    plt.plot(t, raw_sig, label="Raw", linewidth=0.8, alpha=0.8)
    plt.plot(t, denoised, label="Denoised", linewidth=1.0)
    plt.title(f"{subj} — Raw vs Denoised")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, f"{subj}_compare.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"   ✅ Saved {out_path}")

print("\n🎯 BKIC Evaluation Completed.")