import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import istft, butter, filtfilt, stft, resample
from model_var_depth import HybridSTFT_LIRA_VarDepth as HybridSTFT_LIRA
import shutil

# ========================
# CONFIG (Updated for TFCNN_depth4)
# ========================
# Target config for the Model
TARGET_FS = 360
NPERSEG = 20
NOVERLAP = 19
WINDOW = "boxcar"

# Input config (BKIC Dataset)
ORIGINAL_FS = 500

# Filtering Params (From Preprocessing_BKIC code)
HP_CUTOFF = 0.67
LP_CUTOFF = 100.0
HP_ORDER = 4
LP_ORDER = 5

# ========================
# Utility
# ========================
def get_ecg_root():
    """Automatically determine ECG root directory."""
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
    """(2F, T) -> real(F,T), imag(F,T)"""
    F = arr_2F_T.shape[0] // 2
    return arr_2F_T[:F], arr_2F_T[F:]

def istft_from_ri(real_FT, imag_FT):
    # Reconstruct at TARGET_FS (360Hz)
    Zxx = real_FT + 1j * imag_FT
    _, x = istft(Zxx, fs=TARGET_FS, nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW)
    return x

def compute_stft_ri_2ch(x_1d: np.ndarray):
    """
    Compute STFT for a 1D signal and stack Real/Imag.
    Returns: (22, T) tensor (2F where F=11).
    """
    f, t, Z = stft(
        x_1d, fs=TARGET_FS, window=WINDOW,
        nperseg=NPERSEG, noverlap=NOVERLAP,
        boundary=None, padded=False, # Match training config
        detrend=False, return_onesided=True
    )
    # Z shape: (11, T)
    Ri = np.vstack([np.real(Z), np.imag(Z)]) # (22, T)
    return Ri.astype(np.float32)

def read_ecg_txt(path):
    """Read BKIC txt -> column 9 (index 8)."""
    arr = np.loadtxt(path)
    if arr.ndim == 1 or arr.shape[1] < 9:
        raise ValueError(f"Invalid TXT (must have >=9 columns): {path}")
    return arr[:, 8]

def bandpass_filter(data: np.ndarray, fs: int) -> np.ndarray:
    nyq = 0.5 * fs
    b_hp, a_hp = butter(HP_ORDER, HP_CUTOFF / nyq, btype="highpass")
    y = filtfilt(b_hp, a_hp, data)
    b_lp, a_lp = butter(LP_ORDER, LP_CUTOFF / nyq, btype="lowpass")
    y = filtfilt(b_lp, a_lp, y)
    return y

def clear_directory(dir_path):
    """Clean all files in directory."""
    if not os.path.exists(dir_path):
        return
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# ========================
# Setup paths
# ========================
ECG_ROOT = get_ecg_root()
# We use existing preprocessed folders just to get list of subjects, 
# but we will READ RAW .TXT and re-process on the fly.
DATA_DIR = os.path.join(ECG_ROOT, "datasetBKIC_preprocessed") 
RAW_DIR  = os.path.join(ECG_ROOT, "datasetBKIC")
MODEL_DIR = os.path.join(ECG_ROOT, "trained_models")
SAVE_DIR = os.path.join(ECG_ROOT, "evaluation_outputs_BKIC_v3")

print(f"🔄 Clearing old outputs in: {SAVE_DIR}")
clear_directory(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# Update to use the TFCNN_depth4 model
MODEL_PATH = os.path.join(MODEL_DIR, "TFCNN_depth4_best.pth")

# ========================
# Load model
# ========================
print(f"🔹 Loading model: {MODEL_PATH}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model config updated for TFCNN_depth4 (n_blocks=4)
model = HybridSTFT_LIRA(
    input_channels=1, 
    output_channels=1, 
    n_blocks=4,  # Changed from 5 to 4
    channels=64, 
    expected_t=4096 
).to(DEVICE)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Model loaded successfully (from checkpoint dict).")
except Exception as e:
    print(f"⚠️ Failed to load safely, trying raw load... Error: {e}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

# ========================
# Evaluate subjects
# ========================
subjects = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"🔸 Found {len(subjects)} subjects (based on preprocessed folders)")

# Process only a few or all? Let's process all found.
for subj in subjects:
    print(f"\n🔹 Processing {subj} ...")
    
    # RAW ECG (500Hz)
    raw_path = os.path.join(RAW_DIR, f"{subj}.txt")
    if not os.path.exists(raw_path):
        print(f"⚠️ Raw file not found: {raw_path}")
        continue
        
    raw_sig_500 = read_ecg_txt(raw_path)
    
    # 1. Resample 500Hz -> 360Hz
    num_samples_360 = int(len(raw_sig_500) * TARGET_FS / ORIGINAL_FS)
    raw_sig_360 = resample(raw_sig_500, num_samples_360)

    # 1b. NORMALIZE (Z-score)
    # Correct scale mismatch: Raw is ~10^4 (ADC), Model trained on ~10^0 (mV)
    mu = np.mean(raw_sig_360)
    std = np.std(raw_sig_360) + 1e-6
    raw_norm = (raw_sig_360 - mu) / std
    
    # 2. Bandpass Filter (FOR MODEL INPUT ONLY)
    # Apply to normalized signal
    filtered_norm = bandpass_filter(raw_norm, TARGET_FS)
    
    # 3. Segment into chunks (e.g., 4096 samples) to avoid memory issues or matching training context
    SEG_LEN = 4096
    
    # Simple non-overlapping segmentation
    n = len(filtered_norm) 
    k = n // SEG_LEN
    segments = [(i * SEG_LEN, (i + 1) * SEG_LEN) for i in range(k)]
    
    # SEGMENT LOOP
    # We now plot PER SEGMENT instead of concatenating
    for i, (s, e) in enumerate(segments):
        seg = filtered_norm[s:e]
        
        # 4. STFT -> (22, T_stft)
        stft_ri = compute_stft_ri_2ch(seg) # shape (22, T)
        
        # Prepare tensor (Batch=1, Channel=1, F=22, T)
        # Note: Model expects (1, 1, 22, T)
        inp_tensor = torch.from_numpy(stft_ri).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Model forward
            pred = model(inp_tensor)
            pred = pred.squeeze().cpu().numpy() # (22, T)
            
        real_FT, imag_FT = split_ri(pred)
        wave_chunk = istft_from_ri(real_FT, imag_FT)
        
        # ISTFT length adjustment
        if len(wave_chunk) > SEG_LEN:
            wave_chunk = wave_chunk[:SEG_LEN]
        elif len(wave_chunk) < SEG_LEN:
            pad = np.zeros(SEG_LEN - len(wave_chunk))
            wave_chunk = np.concatenate([wave_chunk, pad])
            
        # Get Reference (Raw Unfiltered Normalized) for this segment
        ref_seg = raw_norm[s:e]
        
        # ========================
        # Save comparison plot (PER SEGMENT)
        # ========================
        # Ensure lengths match for plotting (sometimes edge cases exist)
        plot_len = min(len(ref_seg), len(wave_chunk))
        ref_plot = ref_seg[:plot_len]
        wave_plot = wave_chunk[:plot_len]
        
        t_axis = np.arange(plot_len) / TARGET_FS
        
        plt.figure(figsize=(14,5))
        # Raw (Unfiltered) = Blue, Denoised = Red
        plt.plot(t_axis, ref_plot, label="Raw (Resampled, Normalized)", linewidth=0.8, alpha=0.9, color='blue')
        plt.plot(t_axis, wave_plot, label="Denoised (TFCNN, Normalized)", linewidth=1.0, color='red')
        plt.title(f"{subj} — Segment {i} — Raw vs Denoised (TFCNN_depth4)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = os.path.join(SAVE_DIR, f"{subj}_seg{i:03d}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    print(f"   ✅ Saved segments for {subj}")

print("\n🎯 BKIC Segment Evaluation Completed (Depth 4).")
