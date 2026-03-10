import os
import glob
import numpy as np
import torch
import csv
import shutil
import matplotlib.pyplot as plt
from scipy.signal import istft, butter, filtfilt, stft, resample
from model_var_depth import HybridSTFT_LIRA_VarDepth as HybridSTFT_LIRA

# ========================
# CONFIG
# ========================
TARGET_FS = 360
NPERSEG = 20
NOVERLAP = 19
WINDOW = "boxcar"

# Input config (BKIC3)
ORIGINAL_FS = 500

# Filtering Params
HP_CUTOFF = 0.67
LP_CUTOFF = 100.0
HP_ORDER = 4
LP_ORDER = 5

# ========================
# Utility
# ========================
def get_ecg_root():
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
    F = arr_2F_T.shape[0] // 2
    return arr_2F_T[:F], arr_2F_T[F:]

def istft_from_ri(real_FT, imag_FT):
    Zxx = real_FT + 1j * imag_FT
    _, x = istft(Zxx, fs=TARGET_FS, nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW)
    return x

def compute_stft_ri_2ch(x_1d: np.ndarray):
    f, t, Z = stft(
        x_1d, fs=TARGET_FS, window=WINDOW,
        nperseg=NPERSEG, noverlap=NOVERLAP,
        boundary=None, padded=False,
        detrend=False, return_onesided=True
    )
    Ri = np.vstack([np.real(Z), np.imag(Z)])
    return Ri.astype(np.float32)

def read_ecg_bkic3(path):
    # Lấy cột 8 (index 7)
    try:
        data = np.genfromtxt(path)
        if data.ndim == 1 or data.shape[1] < 8:
            print(f"Skipping {path}: missing column 8")
            return None
        return data[:, 7]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def calibrate_f3(amp):
    # f3 = ((5-5/2^24)/(2^24-1))*Amp;
    return ((5 - 5 / (2**24)) / (2**24 - 1)) * amp

def bandpass_filter(data: np.ndarray, fs: int) -> np.ndarray:
    nyq = 0.5 * fs
    b_hp, a_hp = butter(HP_ORDER, HP_CUTOFF / nyq, btype="highpass")
    y = filtfilt(b_hp, a_hp, data)
    b_lp, a_lp = butter(LP_ORDER, LP_CUTOFF / nyq, btype="lowpass")
    y = filtfilt(b_lp, a_lp, y)
    return y

# ========================
# Processor
# ========================
def process_bkic3(depth, model_filename, output_subdir, raw_dir, model_dir_path):
    print(f"\n🚀 Processing all datasetBKIC3 files for Depth {depth}")
    print(f"   Model: {model_filename}")

    os.makedirs(output_subdir, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir_path, model_filename)

    # Load Model
    model = HybridSTFT_LIRA(
        input_channels=1, 
        output_channels=1, 
        n_blocks=depth, 
        channels=64, 
        expected_t=4096 
    ).to(DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Model {model_filename} loaded.")
    except Exception as e:
        print(f"⚠️ Failed to load model {model_filename}: {e}")
        return

    model.eval()

    # Get all .txt files
    txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    print(f"Found {len(txt_files)} files in {raw_dir}")

    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        filename_base = os.path.splitext(filename)[0]
        
        # Read
        amp8 = read_ecg_bkic3(txt_path)
        if amp8 is None: continue

        # 1. Calibrate f3
        raw_calibrated = calibrate_f3(amp8)

        # 2. Resample 500 -> 360
        num_samples_360 = int(len(raw_calibrated) * TARGET_FS / ORIGINAL_FS)
        if num_samples_360 == 0: continue
        raw_sig_360 = resample(raw_calibrated, num_samples_360)

        # 3. Normalize (Z-score)
        mu = np.mean(raw_sig_360)
        std = np.std(raw_sig_360) + 1e-6
        raw_norm = (raw_sig_360 - mu) / std

        # 4. Filter
        filtered_norm = bandpass_filter(raw_norm, TARGET_FS)

        # 5. Segment & Infer
        SEG_LEN = 4096
        n = len(filtered_norm)
        k = n // SEG_LEN
        
        segments = [(i * SEG_LEN, (i + 1) * SEG_LEN) for i in range(k)]
        if n % SEG_LEN != 0:
             segments.append((k * SEG_LEN, n))
        
        full_den_concat = []

        for i, (s, e) in enumerate(segments):
            seg = filtered_norm[s:e]
            seg_len_actual = e - s
            
            if seg_len_actual < SEG_LEN:
                pad_len = SEG_LEN - seg_len_actual
                seg_padded = np.concatenate([seg, np.zeros(pad_len)])
            else:
                seg_padded = seg
            
            stft_ri = compute_stft_ri_2ch(seg_padded)
            inp_tensor = torch.from_numpy(stft_ri).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred = model(inp_tensor)
                pred = pred.squeeze().cpu().numpy()
            
            real_FT, imag_FT = split_ri(pred)
            wave_chunk = istft_from_ri(real_FT, imag_FT)
            wave_chunk = wave_chunk[:seg_len_actual]
            full_den_concat.append(wave_chunk)

        if not full_den_concat: continue

        den_signal_360 = np.concatenate(full_den_concat)
        
        # 6. Save output CSV & Plot
        csv_path = os.path.join(output_subdir, f"{filename_base}.csv")
        img_path = os.path.join(output_subdir, f"{filename_base}.png")
        
        min_len = min(len(raw_norm), len(den_signal_360))
        t_axis = np.arange(min_len) / TARGET_FS
        
        # Save CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time_s", "Raw_Preprocessed_Norm", "Denoised_Norm"])
            for t, r, d in zip(t_axis, raw_norm[:min_len], den_signal_360[:min_len]):
                writer.writerow([f"{t:.4f}", f"{r:.6f}", f"{d:.6f}"])
        
        # Save Plot
        plt.figure(figsize=(15, 6))
        plt.plot(t_axis, raw_norm[:min_len], label="Raw (Normalized)", color='blue', alpha=0.5, linewidth=0.8)
        plt.plot(t_axis, den_signal_360[:min_len], label="Denoised (TFCNN)", color='red', linewidth=1.0)
        plt.title(f"Denoising Evaluation: {filename} (Depth {depth})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normalized)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        plt.close()
        
        print(f"   ✅ Processed {filename}")

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    ECG_ROOT = get_ecg_root()
    RAW_DIR  = os.path.join(ECG_ROOT, "datasetBKIC3")
    MODEL_DIR_PATH = os.path.join(ECG_ROOT, "trained_models")
    MAIN_OUT_DIR = os.path.join(ECG_ROOT, "evaluation_outputs_BKIC3_all")

    if os.path.exists(MAIN_OUT_DIR):
        shutil.rmtree(MAIN_OUT_DIR)
    os.makedirs(MAIN_OUT_DIR, exist_ok=True)

    # 1. Evaluate Depth 4
    process_bkic3(4, "TFCNN_depth4_best.pth", 
                   os.path.join(MAIN_OUT_DIR, "TFCNN_depth4"), 
                   RAW_DIR, MODEL_DIR_PATH)

    # 2. Evaluate Depth 5
    process_bkic3(5, "TFCNN_depth5_best.pth", 
                   os.path.join(MAIN_OUT_DIR, "TFCNN_depth5"), 
                   RAW_DIR, MODEL_DIR_PATH)
    
    print("\n✅ All Tasks Completed.")
