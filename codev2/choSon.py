import os
import glob
import numpy as np
import torch
import csv
import shutil
from scipy.signal import istft, butter, filtfilt, stft, resample
from model_var_depth import HybridSTFT_LIRA_VarDepth as HybridSTFT_LIRA

# ========================
# CONFIG
# ========================
TARGET_FS = 360
NPERSEG = 20
NOVERLAP = 19
WINDOW = "boxcar"

# Input config (BKIC Dataset Old)
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
        boundary=None, padded=False, # Match training config
        detrend=False, return_onesided=True
    )
    Ri = np.vstack([np.real(Z), np.imag(Z)])
    return Ri.astype(np.float32)

def read_ecg_txt(path):
    # datasetBKIC (Old) -> Column 9 (index 8)
    try:
        arr = np.loadtxt(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
        
    if arr.ndim == 1:
        print(f"Skipping {path}: 1D array")
        return None
    if arr.shape[1] < 9:
        print(f"Skipping {path}: Less than 9 columns")
        return None
        
    return arr[:, 8] # Index 8 for datasetBKIC

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
def process_dataset(depth, model_filename, output_subdir, raw_dir, model_dir_path):
    print(f"\n========================================")
    print(f"🚀 Processing datasetBKIC for Depth {depth}")
    print(f"   Model: {model_filename}")
    print(f"========================================")

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

    # Get all txt files in datasetBKIC
    txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    print(f"Found {len(txt_files)} files in {raw_dir}")

    for txt_path in txt_files:
        filename_base = os.path.splitext(os.path.basename(txt_path))[0]
        
        # Read
        raw_sig_500 = read_ecg_txt(txt_path)
        if raw_sig_500 is None: continue

        # 1. Resample 500 -> 360
        num_samples_360 = int(len(raw_sig_500) * TARGET_FS / ORIGINAL_FS)
        if num_samples_360 == 0: continue
        raw_sig_360 = resample(raw_sig_500, num_samples_360)

        # 2. Normalize
        mu = np.mean(raw_sig_360)
        std = np.std(raw_sig_360) + 1e-6
        raw_norm = (raw_sig_360 - mu) / std

        # 3. Filter
        filtered_norm = bandpass_filter(raw_norm, TARGET_FS)

        # 4. Segment & Infer (Full Signal)
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
            
            # Pad if needed
            if seg_len_actual < SEG_LEN:
                pad_len = SEG_LEN - seg_len_actual
                seg_padded = np.concatenate([seg, np.zeros(pad_len)])
            else:
                seg_padded = seg
            
            # STFT
            stft_ri = compute_stft_ri_2ch(seg_padded)
            inp_tensor = torch.from_numpy(stft_ri).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred = model(inp_tensor)
                pred = pred.squeeze().cpu().numpy()
            
            real_FT, imag_FT = split_ri(pred)
            wave_chunk = istft_from_ri(real_FT, imag_FT)

            # Crop output to actual length
            wave_chunk = wave_chunk[:seg_len_actual]
            full_den_concat.append(wave_chunk)

        if not full_den_concat: continue

        den_signal_360 = np.concatenate(full_den_concat)
        
        # 5. Save output CSV
        # We save the Normalized 360Hz signals (Input to Model vs Output of Model)
        # This aligns with evaluate_BKIC.py plots.
        
        csv_path = os.path.join(output_subdir, f"{filename_base}.csv")
        
        # Ensure lengths match exactly (they should)
        min_len = min(len(filtered_norm), len(den_signal_360))
        
        t_axis = np.arange(min_len) / TARGET_FS
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time_s", "Raw_Preprocessed_Norm", "Denoised_Norm"])
            for t, r, d in zip(t_axis, filtered_norm[:min_len], den_signal_360[:min_len]):
                writer.writerow([f"{t:.4f}", f"{r:.6f}", f"{d:.6f}"])
                
        # print(f"   ✅ Processed {filename_base}") # Reduced spam
        
    print(f"✅ Finished processing {len(txt_files)} files.")

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    ECG_ROOT = get_ecg_root()
    RAW_DIR  = os.path.join(ECG_ROOT, "datasetBKIC") # OLD Dataset
    MODEL_DIR_PATH = os.path.join(ECG_ROOT, "trained_models")
    MAIN_OUT_DIR = os.path.join(ECG_ROOT, "evaluation_outputs_BKIC2_choSon")

    print(f" Clearing main output directory: {MAIN_OUT_DIR}")
    if os.path.exists(MAIN_OUT_DIR):
        shutil.rmtree(MAIN_OUT_DIR)
    os.makedirs(MAIN_OUT_DIR, exist_ok=True)

    # 1. Evaluate Depth 4
    process_dataset(4, "TFCNN_depth4_best.pth", 
                   os.path.join(MAIN_OUT_DIR, "TFCNN_depth4"), 
                   RAW_DIR, MODEL_DIR_PATH)

    # 2. Evaluate Depth 5
    process_dataset(5, "TFCNN_depth5_best.pth", 
                   os.path.join(MAIN_OUT_DIR, "TFCNN_depth5"), 
                   RAW_DIR, MODEL_DIR_PATH)
    
    print("\n✅ All Tasks Completed.")
