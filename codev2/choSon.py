import os
import glob
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from scipy.signal import istft, butter, filtfilt, stft, resample
from model_var_depth import HybridSTFT_LIRA_VarDepth as HybridSTFT_LIRA
import shutil

# ========================
# CONFIG
# ========================
TARGET_FS = 360
NPERSEG = 20
NOVERLAP = 19
WINDOW = "boxcar"

# Input config (BKIC Dataset matches original)
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
    # BKIC2 -> Column 8 (index 7) based on user update
    arr = np.loadtxt(path)
    if arr.ndim == 1 or arr.shape[1] < 8:
        raise ValueError(f"Invalid TXT (must have >=8 columns): {path}")
    return arr[:, 7] # Adjusted to column 8 (idx 7)

def bandpass_filter(data: np.ndarray, fs: int) -> np.ndarray:
    nyq = 0.5 * fs
    b_hp, a_hp = butter(HP_ORDER, HP_CUTOFF / nyq, btype="highpass")
    y = filtfilt(b_hp, a_hp, data)
    b_lp, a_lp = butter(LP_ORDER, LP_CUTOFF / nyq, btype="lowpass")
    y = filtfilt(b_lp, a_lp, y)
    return y

def clear_directory(dir_path):
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
# Main Model Evaluator Logic
# ========================
def process_picked_files(depth, model_filename, output_subdir, raw_dir, picked_list, model_dir_path):
    print(f"\n========================================")
    print(f"🚀 Processing CHO SON'S LIST for Depth {depth}")
    print(f"   Model: {model_filename}")
    print(f"   Target Files: {len(picked_list)}")
    print(f"========================================")

    save_dir = output_subdir
    os.makedirs(save_dir, exist_ok=True)
    
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

    for filename_base in picked_list:
        txt_path = os.path.join(raw_dir, f"{filename_base}.txt")
        if not os.path.exists(txt_path):
            print(f"   ⚠️ Raw file missing: {txt_path}")
            continue
            
        # Read & Preprocess
        try:
            raw_sig_500 = read_ecg_txt(txt_path)
        except Exception as e:
            print(f"   ❌ Error reading {filename_base}: {e}")
            continue

        # 0. Keep Original Raw for Plotting/CSV
        # raw_sig_500 is the True Raw (500Hz, unnormalized, unfiltered)

        # 1. Resample 500 -> 360 (ONLY for Model Input)
        num_samples_360 = int(len(raw_sig_500) * TARGET_FS / ORIGINAL_FS)
        if num_samples_360 == 0: continue
        raw_sig_360 = resample(raw_sig_500, num_samples_360)

        # 2. Normalize (ONLY for Model Input)
        mu = np.mean(raw_sig_360)
        std = np.std(raw_sig_360) + 1e-6
        raw_norm = (raw_sig_360 - mu) / std

        # 3. Filter (ONLY for Model Input)
        filtered_norm = bandpass_filter(raw_norm, TARGET_FS)

        # 4. Segment & Infer
        SEG_LEN = 4096
        n = len(filtered_norm)
        k = n // SEG_LEN
        
        segments = [(i * SEG_LEN, (i + 1) * SEG_LEN) for i in range(k)]
        if n % SEG_LEN != 0:
             segments.append((k * SEG_LEN, n))
        
        full_den_concat = []

        # PROCESSING SEGMENTS
        for i, (s, e) in enumerate(segments):
            seg = filtered_norm[s:e]
            seg_len_actual = e - s
            
            # Pad if segment is shorter than SEG_LEN (e.g., last segment)
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

            # Match length (Crop to actual segment length)
            # Inverse STFT might return slight diff length + padding we added
            wave_chunk = wave_chunk[:seg_len_actual]
            
            full_den_concat.append(wave_chunk)

        if not full_den_concat: continue

        den_signal_360 = np.concatenate(full_den_concat)
        
        # Resample Denoised back to 500Hz (Original Length) for comparison
        target_len_500 = len(raw_sig_500)
        # Using simple resize/resample to match length exactly
        # Note: den_signal_360 covers only the processed part (might be slightly shorter due to segmentation logic if remainder dropped)
        # If we dropped remainder in input processing, we must crop raw_sig_500 too or handle it.
        # Current logic: segments cover up to `k*SEG_LEN` or `n` if k=0. 
        # Actually logic above effectively covers `k*SEG_LEN`. If n > k*SEG_LEN, the tail was ignored.
        
        processed_len_360 = len(den_signal_360)
        # Calculate how much of original 500Hz this corresponds to
        processed_len_500 = int(processed_len_360 * ORIGINAL_FS / TARGET_FS)
        
        if processed_len_500 > len(raw_sig_500):
            processed_len_500 = len(raw_sig_500)
            
        # Crop Raw to match processed duration
        raw_final_500 = raw_sig_500[:processed_len_500]
        
        # Resample Denoised (360Hz) -> 500Hz to match Raw
        den_final_500 = resample(den_signal_360, processed_len_500)
        
        # Time Axis (seconds)
        t_axis = np.arange(processed_len_500) / ORIGINAL_FS
        
        plt.figure(figsize=(18, 6))
        plt.plot(t_axis, raw_final_500, label="True Raw (500Hz)", color='blue', alpha=0.6, linewidth=0.8)
        # Scale Denoised to look comparable? No, user requested just raw and denoised amplitudes.
        # But denoised output is Z-scored. Raw is likely in ADC units or mV. They will be on totally different scales.
        # Plotting them together might look weird (one is huge, one is small).
        # We will plot them on different axes? OR just normalize the Raw for plotting purposes?
        # User said: "số liệu biên độ trong các file .csv là của raw trước khi trải qua bất kỳ xử lý nào" -> CSV must have True Values.
        # For Plot: "file ảnh để so sánh". If scales differ by 1000x, plot is useless.
        # Solution: Use Twin Axes for plotting.
        
        ax1 = plt.gca()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Raw Amplitude", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(t_axis, den_final_500, label=f"Denoised (Depth {depth})", color='red', linewidth=1.0)
        ax2.set_ylabel("Denoised Amplitude (Z-score)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f"{filename_base} - Raw (500Hz) vs Denoised (Resampled to 500Hz) - Depth {depth}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        img_path = os.path.join(save_dir, f"{filename_base}_compare.png")
        plt.savefig(img_path, dpi=150)
        plt.close()
        
        # OUTPUT 2: CSV Data File
        csv_data_path = os.path.join(save_dir, f"{filename_base}_data.csv")
        with open(csv_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "True_Raw_500Hz", "Denoised_Resampled_500Hz"])
            for t_val, r_val, d_val in zip(t_axis, raw_final_500, den_final_500):
                writer.writerow([f"{t_val:.4f}", f"{r_val:.6f}", f"{d_val:.6f}"])
        
        print(f"   ✅ Saved plot & data for {filename_base} (Len: {processed_len_500})")


# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    ECG_ROOT = get_ecg_root()
    RAW_DIR  = os.path.join(ECG_ROOT, "datasetBKIC2")
    PICKED_DIR = os.path.join(ECG_ROOT, "datasetBKIC2_picked")
    MODEL_DIR_PATH = os.path.join(ECG_ROOT, "trained_models")
    MAIN_OUT_DIR = os.path.join(ECG_ROOT, "evaluation_outputs_BKIC2_choSon")

    # 1. Get Picked List
    if not os.path.exists(PICKED_DIR):
        print(f"❌ Error: {PICKED_DIR} not found.")
        exit(1)
        
    picked_files = os.listdir(PICKED_DIR)
    # Filter for image files and extract base name
    # Expected format: "filename_Ch8.png" or just "filename.png"
    # We strip the Known Suffixes
    base_names = []
    for f in picked_files:
        if f.startswith("."): continue # Skip hidden
        name, ext = os.path.splitext(f)
        # Often suffix is "_Ch8" or "_Ch9" from visualization script
        if name.endswith("_Ch8"):
            base_name = name[:-4]
        elif name.endswith("_Ch9"):
            base_name = name[:-4]
        else:
            base_name = name
        
        if base_name not in base_names:
            base_names.append(base_name)
    
    print(f"📋 Found {len(base_names)} unique files in picked list: {base_names}")

    # Clear Main Output
    print(f"🔄 Clearing main output directory: {MAIN_OUT_DIR}")
    if os.path.exists(MAIN_OUT_DIR):
        shutil.rmtree(MAIN_OUT_DIR)
    os.makedirs(MAIN_OUT_DIR, exist_ok=True)

    # 2. Evaluate Depth 4
    process_picked_files(4, "TFCNN_depth4_best.pth", 
                   os.path.join(MAIN_OUT_DIR, "TFCNN_depth4"), 
                   RAW_DIR, base_names, MODEL_DIR_PATH)

    # 3. Evaluate Depth 5
    process_picked_files(5, "TFCNN_depth5_best.pth", 
                   os.path.join(MAIN_OUT_DIR, "TFCNN_depth5"), 
                   RAW_DIR, base_names, MODEL_DIR_PATH)
    
    print("\n✅ All ChoSon Tasks Completed.")
