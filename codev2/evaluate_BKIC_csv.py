import os
import glob
import numpy as np
import torch
import csv
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

# Input config (BKIC Dataset)
ORIGINAL_FS = 500

# Filtering Params
HP_CUTOFF = 0.67
LP_CUTOFF = 100.0
HP_ORDER = 4
LP_ORDER = 5

# Metrics
def calculate_metrics(raw_normalized, denoised_normalized):
    """
    Calculate metrics.
    1. RMSE: Between Filtered Input (Reference) and Denoised Output.
    2. PRD: Percent Root Mean Square Difference.
    3. SNRimp (Robust): 10 * log10( Noise_Var_In / Noise_Var_Out )
       - Noise Variance is estimated using MAD of the high-frequency component (diff).
    """
    # 1. RMSE
    mse = np.mean((raw_normalized - denoised_normalized) ** 2)
    rmse = np.sqrt(mse)

    # 2. PRD (Percent Root Mean Square Difference)
    norm_ref = np.linalg.norm(raw_normalized)
    diff_norm = np.linalg.norm(raw_normalized - denoised_normalized)
    prd = 100.0 * (diff_norm / (norm_ref + 1e-6))

    # 3. Robust SNR Improvement (Noise Reduction in dB)
    # Estimate noise standard deviation using MAD of the derivative (high-pass proxy)
    def estimate_noise_std(x):
        # Median Absolute Deviation of the first difference
        sigma = np.median(np.abs(np.diff(x))) / 0.6745
        return sigma

    sigma_in = estimate_noise_std(raw_normalized)
    sigma_out = estimate_noise_std(denoised_normalized)

    # Avoid div by zero
    if sigma_out == 0:
        snr_imp = 100.0 # High improvement
    else:
        # SNRimp = 20 * log10(sigma_in / sigma_out)  (since power ~ sigma^2)
        # Power Ratio: (sigma_in^2) / (sigma_out^2)
        # dB = 10 * log10(Power_Ratio) = 20 * log10(Sigma_Ratio)
        snr_imp = 20.0 * np.log10((sigma_in + 1e-9) / (sigma_out + 1e-9))

    return rmse, prd, snr_imp

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

def read_ecg_txt(path):
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
    if not os.path.exists(dir_path):
        return
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path): os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# ========================
# Main Model Evaluator
# ========================
def evaluate_model(depth, model_filename, output_subdir, raw_dir, model_dir_path):
    print(f"\n========================================")
    print(f"🚀 Starting Evaluation for Depth {depth}")
    print(f"   Model: {model_filename}")
    print(f"   Output: {output_subdir}")
    print(f"========================================")

    # Setup Paths
    save_dir = output_subdir
    os.makedirs(save_dir, exist_ok=True)
    
    # Init CSV
    csv_path = os.path.join(save_dir, "metrics.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Filename", "RMSE", "PRD (%)", "SNRimp (dB)"]) # Header

    model_path = os.path.join(model_dir_path, model_filename)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Get Subjects
    # We use raw_dir to find files. Assumption: files are .txt in raw_dir
    # Or based on previous scripts, datasetBKIC contains .txt files
    txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not txt_files:
        print(f"⚠️ No .txt files found in {raw_dir}")
        return

    all_rmse = []
    all_prd = []
    all_snr = []

    for file_path in txt_files:
        subj = os.path.basename(file_path).replace(".txt", "")
        # print(f"Processing {subj}...")
        
        try:
            raw_sig_500 = read_ecg_txt(file_path)
        except Exception as e:
            print(f"Skipping {subj}: {e}")
            continue

        # 1. Resample
        num_samples_360 = int(len(raw_sig_500) * TARGET_FS / ORIGINAL_FS)
        if num_samples_360 == 0: continue
        raw_sig_360 = resample(raw_sig_500, num_samples_360)

        # 2. Normalize
        mu = np.mean(raw_sig_360)
        std = np.std(raw_sig_360) + 1e-6
        raw_norm = (raw_sig_360 - mu) / std

        # 3. Filter
        filtered_norm = bandpass_filter(raw_norm, TARGET_FS)

        # 4. Segment & Infer
        SEG_LEN = 4096
        n = len(filtered_norm)
        k = n // SEG_LEN
        
        # We need to construct the FULL denoised signal to compare with FULL raw signal
        full_denoised = []
        
        # Determine segments
        segments = []
        if k == 0:
            segments.append((0, n))
        else:
            # Main segments
            full_segments = [(i * SEG_LEN, (i + 1) * SEG_LEN) for i in range(k)]
            segments.extend(full_segments)
            # Remainder (if any significant length)
            # Standard logic usually drops remainder or pads. Let's pad last chunk if needed or just use segments.
            # For accurate metrics, let's just stick to the segmented parts to avoid padding artifacts in metrics.
            
        full_ref_concat = []
        full_den_concat = []

        for i, (s, e) in enumerate(segments):
            seg = filtered_norm[s:e]
            seg_len_actual = e - s
            
            # STFT
            stft_ri = compute_stft_ri_2ch(seg)
            inp_tensor = torch.from_numpy(stft_ri).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred = model(inp_tensor)
                pred = pred.squeeze().cpu().numpy()
            
            real_FT, imag_FT = split_ri(pred)
            wave_chunk = istft_from_ri(real_FT, imag_FT)

            # Match length
            if len(wave_chunk) > seg_len_actual:
                wave_chunk = wave_chunk[:seg_len_actual]
            elif len(wave_chunk) < seg_len_actual:
                pad = np.zeros(seg_len_actual - len(wave_chunk))
                wave_chunk = np.concatenate([wave_chunk, pad])
            
            full_ref_concat.append(seg)
            full_den_concat.append(wave_chunk)

        if not full_ref_concat:
            continue

        ref_signal = np.concatenate(full_ref_concat)
        den_signal = np.concatenate(full_den_concat)

        # Calculate Metrics
        rmse, prd, snr = calculate_metrics(ref_signal, den_signal)
        
        all_rmse.append(rmse)
        all_prd.append(prd)
        all_snr.append(snr)

        csv_writer.writerow([subj, f"{rmse:.4f}", f"{prd:.4f}", f"{snr:.4f}"])

    csv_file.close()

    # Calculate Average
    avg_rmse = np.mean(all_rmse) if all_rmse else 0
    avg_prd = np.mean(all_prd) if all_prd else 0
    avg_snr = np.mean(all_snr) if all_snr else 0

    print(f"Finished {model_filename}. Avg RMSE: {avg_rmse:.4f}, Avg PRD: {avg_prd:.4f}, Avg SNRimp: {avg_snr:.4f}")

    # Write Summary
    summary_path = os.path.join(save_dir, "summary_average.csv")
    with open(summary_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Average Value"])
        writer.writerow(["RMSE", f"{avg_rmse:.6f}"])
        writer.writerow(["PRD (%)", f"{avg_prd:.6f}"])
        writer.writerow(["SNRimp (dB)", f"{avg_snr:.6f}"])
    
    print(f"📄 Saved csv to {csv_path}")
    print(f"📄 Saved summary to {summary_path}")


# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    ECG_ROOT = get_ecg_root()
    RAW_DIR  = os.path.join(ECG_ROOT, "datasetBKIC")
    MODEL_DIR_PATH = os.path.join(ECG_ROOT, "trained_models")
    MAIN_OUT_DIR = os.path.join(ECG_ROOT, "evaluation_outputs_BKIC_csv")

    # Clear Main Output
    print(f"🔄 Clearing main output directory: {MAIN_OUT_DIR}")
    if os.path.exists(MAIN_OUT_DIR):
        shutil.rmtree(MAIN_OUT_DIR)
    os.makedirs(MAIN_OUT_DIR, exist_ok=True)

    # 1. Evaluate Depth 4
    evaluate_model(4, "TFCNN_depth4_best.pth", 
                   os.path.join(MAIN_OUT_DIR, "TFCNN_depth4"), 
                   RAW_DIR, MODEL_DIR_PATH)

    # 2. Evaluate Depth 5
    evaluate_model(5, "TFCNN_depth5_best.pth", 
                   os.path.join(MAIN_OUT_DIR, "TFCNN_depth5"), 
                   RAW_DIR, MODEL_DIR_PATH)
    
    print("\n✅ All Evaluations Completed.")
