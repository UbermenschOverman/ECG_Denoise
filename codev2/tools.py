import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, medfilt

# ========================
# Config
# ========================
TARGET_FS = 360
ORIGINAL_FS = 500

# Task specific Config: (Filename, StartSample, EndSample) - Input @ 500Hz
TASKS = [
    ("170425_ste_thuonghtk_1.txt", 2000, 3500),
    ("170625_ste_thuonghtk_3(1).txt", 900, 2400),
    ("270525_ste_anhtd2(1).txt", 1500, 3000),
    ("270524_ste_thuong2.txt", 7400, 9200)
]

# Filtering Params
HP_CUTOFF = 0.5   
HP_ORDER_BL = 4   
BP_HP = 0.67      
BP_LP = 100.0     
MEDIAN_KERNEL_S = 0.6 

# Calibration
ADC_GAIN = 15000.0 

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

def read_ecg_txt(path):
    try:
        # Use delim_whitespace=True for faster parsing (c engine usually) or sep='\s+'
        # But 'python' engine is slow. Let's try default engine with delim_whitespace=True despite warning if it works.
        # Actually, best is creating a custom separator or using fixed width if structure is known.
        # But let's stick to delim_whitespace=True which was used in original script and worked fast enough there.
        # The warning is annoyance but performance matters.
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        if df.shape[1] < 9:
            print(f"Skipping {path}: Less than 9 columns", flush=True)
            return None
        return df.iloc[:, 8].values.astype(np.float64)
    except Exception as e:
        print(f"Error loading {path}: {e}", flush=True)
        return None

def remove_baseline(x: np.ndarray, fs: int):
    k = int(round(MEDIAN_KERNEL_S * fs))
    if k % 2 == 0: k += 1
    baseline = medfilt(x, kernel_size=k)
    x_detrend = x - baseline
    nyq = 0.5 * fs
    b_hp, a_hp = butter(HP_ORDER_BL, HP_CUTOFF / nyq, btype="highpass")
    y = filtfilt(b_hp, a_hp, x_detrend)
    return y

def bandpass_filter(data: np.ndarray, fs: int) -> np.ndarray:
    nyq = 0.5 * fs
    b_hp, a_hp = butter(4, BP_HP / nyq, btype="highpass")
    y = filtfilt(b_hp, a_hp, data)
    b_lp, a_lp = butter(5, BP_LP / nyq, btype="lowpass")
    y = filtfilt(b_lp, a_lp, y)
    return y

# ========================
# Main
# ========================
if __name__ == "__main__":
    ECG_ROOT = get_ecg_root()
    OUT_DIR = os.path.join(ECG_ROOT, "choSon")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for filename, start_samp, end_samp in TASKS:
        RAW_FILE = os.path.join(ECG_ROOT, "datasetBKIC", filename)
        
        print(f"\n🚀 Processing {filename} | Range [{start_samp}:{end_samp}]")
        
        # 1. Read
        if not os.path.exists(RAW_FILE):
            print(f"❌ File not found: {RAW_FILE}")
            continue
            
        raw_500 = read_ecg_txt(RAW_FILE)
        if raw_500 is None: continue
            
        # 2. Resample
        num_samples_360 = int(round(len(raw_500) * TARGET_FS / ORIGINAL_FS))
        raw_360 = resample(raw_500, num_samples_360)
        
        # 3. Process Full Signal
        sig_no_baseline = remove_baseline(raw_360, TARGET_FS)
        sig_filtered = bandpass_filter(sig_no_baseline, TARGET_FS)
        
        # 4. Extract Segment
        start_idx_360 = int(start_samp * TARGET_FS / ORIGINAL_FS)
        end_idx_360   = int(end_samp * TARGET_FS / ORIGINAL_FS)
        
        if end_idx_360 > len(raw_360):
            print(f"⚠️ Warning: Segment end {end_idx_360} exceeds signal length {len(raw_360)}. Truncating.")
            end_idx_360 = len(raw_360)
            
        print(f"   Mapping: [{start_samp}:{end_samp}] (500Hz) -> [{start_idx_360}:{end_idx_360}] (360Hz)")
        
        seg_filt = sig_filtered[start_idx_360:end_idx_360]
        seg_raw = raw_360[start_idx_360:end_idx_360]
        
        # 5. Apply Calibration (Counts -> mV)
        seg_raw_mv = seg_raw / ADC_GAIN 
        seg_filt_mv = seg_filt / ADC_GAIN
        
        # 6. Apply Normalization (SKIPPED as requested)
        # Segments are now directly in mV
        
        time_axis = np.arange(len(seg_filt)) / TARGET_FS 
        
        # 7. Save Output
        base_name = os.path.splitext(filename)[0]
        
        # A. Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, seg_raw_mv, label="Original (mV)", color='blue', alpha=0.5, linewidth=1.0)
        plt.plot(time_axis, seg_filt_mv, label="Denoised (mV)", color='red', linewidth=1.2)
        plt.title(f"{filename} (Sample {start_samp}-{end_samp})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        img_path = os.path.join(OUT_DIR, f"{base_name}_seg_{start_samp}_{end_samp}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()
        # print(f"✅ Saved plot: {img_path}")
        
        # B. CSV
        csv_path = os.path.join(OUT_DIR, f"{base_name}_seg_{start_samp}_{end_samp}.csv")
        df_out = pd.DataFrame({
            "Time_s": time_axis,
            "Raw_mV": seg_raw_mv,
            "Denoised_mV": seg_filt_mv
        })
        df_out.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV & Plot for {filename}")
