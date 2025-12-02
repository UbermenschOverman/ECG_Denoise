import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample, medfilt, find_peaks
from tqdm import tqdm
import time
from typing import Tuple

# ===== Config =====
FS_ORIGINAL = 500
TARGET_FS = 360
DATA_DIR = "datasetBKIC"
OUTPUT_FILE = "BKIC_noise_quantification_v2.csv"
# Cutoffs cho phân tích nhiễu
BW_MAX_FREQ = 0.5  # Hz
MA_MIN_FREQ = 25.0 # Hz (thường dùng 25 Hz để tránh QRS)

class NoiseEstimator:
    def __init__(self, data_dir: str, output_file: str, fs_original: int = FS_ORIGINAL, target_fs: int = TARGET_FS):
        self.data_dir = data_dir
        self.output_file = output_file
        self.fs_original = fs_original
        self.target_fs = target_fs
        self.results = []
        
    def load_and_resample(self, file_path: str) -> np.ndarray:
        # Giữ nguyên logic load và resample
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            if df.shape[1] < 9:
                return None
            sig_raw = df.iloc[:, 8].values.astype(np.float64)
            num_samples = int(len(sig_raw) * self.target_fs / self.fs_original)
            sig_resampled = resample(sig_raw, num_samples)
            
            # CHUẨN HÓA Z-SCORE ĐỂ SO SÁNH ĐƯỢC VỚI CÁC DATASET KHÁC (BẤT KỂ GAIN)
            if np.std(sig_resampled) > 1e-6:
                sig_resampled = (sig_resampled - np.mean(sig_resampled)) / np.std(sig_resampled)
            else:
                sig_resampled = sig_resampled - np.mean(sig_resampled)
            return sig_resampled
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def remove_baseline(self, x: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Loại bỏ BW bằng median filter. Trả về tín hiệu sạch ước tính và thành phần BW."""
        # Kernel size 0.6s * 360Hz = 216 -> làm tròn thành 217
        k = int(round(0.6 * fs)) 
        if k % 2 == 0: k += 1
            
        baseline = medfilt(x, kernel_size=k)
        x_detrend = x - baseline
        return x_detrend, baseline # x_detrend: Tín hiệu đã khử BW; baseline: Ước tính BW

    def calculate_bw(self, signal: np.ndarray) -> float:
        """
        Quantify Baseline Wander (BW).
        Metric: RMS of the estimated baseline component.
        """
        _, baseline_estimate = self.remove_baseline(signal, self.target_fs)
        # Sóng R/QRS có thể ảnh hưởng đến median filter. 
        # Cần xác nhận baseline estimate này là nhiễu, không phải sóng R
        # Cách an toàn hơn: Lọc tín hiệu đã detrended (đã khử BW) với highpass nhẹ để tách noise (đã dùng trong preprocessing)
        
        # Phương pháp 1: Dùng RMS của ước tính Baseline (Chính xác hơn phương pháp cũ)
        return np.sqrt(np.mean(baseline_estimate**2))

    def calculate_ma(self, signal: np.ndarray) -> float:
        """
        Quantify Muscle Artifact (MA).
        Metric: RMS of the high-frequency component (> 25 Hz) after baseline removal.
        """
        # Bước 1: Loại bỏ Baseline Wander
        signal_no_bw, _ = self.remove_baseline(signal, self.target_fs)
        
        # Bước 2: Lọc High-Pass để tách MA (MA_MIN_FREQ = 25 Hz)
        nyq = 0.5 * self.target_fs
        cutoff = MA_MIN_FREQ
        # Lọc high-pass để lấy các thành phần > 25 Hz (MA + QRS cao tần)
        b, a = butter(4, cutoff / nyq, btype='highpass')
        high_freq_component = filtfilt(b, a, signal_no_bw)
        
        # NOTE: Thành phần này vẫn bao gồm QRS. Để giảm ảnh hưởng của QRS, có thể cần zero-out QRS hoặc dùng lọc nâng cao hơn.
        # Ở đây, ta chấp nhận rằng MA_score sẽ là chỉ số tổng hợp nhiễu cao tần (MA) và độ sắc nét QRS.
        return np.sqrt(np.mean(high_freq_component**2))

    def calculate_em(self, signal: np.ndarray) -> float:
        """
        Quantify Electrode Motion (EM).
        Metric: RMS of the first derivative (d/dt) (đã chuẩn hóa).
        """
        diff_sig = np.diff(signal)
        # Chuẩn hóa đạo hàm theo tần số lấy mẫu để dễ so sánh giữa các dataset
        normalized_diff = diff_sig * self.target_fs 
        return np.sqrt(np.mean(normalized_diff**2))

    def calculate_total_noise_index(self, signal: np.ndarray) -> float:
        """
        Tính chỉ số nhiễu tổng thể (Total Noise Index - TNR).
        Metric: SNR (dB) của tín hiệu.
        """
        # Ước tính tín hiệu sạch (lọc bandpass nhẹ)
        nyq = 0.5 * self.target_fs
        b_bp, a_bp = butter(4, [0.67 / nyq, 35.0 / nyq], btype='bandpass') 
        clean_estimate = filtfilt(b_bp, a_bp, signal)
        
        # Ước tính nhiễu (phần còn lại)
        noise_estimate = signal - clean_estimate
        
        P_s = np.mean(clean_estimate ** 2) + 1e-12
        P_n = np.mean(noise_estimate ** 2) + 1e-12
        
        snr_db = 10 * np.log10(P_s / P_n)
        return snr_db, np.sqrt(P_n) # Trả về SNR (dB) và RMS Noise (mV)

    def process_all(self):
        # ... (Giữ nguyên logic load files) ...
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return

        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".txt")])
        print(f"Found {len(files)} files in {self.data_dir}")

        for fname in tqdm(files, desc="Quantifying Noise"):
            file_path = os.path.join(self.data_dir, fname)
            record_id = os.path.splitext(fname)[0]

            signal = self.load_and_resample(file_path)
            if signal is None or len(signal) < 1000:
                continue
            
            # --- Tính toán các chỉ số ---
            # 1. BW (RMS của thành phần Baseline)
            bw_val = self.calculate_bw(signal)
            # 2. MA (RMS của thành phần cao tần sau khi khử BW)
            ma_val = self.calculate_ma(signal)
            # 3. EM (RMS của đạo hàm bậc một)
            em_val = self.calculate_em(signal)
            # 4. Total Noise (SNR và RMS Noise)
            snr_db, rms_noise = self.calculate_total_noise_index(signal)
            
            self.results.append({
                "record_id": record_id,
                "BW_RMS": f"{bw_val:.3f}",
                "MA_RMS": f"{ma_val:.3f}",
                "EM_RMS_deriv": f"{em_val:.3f}",
                "SNR_estimate_dB": f"{snr_db:.3f}",
                "RMS_Noise_mV": f"{rms_noise:.3e}",
            })

        # Save results
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_file, index=False)
        print(f"\n✅ Noise quantification (V2) complete. Results saved to: {self.output_file}")
        print(df.head())

if __name__ == "__main__":
    estimator = NoiseEstimator(DATA_DIR, OUTPUT_FILE)
    estimator.process_all()