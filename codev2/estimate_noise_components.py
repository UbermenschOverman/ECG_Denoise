import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm
import time

# ===== Config =====
FS_ORIGINAL = 500
TARGET_FS = 360
DATA_DIR = "datasetBKIC"
OUTPUT_FILE = "BKIC_noise_quantification.csv"

class NoiseEstimator:
    def __init__(self, data_dir: str, output_file: str, fs_original: int = FS_ORIGINAL, target_fs: int = TARGET_FS):
        self.data_dir = data_dir
        self.output_file = output_file
        self.fs_original = fs_original
        self.target_fs = target_fs
        self.results = []

    def load_and_resample(self, file_path: str) -> np.ndarray:
        """Reads .txt file and resamples to target_fs."""
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            if df.shape[1] < 9:
                return None
            # Column 8 is usually the lead II or primary signal in this dataset context
            sig_raw = df.iloc[:, 8].values.astype(np.float64)
            
            # Resample
            num_samples = int(len(sig_raw) * self.target_fs / self.fs_original)
            sig_resampled = resample(sig_raw, num_samples)
            return sig_resampled
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def calculate_bw(self, signal: np.ndarray) -> float:
        """
        Quantify Baseline Wander (BW).
        Metric: RMS of Low-Pass filtered signal (< 0.5 Hz).
        """
        nyq = 0.5 * self.target_fs
        cutoff = 0.5
        b, a = butter(4, cutoff / nyq, btype='lowpass')
        bw_component = filtfilt(b, a, signal)
        return np.sqrt(np.mean(bw_component**2))

    def calculate_ma(self, signal: np.ndarray) -> float:
        """
        Quantify Muscle Artifact (MA).
        Metric: RMS of High-Pass filtered signal (> 20 Hz).
        """
        nyq = 0.5 * self.target_fs
        cutoff = 20.0
        b, a = butter(4, cutoff / nyq, btype='highpass')
        ma_component = filtfilt(b, a, signal)
        return np.sqrt(np.mean(ma_component**2))

    def calculate_em(self, signal: np.ndarray) -> float:
        """
        Quantify Electrode Motion (EM).
        Metric: RMS of the first derivative (d/dt).
        Captures rapid transient shifts.
        """
        # diff returns array of length N-1
        diff_sig = np.diff(signal)
        return np.sqrt(np.mean(diff_sig**2))

    def process_all(self):
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return

        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".txt")])
        print(f"Found {len(files)} files in {self.data_dir}")

        for fname in tqdm(files, desc="Quantifying Noise"):
            file_path = os.path.join(self.data_dir, fname)
            record_id = os.path.splitext(fname)[0]

            signal = self.load_and_resample(file_path)
            if signal is None:
                continue

            bw_val = self.calculate_bw(signal)
            ma_val = self.calculate_ma(signal)
            em_val = self.calculate_em(signal)

            self.results.append({
                "record_id": record_id,
                "BW_score": bw_val,
                "MA_score": ma_val,
                "EM_score": em_val
            })

        # Save results
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_file, index=False)
        print(f"\n✅ Noise quantification complete. Results saved to: {self.output_file}")
        print(df.head())

if __name__ == "__main__":
    estimator = NoiseEstimator(DATA_DIR, OUTPUT_FILE)
    estimator.process_all()
