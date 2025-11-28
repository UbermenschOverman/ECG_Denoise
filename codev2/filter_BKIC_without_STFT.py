import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, medfilt
from tqdm import tqdm

# ===== Config =====
FS = 360
SEG_LEN = 4096


class Preprocessor:
    def __init__(self, data_dir: str, output_dir: str,
                 fs_original: int = 500, target_fs: int = FS):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.fs_original = fs_original
        self.target_fs = target_fs

        # Xóa toàn bộ nội dung cũ nếu thư mục output đã tồn tại
        if os.path.exists(self.output_dir):
            print(f"🧹 Xóa nội dung cũ trong {self.output_dir} ...")
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ===== Loại bỏ baseline wander (median filter + highpass) =====
    def remove_baseline(self, x: np.ndarray, fs: int,
                        median_kernel_s: float = 0.6, hp_cutoff: float = 0.5, hp_order: int = 4):
        """Loại bỏ baseline wander bằng median filter + high-pass filter."""
        # --- median filter ---
        k = int(round(median_kernel_s * fs))
        if k % 2 == 0:
            k += 1
        baseline = medfilt(x, kernel_size=k)
        x_detrend = x - baseline

        # --- high-pass filter ---
        nyq = 0.5 * fs
        b_hp, a_hp = butter(hp_order, hp_cutoff / nyq, btype="highpass")
        y = filtfilt(b_hp, a_hp, x_detrend)

        return y

    # ===== Lọc thông dải (sau khi loại baseline) =====
    def bandpass_filter(self, data: np.ndarray, fs: int) -> np.ndarray:
        nyq = 0.5 * fs
        b_hp, a_hp = butter(4, 0.67 / nyq, btype="highpass")
        y = filtfilt(b_hp, a_hp, data)
        b_lp, a_lp = butter(5, 100.0 / nyq, btype="lowpass")
        y = filtfilt(b_lp, a_lp, y)
        return y

    # ===== Chia đoạn không chồng lắp =====
    def segment_nonoverlap(self, x: np.ndarray, seg_len: int = SEG_LEN):
        n = len(x)
        k = n // seg_len
        return [(i * seg_len, (i + 1) * seg_len) for i in range(k)]

    # ===== Xử lý tất cả file .txt =====
    def process_all_txt_files(self):
        txt_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".txt")])
        if not txt_files:
            print(f"⚠️ Không tìm thấy file .txt trong {self.data_dir}")
            return

        print(f"📁 Đang xử lý {len(txt_files)} file trong {self.data_dir}")
        print(f"📤 Kết quả sẽ lưu tại: {self.output_dir}")

        for fname in tqdm(txt_files, desc="Filtering BKIC"):
            fpath = os.path.join(self.data_dir, fname)
            try:
                df = pd.read_csv(fpath, delim_whitespace=True, header=None)
                if df.shape[1] < 9:
                    print(f"⚠️ Bỏ qua {fname}: ít hơn 9 cột.")
                    continue

                sig = df.iloc[:, 8].values.astype(np.float64)

                # --- Nội suy về tần số lấy mẫu mục tiêu ---
                resampled = resample(sig, int(round(len(sig) * self.target_fs / self.fs_original)))

                # --- Loại bỏ baseline wander ---
                no_baseline = self.remove_baseline(resampled, self.target_fs)

                # --- Lọc thông dải ---
                filtered = self.bandpass_filter(no_baseline, self.target_fs)

                # --- Lưu waveform ---
                base = os.path.splitext(fname)[0]
                file_dir = os.path.join(self.output_dir, base)
                os.makedirs(file_dir, exist_ok=True)

                plt.figure(figsize=(10, 4))
                plt.plot(resampled, label="Original (resampled)", alpha=0.6)
                plt.plot(filtered, label="Filtered (no baseline + bandpass)", linewidth=1)
                plt.title(f"{base} - Filtered waveform")
                plt.xlabel("Sample index")
                plt.ylabel("Amplitude")
                plt.legend(loc="upper right")
                plt.tight_layout()

                fig_path = os.path.join(file_dir, f"{base}_filtered_waveform.png")
                plt.savefig(fig_path, dpi=150)
                plt.close()

            except Exception as e:
                print(f"❌ Lỗi khi xử lý {fname}: {e}")

        print(f"\n✅ Hoàn tất! Kết quả nằm tại: {self.output_dir}")


# ===== RUN =====
if __name__ == "__main__":
    processor = Preprocessor(
        data_dir="datasetBKIC",
        output_dir="datasetBKIC_filtered_without_STFT"
    )
    processor.process_all_txt_files()
