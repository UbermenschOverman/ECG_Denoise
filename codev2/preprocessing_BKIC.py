import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, medfilt, stft
from tqdm import tqdm

# ===== Config =====
FS = 360
SEG_LEN = 4096
STFT_NPERSEG = 8
STFT_NOVERLAP = 7
STFT_WINDOW = "boxcar"
STFT_BOUNDARY = None
STFT_PADDED = False
PLOT_FILTERED_WAVEFORM = True  # Bật/tắt việc lưu waveform sau lọc


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
        k = int(round(median_kernel_s * fs))
        if k % 2 == 0:
            k += 1
        baseline = medfilt(x, kernel_size=k)
        x_detrend = x - baseline

        nyq = 0.5 * fs
        b_hp, a_hp = butter(hp_order, hp_cutoff / nyq, btype="highpass")
        y = filtfilt(b_hp, a_hp, x_detrend)
        return y

    # ===== Lọc thông dải =====
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

    # ===== Tính STFT (real + imag) =====
    def compute_stft_ri(self, x: np.ndarray):
        f, t, Z = stft(
            x, fs=self.target_fs, window=STFT_WINDOW,
            nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
            boundary=STFT_BOUNDARY, padded=STFT_PADDED,
            detrend=False, return_onesided=True
        )
        Ri = np.vstack([np.real(Z), np.imag(Z)])  # (2F, T)
        return Ri.astype(np.float32), f.astype(np.float32), t.astype(np.float32)

    # ===== Xử lý tất cả file .txt =====
    def process_all_txt_files(self):
        txt_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".txt")])
        if not txt_files:
            print(f"⚠️ Không tìm thấy file .txt trong {self.data_dir}")
            return

        print(f"📁 Đang xử lý {len(txt_files)} file trong {self.data_dir}")
        print(f"📤 Kết quả sẽ lưu tại: {self.output_dir}")

        for fname in tqdm(txt_files, desc="Processing BKIC"):
            fpath = os.path.join(self.data_dir, fname)
            try:
                df = pd.read_csv(fpath, delim_whitespace=True, header=None)
                if df.shape[1] < 9:
                    print(f"⚠️ Bỏ qua {fname}: ít hơn 9 cột.")
                    continue

                # --- Đọc tín hiệu và nội suy ---
                sig = df.iloc[:, 8].values.astype(np.float64)
                resampled = resample(sig, int(round(len(sig) * self.target_fs / self.fs_original)))

                # --- Loại bỏ baseline wander ---
                no_baseline = self.remove_baseline(resampled, self.target_fs)

                # --- Lọc thông dải ---
                clean = self.bandpass_filter(no_baseline, self.target_fs)

                # --- Thư mục riêng cho mỗi file ---
                base = os.path.splitext(fname)[0]
                file_dir = os.path.join(self.output_dir, base)
                os.makedirs(file_dir, exist_ok=True)

                # --- (Tuỳ chọn) Lưu waveform sau lọc ---
                if PLOT_FILTERED_WAVEFORM:
                    plt.figure(figsize=(10, 4))
                    plt.plot(resampled, label="Original (resampled)", alpha=0.5)
                    plt.plot(no_baseline, label="After baseline removal", alpha=0.7)
                    plt.plot(clean, label="Final filtered", linewidth=1)
                    plt.legend(loc="upper right")
                    plt.title(f"{base} - Filter stages")
                    plt.xlabel("Sample index")
                    plt.ylabel("Amplitude")
                    plt.tight_layout()
                    plt.savefig(os.path.join(file_dir, f"{base}_filtered_waveform.png"), dpi=150)
                    plt.close()

                # --- Chia đoạn & tính STFT ---
                segments = self.segment_nonoverlap(clean, SEG_LEN)
                if not segments:
                    print(f"⚠️ {fname} quá ngắn, bỏ qua.")
                    continue

                for k, (s, e) in enumerate(segments):
                    seg = clean[s:e]
                    stft_ri, f, t = self.compute_stft_ri(seg)

                    # ---- Lưu STFT ----
                    npy_path = os.path.join(file_dir, f"seg{k:03d}_stft.npy")
                    np.save(npy_path, stft_ri)

                    # ---- Vẽ spectrogram ----
                    magnitude = np.abs(
                        stft_ri[:len(stft_ri)//2, :] +
                        1j * stft_ri[len(stft_ri)//2:, :]
                    )

                    plt.figure(figsize=(8, 4))
                    plt.imshow(
                        20 * np.log10(magnitude + 1e-6),
                        aspect="auto", origin="lower",
                        extent=[t[0], t[-1], f[0], f[-1]],
                        cmap="turbo"
                    )
                    plt.colorbar(label="Magnitude (dB)")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Frequency (Hz)")
                    plt.title(f"{base} - Segment {k}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(file_dir, f"seg{k:03d}_spec.png"), dpi=150)
                    plt.close()

            except Exception as e:
                print(f"❌ Lỗi khi xử lý {fname}: {e}")

        print(f"\n✅ Hoàn tất! Kết quả nằm tại: {self.output_dir}")


# ===== RUN =====
if __name__ == "__main__":
    processor = Preprocessor(
        data_dir="datasetBKIC",
        output_dir="datasetBKIC_preprocessed"
    )
    processor.process_all_txt_files()
