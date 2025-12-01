import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, medfilt, stft
from tqdm import tqdm
import time
from datetime import datetime
import json
from typing import Tuple, List

# ===== Config =====
FS = 360
SEG_LEN = 4096
STFT_NPERSEG = 8
STFT_NOVERLAP = 7
STFT_WINDOW = "boxcar"
STFT_BOUNDARY = None
STFT_PADDED = False
PLOT_FILTERED_WAVEFORM = True  # Bật/tắt việc lưu waveform sau lọc

# Hằng số filter (để lưu vào log)
HP_CUTOFF = 0.67
HP_ORDER = 4
LP_CUTOFF = 100.0
LP_ORDER = 5


class Preprocessor:
    def __init__(self, data_dir: str, output_dir: str,
                 fs_original: int = 500, target_fs: int = FS,
                 summary_log_path: str = "preprocessing_summary_BKIC.csv"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.fs_original = fs_original
        self.target_fs = target_fs
        self.summary_log_path = summary_log_path
        self.processed_records_log = []

        # Xóa toàn bộ nội dung cũ nếu thư mục output đã tồn tại
        if os.path.exists(self.output_dir):
            print(f"🧹 Xóa nội dung cũ trong {self.output_dir} ...")
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Log: Xóa log cũ nếu tồn tại để chỉ chứa kết quả của lần chạy mới nhất
        if os.path.exists(self.summary_log_path):
             os.remove(self.summary_log_path)


    # ===== Hàm tính SNR ước tính (dùng cho tín hiệu THÔ) =====
    def compute_snr_estimate(self, signal: np.ndarray) -> Tuple[float, float]:
        """Ước tính SNR theo độ lệch chuẩn. Giả định noise là phần còn lại sau khi lọc baseline/lowpass."""
        if len(signal) < 1000: # Ngắn quá không ước tính
            return 0.0, 0.0

        # Lọc baseband: Sử dụng highpass/bandpass nhẹ để tách tín hiệu ECG
        nyq = 0.5 * self.target_fs
        # Lọc lowpass mạnh để lấy thành phần nhiễu tần số cao (MA)
        b_hp, a_hp = butter(4, 3.0 / nyq, btype="highpass")
        clean_estimate = filtfilt(b_hp, a_hp, signal) 

        # Noise estimate: Phần còn lại sau khi trừ đi ước tính sạch
        noise_estimate = signal - clean_estimate
        
        # P_signal = mean(clean_estimate^2), P_noise = mean(noise_estimate^2)
        P_s = np.mean(clean_estimate ** 2) + 1e-12
        P_n = np.mean(noise_estimate ** 2) + 1e-12
        
        snr_db = 10 * np.log10(P_s / P_n)
        return snr_db, np.sqrt(P_n) # Trả về SNR (dB) và RMS Noise (mV)


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
        b_hp, a_hp = butter(HP_ORDER, HP_CUTOFF / nyq, btype="highpass")
        y = filtfilt(b_hp, a_hp, data)
        b_lp, a_lp = butter(LP_ORDER, LP_CUTOFF / nyq, btype="lowpass")
        y = filtfilt(b_lp, a_lp, y)
        return y

    # ===== Chia đoạn không chồng lắp =====
    def segment_nonoverlap(self, x: np.ndarray, seg_len: int = SEG_LEN) -> List[Tuple[int, int]]:
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

    # ===== Ghi log tổng hợp vào CSV =====
    def log_summary(self):
        df_log = pd.DataFrame(self.processed_records_log)
        # Ghi log với timestamp trong tên file để tránh ghi đè kết quả của các lần chạy khác nhau
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"BKIC_preprocessing_summary_{timestamp}.csv"
        df_log.to_csv(log_filename, index=False)
        print(f"\n📝 Log tổng hợp được lưu tại: {log_filename}")

    # ===== Xử lý tất cả file .txt =====
    def process_all_txt_files(self):
        start_time_total = time.time()
        txt_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".txt")])
        
        # ... (Phần hiển thị thông báo) ...

        for fname in tqdm(txt_files, desc="Processing BKIC"):
            start_time_record = time.time()
            record_id = os.path.splitext(fname)[0]
            fpath = os.path.join(self.data_dir, fname)
            
            try:
                df = pd.read_csv(fpath, delim_whitespace=True, header=None)
                if df.shape[1] < 9:
                    print(f"⚠️ Bỏ qua {fname}: ít hơn 9 cột.")
                    continue

                # --- Đọc tín hiệu thô ---
                sig_raw = df.iloc[:, 8].values.astype(np.float64)
                len_raw = len(sig_raw)
                
                # --- Nội suy ---
                resampled = resample(sig_raw, int(round(len_raw * self.target_fs / self.fs_original)))
                len_resampled = len(resampled)
                
                # --- TÍNH SNR ƯỚC TÍNH (TRÊN TÍN HIỆU RESAMPLED) ---
                snr_db_estimate_raw, rms_noise_estimate_raw = self.compute_snr_estimate(resampled)

                # --- Loại bỏ baseline wander ---
                no_baseline = self.remove_baseline(resampled, self.target_fs)

                # --- Lọc thông dải ---
                clean = self.bandpass_filter(no_baseline, self.target_fs)
                len_filtered = len(clean)

                # --- Chia đoạn & tính STFT ---
                segments = self.segment_nonoverlap(clean, SEG_LEN)
                n_segments = len(segments)
                
                # --- (Tuỳ chọn) Lưu waveform sau lọc ---
                # ... (Giữ nguyên logic plotting) ...
                
                base = os.path.splitext(fname)[0]
                file_dir = os.path.join(self.output_dir, base)
                os.makedirs(file_dir, exist_ok=True)
                
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

                if not segments:
                    # ... (Bỏ qua logic) ...
                    continue

                for k, (s, e) in enumerate(segments):
                    seg = clean[s:e]
                    stft_ri, f, t = self.compute_stft_ri(seg)
                    
                    # ---- Lưu STFT + Vẽ spectrogram ----
                    # ... (Giữ nguyên logic saving/plotting STFT) ...
                    
                    npy_path = os.path.join(file_dir, f"seg{k:03d}_stft.npy")
                    np.save(npy_path, stft_ri)

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
                
                # --- Ghi log cho record này ---
                runtime_record = time.time() - start_time_record
                log_entry = {
                    "record_id": record_id,
                    "dataset": "BKIC",
                    "fs": self.target_fs,
                    "fs_original": self.fs_original,
                    "bandpass_params": f"HP:{HP_CUTOFF}Hz, LP:{LP_CUTOFF}Hz, Order:{HP_ORDER}/{LP_ORDER}",
                    "stft_params": f"N={STFT_NPERSEG}, OVL={STFT_NOVERLAP}, W={STFT_WINDOW}",
                    "segment_len": SEG_LEN,
                    "n_segments": n_segments,
                    "len_raw_original": len_raw,
                    "len_filtered_resampled": len_filtered,
                    "snr_estimate_raw_db": f"{snr_db_estimate_raw:.3f}",
                    "rms_noise_estimate_raw": f"{rms_noise_estimate_raw:.3e}",
                    "runtime_s": f"{runtime_record:.3f}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.processed_records_log.append(log_entry)
                
                # --- LƯU PREPROCESS INFO JSON TỪNG RECORD ---
                with open(os.path.join(file_dir, "preprocess_info.json"), "w") as f:
                    json.dump(log_entry, f, indent=2)

            except Exception as e:
                print(f"❌ Lỗi khi xử lý {fname}: {e}")
                
        # --- KẾT THÚC VÀ GHI LOG TỔNG HỢP ---
        runtime_total = time.time() - start_time_total
        print(f"\n✅ Hoàn tất! Tổng thời gian: {runtime_total:.2f}s. Kết quả tại: {self.output_dir}")
        self.log_summary()


# ===== RUN (giữ nguyên) =====
if __name__ == "__main__":
    processor = Preprocessor(
        data_dir="datasetBKIC",
        output_dir="datasetBKIC_preprocessed"
    )
    processor.process_all_txt_files()