# preprocessing_mitbih_nstdb_stft_test_only.py
import os
import math
import json
import random
from typing import Dict, List, Tuple
import time
from datetime import datetime
import numpy as np
import torch
import wfdb
from scipy.signal import butter, filtfilt, stft
import pandas as pd # Thêm thư viện pandas

# ===================== Config =====================
FS = 360
SEG_LEN = 4096
RECORDS = ["103","105","111","116","122","205","213","219","223","230"] # MITBIH
NOISE_TYPES = ["bw", "em", "ma"] # NSTDB
SNR_LEVELS = [-10.0, -7.0, -5.0, -3.0, -1.0, 3.0, 7.0, 10.0] # dB

# STFT params (không normalize)
STFT_NPERSEG = 8
STFT_NOVERLAP = 7
STFT_WINDOW = "boxcar"
STFT_BOUNDARY = None
STFT_PADDED = False

# Paths
DATA_DIR = "./mitdb" 
NOISE_DIR = "./nstdb" 
# CHỈNH SỬA: Đường dẫn đầu ra mới là ECG/data_noised (do bạn chạy từ thư mục ECG)
OUT_DIR = "data_noised" 
# CHỈNH SỬA: Log sẽ được lưu trong thư mục gốc ECG
SUMMARY_LOG_FILENAME = "noisedb_preprocessing_summary.csv" 

# Seed tái lập
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== Utility =====================
def ensure_dir(path: str):
    """Đảm bảo thư mục tồn tại, tạo nếu chưa có."""
    os.makedirs(path, exist_ok=True)

# ===================== Log Class Mới =====================
class TestPreprocessorLog:
    # CHỈNH SỬA: Cần truyền root_dir để lưu log đúng chỗ
    def __init__(self, log_filename: str = SUMMARY_LOG_FILENAME):
        self.log_filename = log_filename
        self.log_entries = []
        # Không cần xóa file log cũ ở đây vì sẽ thêm timestamp khi lưu

    def add_entry(self, entry: Dict):
        """Thêm một mục log mới."""
        self.log_entries.append(entry)

    def save_log(self, log_root_dir: str):
        """Ghi log tổng hợp ra file CSV."""
        if not self.log_entries:
            print("Không có mục log nào để lưu.")
            return
        
        df_log = pd.DataFrame(self.log_entries)
        # Ghi log với timestamp trong tên file để tránh ghi đè
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sử dụng log_root_dir (là thư mục ECG)
        log_filename = os.path.join(log_root_dir, f"noisedb_preprocessing_summary_{timestamp}.csv") 
        df_log.to_csv(log_filename, index=False)
        print(f"\n📝 Log tổng hợp được lưu tại: {log_filename}")

# ===================== Filters =====================
def bandpass_hp_lp(x: np.ndarray, fs: int = FS, low_hz: float = 0.67, high_hz: float = 100.0) -> np.ndarray:
    nyq = 0.5 * fs
    b_hp, a_hp = butter(4, low_hz / nyq, btype="highpass")
    y = filtfilt(b_hp, a_hp, x)
    b_lp, a_lp = butter(5, high_hz / nyq, btype="lowpass")
    y = filtfilt(b_lp, a_lp, y)
    return y

# ===================== IO Helpers =====================
def read_mitbih_record(data_dir: str, rec_name: str) -> Tuple[np.ndarray, int, np.ndarray]:
    rec = wfdb.rdrecord(os.path.join(data_dir, rec_name))
    fs = int(rec.fs)
    sig = rec.p_signal[:, 0].astype(np.float64)
    return sig, fs, np.array([]) 

def read_nstdb(noise_dir: str) -> Dict[str, np.ndarray]:
    out = {}
    for n in NOISE_TYPES:
        rec = wfdb.rdrecord(os.path.join(noise_dir, n))
        fs = int(rec.fs)
        if fs != FS:
            raise ValueError(f"Noise {n}: fs={fs}, expected {FS}")
        out[n] = rec.p_signal[:, 0].astype(np.float64)
    return out

# ===================== Core Utils =====================
def segment_nonoverlap(x: np.ndarray, seg_len: int = SEG_LEN) -> List[Tuple[int, int]]:
    n = len(x)
    return [(i * seg_len, (i + 1) * seg_len) for i in range(n // seg_len)]

def choose_noise_slice(noise: np.ndarray, seg_len: int) -> np.ndarray:
    if len(noise) < seg_len:
        noise = np.tile(noise, math.ceil(seg_len / len(noise)))
    start = np.random.randint(0, len(noise) - seg_len + 1)
    return noise[start:start + seg_len].copy()

def scale_noise_for_snr(clean_seg: np.ndarray, noise_seg: np.ndarray, snr_db: float) -> np.ndarray:
    noise_zm = noise_seg - np.mean(noise_seg)
    Ps = np.mean(clean_seg ** 2) + 1e-12
    Pn = np.mean(noise_zm ** 2) + 1e-12
    # Công thức SNR (dB) = 10 * log10(Ps / Pn_scaled)
    # => Ps / Pn_scaled = 10^(snr_db / 10)
    # Pn_scaled = a^2 * Pn
    # => a^2 * Pn = Ps / 10^(snr_db / 10)
    # => a = sqrt(Ps / (Pn * 10^(snr_db / 10)))
    a = math.sqrt(Ps / (Pn * 10 ** (snr_db / 10.0))) 
    return a * noise_zm

def compute_stft_ri(x: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    f, t, Z = stft(
        x, fs=FS, window=STFT_WINDOW,
        nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
        boundary=STFT_BOUNDARY, padded=STFT_PADDED, detrend=False
    )
    # Trả về Real và Imaginary part stacked
    ri_data = np.vstack([np.real(Z), np.imag(Z)]).astype(np.float32) 
    return torch.from_numpy(ri_data), f, t

# ===================== Pipeline =====================
def process_record(rec_name: str, clean_sig: np.ndarray, noises: Dict[str, np.ndarray], logger: TestPreprocessorLog) -> Tuple[Dict, np.ndarray, np.ndarray]:
    start_time_record = time.time()
    
    clean_filt = bandpass_hp_lp(clean_sig)
    segments = segment_nonoverlap(clean_filt)
    
    if len(segments) == 0:
        logger.add_entry({"record_id": rec_name, "status": "Too Short", "n_segments_total": 0, "n_segments_kept": 0})
        return {}, None, None

    # Lọc bỏ các segment có năng lượng quá cao hoặc quá thấp (outliers)
    energies = [np.sum(clean_filt[s:e] ** 2) for (s, e) in segments]
    p5, p95 = np.percentile(energies, [5, 95])
    kept = [(s, e) for (s, e), en in zip(segments, energies) if p5 <= en <= p95]
    n_kept = len(kept)
    
    test_out = {rec_name: {n: {s: {"inputs": [], "targets": [], "meta": []} for s in SNR_LEVELS} for n in NOISE_TYPES}}
    f_ref, t_ref = None, None

    for i, (s, e) in enumerate(kept):
        clean_seg = clean_filt[s:e]
        target_ri, f, t = compute_stft_ri(clean_seg)
        if f_ref is None: f_ref, t_ref = f, t
        
        for nz in NOISE_TYPES:
            noise_seg = choose_noise_slice(noises[nz], SEG_LEN)
            for snr in SNR_LEVELS:
                scaled_noise = scale_noise_for_snr(clean_seg, noise_seg, snr)
                noisy_seg = clean_seg + scaled_noise
                input_ri, _, _ = compute_stft_ri(noisy_seg)

                meta = {"record": rec_name, "noise": nz, "snr_db": snr, "seg_idx": i, "s": s, "e": e}
                
                test_out[rec_name][nz][snr]["inputs"].append(input_ri)
                test_out[rec_name][nz][snr]["targets"].append(target_ri)
                test_out[rec_name][nz][snr]["meta"].append(meta)

    runtime_record = time.time() - start_time_record
    
    # Ghi log cho record này (tổng hợp)
    logger.add_entry({
        "record_id": rec_name,
        "status": "Success",
        "dataset": "MITDB",
        "fs": FS,
        "seg_len": SEG_LEN,
        "n_segments_total": len(segments),
        "n_segments_kept": n_kept,
        "n_noise_types": len(NOISE_TYPES),
        "n_snr_levels": len(SNR_LEVELS),
        "n_total_test_samples": n_kept * len(NOISE_TYPES) * len(SNR_LEVELS),
        "runtime_s": f"{runtime_record:.3f}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return test_out, f_ref, t_ref

# ===================== Save Helpers =====================
def save_test_bucket(root: str, buckets, f, t):
    ensure_dir(root)
    total_saved = 0
    for rec_name, nz_dict in buckets.items():
        for nz, snr_dict in nz_dict.items():
            for snr, data in snr_dict.items():
                if len(data["inputs"]) == 0:
                    continue
                # Tên file: rec_noise_snrdb.pt
                # Thay thế dấu thập phân bằng 'p', dấu âm bằng 'n'
                snr_str = str(snr).replace('.', 'p').replace('-', 'n') 
                fname = f"{rec_name}_{nz}_{snr_str}db.pt"
                path = os.path.join(root, fname)
                pkg = {"inputs": data["inputs"], "targets": data["targets"], "meta": data["meta"],
                       "frequencies": torch.from_numpy(f), "times": torch.from_numpy(t),
                       "fs": FS, "seg_len": SEG_LEN}
                torch.save(pkg, path)
                print(f"Saved: {path} ({len(data['inputs'])} segments)")
                total_saved += 1
    print(f"\n✨ Total {total_saved} test files saved.")

# ===================== Main =====================
def main():
    start_time_total = time.time()
    
    # Do bạn chạy từ thư mục ECG, OUT_DIR là data_noised.
    ensure_dir(OUT_DIR)
    # Thư mục con sẽ là data_noised/noisedb
    out_test_dir = os.path.join(OUT_DIR, "noisedb") 
    ensure_dir(out_test_dir)
    
    # Khởi tạo logger. Log sẽ được lưu trong thư mục ECG (thư mục chạy)
    logger = TestPreprocessorLog(SUMMARY_LOG_FILENAME) 

    noises = read_nstdb(NOISE_DIR)

    test_all_buckets = {}
    f_ref, t_ref = None, None

    print(f"🛠️ Bắt đầu tiền xử lý MITDB Records ({len(RECORDS)} records, {len(NOISE_TYPES)} noise types, {len(SNR_LEVELS)} SNR levels)...")
    for rec in RECORDS:
        print(f"-> Processing Record {rec}...")
        clean, fs, _ = read_mitbih_record(DATA_DIR, rec)
        # Truyền logger vào hàm process_record
        te, f, t = process_record(rec, clean, noises, logger) 
        test_all_buckets.update(te)
        if f_ref is None and f is not None:
            f_ref, t_ref = f, t

    if f_ref is None:
        raise RuntimeError("Không có dữ liệu hợp lệ để lưu. Kiểm tra lại nguồn dữ liệu.")

    save_test_bucket(out_test_dir, test_all_buckets, f_ref, t_ref)

    runtime_total = time.time() - start_time_total
    
    # Ghi log tổng hợp sau khi hoàn tất. Log_root_dir là thư mục chạy ("." hoặc "ECG")
    logger.save_log(".") 

    info = {
        "records": RECORDS,
        "noise_types": NOISE_TYPES,
        "snr_levels_db": SNR_LEVELS,
        "fs": FS,
        "seg_len": SEG_LEN,
        "stft": {"nperseg": STFT_NPERSEG, "noverlap": STFT_NOVERLAP, "window": STFT_WINDOW},
        "energy_filter_percentile": [5, 95],
        "channel_used": 0,
        "output_dir": out_test_dir,
        "total_runtime_s": f"{runtime_total:.3f}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(out_test_dir, "preprocess_info_noisedb.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\n✅ Hoàn tất! (Tổng thời gian:", f"{runtime_total:.2f}s)")
    print(f"Dữ liệu test tại: {out_test_dir}")

if __name__ == "__main__":
    main()