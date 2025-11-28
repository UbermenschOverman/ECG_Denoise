# preprocessing_mitbih_nstdb_stft_test_only.py
import os
import math
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import wfdb
from scipy.signal import butter, filtfilt, stft

# ===================== Config =====================
FS = 360
SEG_LEN = 4096
RECORDS = ["103","105","111","116","122","205","213","219","223","230"]  # MITBIH
NOISE_TYPES = ["bw", "em", "ma"]  # NSTDB
SNR_LEVELS = [-10.0, -7.0, -5.0, -3.0, -1.0, 3.0, 7.0, 10.0]  # dB

# STFT params (không normalize)
STFT_NPERSEG = 8
STFT_NOVERLAP = 7
STFT_WINDOW = "boxcar"
STFT_BOUNDARY = None
STFT_PADDED = False

# Paths
DATA_DIR = "./dataset/mitdb"     # chứa các record sạch MITBIH
NOISE_DIR = "./dataset/nstdb"    # chứa bw.dat, em.dat, ma.dat của NSTDB
OUT_DIR = "./noise"          # sẽ lưu vào OUT_DIR/noisedb/

# Seed tái lập
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== Filters =====================
def bandpass_hp_lp(x: np.ndarray, fs: int = FS, low_hz: float = 0.67, high_hz: float = 100.0) -> np.ndarray:
    nyq = 0.5 * fs
    b_hp, a_hp = butter(4, low_hz / nyq, btype="highpass")
    y = filtfilt(b_hp, a_hp, x)
    b_lp, a_lp = butter(5, high_hz / nyq, btype="lowpass")
    y = filtfilt(b_lp, a_lp, y)
    return y

# ===================== IO helpers =====================
def read_mitbih_record(data_dir: str, rec_name: str) -> Tuple[np.ndarray, int, List[str]]:
    rec = wfdb.rdrecord(os.path.join(data_dir, rec_name))
    fs = int(rec.fs)
    if fs != FS:
        raise ValueError(f"Record {rec_name} fs={fs}Hz khác {FS}Hz. Yêu cầu dữ liệu 360Hz.")
    sig = rec.p_signal
    labels = list(rec.sig_name)
    ch_idx = 0  # luôn kênh đầu tiên
    return sig[:, ch_idx].astype(np.float64), fs, labels

def read_nstdb(noise_dir: str) -> Dict[str, np.ndarray]:
    out = {}
    for n in NOISE_TYPES:
        rec = wfdb.rdrecord(os.path.join(noise_dir, n))
        fs = int(rec.fs)
        if fs != FS:
            raise ValueError(f"NSTDB {n} fs={fs}Hz khác {FS}Hz. Không nội suy theo yêu cầu.")
        x = rec.p_signal[:, 0].astype(np.float64)
        out[n] = x
    return out

# ===================== Core utils =====================
def segment_nonoverlap(x: np.ndarray, seg_len: int = SEG_LEN) -> List[Tuple[int, int]]:
    n = len(x)
    k = n // seg_len
    idx = [(i * seg_len, (i + 1) * seg_len) for i in range(k)]
    return idx

def choose_noise_slice(noise: np.ndarray, seg_len: int) -> np.ndarray:
    if len(noise) < seg_len:
        reps = math.ceil(seg_len / len(noise))
        noise_ext = np.tile(noise, reps)
        return noise_ext[:seg_len].copy()
    start = np.random.randint(0, len(noise) - seg_len + 1)
    return noise[start:start + seg_len].copy()

def scale_noise_for_snr(clean_seg: np.ndarray, noise_seg: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float, float]:
    noise_zm = noise_seg - np.mean(noise_seg)
    Ps = np.mean(clean_seg ** 2) + 1e-12
    Pn0 = np.mean(noise_zm ** 2) + 1e-12
    target_ratio = 10.0 ** (snr_db / 10.0)  # Ps/Pn
    a = math.sqrt(Ps / (Pn0 * target_ratio))
    return a * noise_zm, a, 0.0

def compute_stft_ri(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Z = stft(
        x,
        fs=FS,
        window=STFT_WINDOW,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        boundary=STFT_BOUNDARY,
        padded=STFT_PADDED,
        detrend=False,
        return_onesided=True
    )
    Ri = np.vstack([np.real(Z), np.imag(Z)])  # shape (2*F, T)
    return Ri.astype(np.float32), f.astype(np.float32), t.astype(np.float32)

# ===================== Pipeline =====================
def process_record(
    rec_name: str,
    clean_sig: np.ndarray,
    noises: Dict[str, np.ndarray],
):
    """
    Tạo CHỈ TEST BUCKETS theo từng noise & SNR.
    Áp dụng lọc năng lượng [5,95] trước khi sinh mẫu.
    """
    clean_filt = bandpass_hp_lp(clean_sig, fs=FS)
    segments_all = segment_nonoverlap(clean_filt, SEG_LEN)

    energies = [float(np.sum(clean_filt[s:e] ** 2)) for (s, e) in segments_all]
    if len(energies) == 0:
        return {rec_name: {}}, None, None
    p5, p95 = np.percentile(energies, [5, 95])
    kept = [(s, e) for (s, e), en in zip(segments_all, energies) if (en >= p5 and en <= p95)]

    if len(kept) == 0:
        return {rec_name: {}}, None, None

    test_out: Dict[str, Dict[str, Dict[float, Dict[str, List[torch.Tensor]]]]] = {
        rec_name: {nz: {snr: {"inputs": [], "targets": [], "meta": []} for snr in SNR_LEVELS} for nz in NOISE_TYPES}
    }

    f_ref, t_ref = None, None

    # dùng toàn bộ kept làm test
    for idx, (s, e) in enumerate(kept):
        clean_seg = clean_filt[s:e]
        target_ri, f, t = compute_stft_ri(clean_seg)
        if f_ref is None:
            f_ref, t_ref = f, t

        for nz in NOISE_TYPES:
            base_noise = noises[nz]
            noise_seg = choose_noise_slice(base_noise, SEG_LEN)

            for snr in SNR_LEVELS:
                scaled_noise, a, b = scale_noise_for_snr(clean_seg, noise_seg, snr)
                noisy_seg = clean_seg + scaled_noise

                input_ri, _, _ = compute_stft_ri(noisy_seg)

                meta = {
                    "record": rec_name,
                    "noise": nz,
                    "snr_db": snr,
                    "seg_idx": idx,
                    "s": s,
                    "e": e,
                    "energy": float(np.sum(clean_seg**2)),
                    "pct_energy_bounds": [float(p5), float(p95)],
                    "stft": {"nperseg": STFT_NPERSEG, "noverlap": STFT_NOVERLAP, "window": STFT_WINDOW}
                }

                test_out[rec_name][nz][snr]["inputs"].append(torch.from_numpy(input_ri))
                test_out[rec_name][nz][snr]["targets"].append(torch.from_numpy(target_ri))
                test_out[rec_name][nz][snr]["meta"].append(meta)

    return test_out, f_ref, t_ref

# ===================== Save helpers =====================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_test_bucket(root: str, buckets, f: np.ndarray, t: np.ndarray):
    ensure_dir(root)
    for rec_name, nz_dict in buckets.items():
        for nz, snr_dict in nz_dict.items():
            for snr, data in snr_dict.items():
                if len(data["inputs"]) == 0:
                    continue
                fname = f"{rec_name}_{nz}_{str(snr).replace('.','p')}db.pt"
                path = os.path.join(root, fname)
                pkg = {
                    "inputs": data["inputs"],
                    "targets": data["targets"],
                    "meta": data["meta"],
                    "frequencies": torch.from_numpy(f) if isinstance(f, np.ndarray) else f,
                    "times": torch.from_numpy(t) if isinstance(t, np.ndarray) else t,
                    "fs": FS,
                    "seg_len": SEG_LEN
                }
                torch.save(pkg, path)

# ===================== Main =====================
def main():
    ensure_dir(OUT_DIR)
    out_test_dir = os.path.join(OUT_DIR, "noisedb")
    ensure_dir(out_test_dir)

    noises = read_nstdb(NOISE_DIR)

    test_all_buckets = {}
    f_ref, t_ref = None, None

    for rec in RECORDS:
        clean, fs, labels = read_mitbih_record(DATA_DIR, rec)
        te, f, t = process_record(rec, clean, noises)
        test_all_buckets.update(te)
        if f_ref is None and f is not None:
            f_ref, t_ref = f, t

    if f_ref is None:
        raise RuntimeError("Không có dữ liệu hợp lệ để lưu. Kiểm tra lại nguồn dữ liệu.")

    # Lưu test theo file riêng vào noisedb/
    save_test_bucket(out_test_dir, test_all_buckets, f_ref, t_ref)

    info = {
        "records": RECORDS,
        "noise_types": NOISE_TYPES,
        "snr_levels_db": SNR_LEVELS,
        "fs": FS,
        "seg_len": SEG_LEN,
        "stft": {"nperseg": STFT_NPERSEG, "noverlap": STFT_NOVERLAP, "window": STFT_WINDOW},
        "energy_filter_percentile": [5, 95],
        "channel_used": 0,
        "output_dir": out_test_dir
    }
    with open(os.path.join(out_test_dir, "preprocess_info_noisedb.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("Done.",
          f"\nnoisedb files at: {out_test_dir}")

if __name__ == "__main__":
    main()
