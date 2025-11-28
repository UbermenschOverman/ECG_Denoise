import os
import math
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import wfdb
from scipy.signal import butter, filtfilt, stft
from tqdm import tqdm

# ===================== Config =====================
FS = 360
SEG_LEN = 4096

BASE_DIR = "/mnt/sda1/home/sparc/ducnguyen/ECG"
DATA_DIR = os.path.join(BASE_DIR, "mitdb")
NOISE_DIR = os.path.join(BASE_DIR, "nstdb")
OUT_DIR = os.path.join(BASE_DIR, "preprocessed_mitdb")

# lấy danh sách record MIT-BIH
RECORDS = [f.split(".")[0] for f in os.listdir(DATA_DIR) if f.endswith(".hea")]

NOISE_TYPES = ["bw", "em", "ma"]
SNR_LEVELS = [0.0, 1.25, 5.0]

# STFT params
STFT_NPERSEG = 8
STFT_NOVERLAP = 7
STFT_WINDOW = "boxcar"
STFT_BOUNDARY = None
STFT_PADDED = False

# Seed cố định
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== Utility =====================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def bandpass_hp_lp(x: np.ndarray, fs: int = FS, low_hz: float = 0.67, high_hz: float = 100.0):
    nyq = 0.5 * fs
    b_hp, a_hp = butter(4, low_hz / nyq, btype="highpass")
    y = filtfilt(b_hp, a_hp, x)
    b_lp, a_lp = butter(5, high_hz / nyq, btype="lowpass")
    return filtfilt(b_lp, a_lp, y)

def read_mitbih_record(data_dir: str, rec_name: str):
    rec = wfdb.rdrecord(os.path.join(data_dir, rec_name))
    fs = int(rec.fs)
    if fs != FS:
        raise ValueError(f"{rec_name}: fs={fs}, expected {FS}")
    sig = rec.p_signal[:, 0].astype(np.float64)
    return sig, fs

def read_nstdb(noise_dir: str) -> Dict[str, np.ndarray]:
    out = {}
    for n in NOISE_TYPES:
        rec = wfdb.rdrecord(os.path.join(noise_dir, n))
        fs = int(rec.fs)
        if fs != FS:
            raise ValueError(f"Noise {n}: fs={fs}, expected {FS}")
        out[n] = rec.p_signal[:, 0].astype(np.float64)
    return out

def segment_nonoverlap(x: np.ndarray, seg_len: int = SEG_LEN):
    n = len(x)
    return [(i * seg_len, (i + 1) * seg_len) for i in range(n // seg_len)]

def choose_noise_slice(noise: np.ndarray, seg_len: int):
    if len(noise) < seg_len:
        noise = np.tile(noise, math.ceil(seg_len / len(noise)))
    start = np.random.randint(0, len(noise) - seg_len + 1)
    return noise[start:start + seg_len].copy()

def scale_noise_for_snr(clean_seg: np.ndarray, noise_seg: np.ndarray, snr_db: float):
    noise_zm = noise_seg - np.mean(noise_seg)
    Ps = np.mean(clean_seg ** 2) + 1e-12
    Pn = np.mean(noise_zm ** 2) + 1e-12
    a = math.sqrt(Ps / (Pn * 10 ** (snr_db / 10.0)))
    return a * noise_zm

def compute_stft_ri(x: np.ndarray):
    f, t, Z = stft(
        x, fs=FS, window=STFT_WINDOW,
        nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
        boundary=STFT_BOUNDARY, padded=STFT_PADDED, detrend=False
    )
    return np.vstack([np.real(Z), np.imag(Z)]).astype(np.float32), f, t

# ===================== Core =====================
def process_record(rec_name: str, clean_sig: np.ndarray, noises: Dict[str, np.ndarray]):
    clean_filt = bandpass_hp_lp(clean_sig)
    segments = segment_nonoverlap(clean_filt)
    energies = [np.sum(clean_filt[s:e] ** 2) for (s, e) in segments]
    if len(energies) == 0:
        return None
    p5, p95 = np.percentile(energies, [5, 95])
    kept = [(s, e) for (s, e), en in zip(segments, energies) if p5 <= en <= p95]

    rng = random.Random(SEED + hash(rec_name) % (10**6))
    idxs = list(range(len(kept)))
    rng.shuffle(idxs)
    n_train = int(0.8 * len(kept))
    n_val = int(0.1 * len(kept))
    train_idx = set(idxs[:n_train])
    val_idx = set(idxs[n_train:n_train + n_val])

    train_out, val_out, test_out = {"inputs": [], "targets": [], "meta": []}, {"inputs": [], "targets": [], "meta": []}, {rec_name: {n: {s: {"inputs": [], "targets": [], "meta": []} for s in SNR_LEVELS} for n in NOISE_TYPES}}
    f_ref, t_ref = None, None

    print(f"📊 Processing record {rec_name} ({len(kept)} segments)...")
    for i, (s, e) in enumerate(tqdm(kept, desc=f"{rec_name}", ncols=80)):
        clean_seg = clean_filt[s:e]
        target_ri, f, t = compute_stft_ri(clean_seg)
        if f_ref is None: f_ref, t_ref = f, t
        bucket = "train" if i in train_idx else ("val" if i in val_idx else "test")

        for nz in NOISE_TYPES:
            noise_seg = choose_noise_slice(noises[nz], SEG_LEN)
            for snr in SNR_LEVELS:
                scaled_noise = scale_noise_for_snr(clean_seg, noise_seg, snr)
                noisy_seg = clean_seg + scaled_noise
                input_ri, _, _ = compute_stft_ri(noisy_seg)

                meta = {"record": rec_name, "noise": nz, "snr_db": snr, "seg_idx": i, "s": s, "e": e}
                if bucket == "train":
                    train_out["inputs"].append(torch.from_numpy(input_ri))
                    train_out["targets"].append(torch.from_numpy(target_ri))
                    train_out["meta"].append(meta)
                elif bucket == "val":
                    val_out["inputs"].append(torch.from_numpy(input_ri))
                    val_out["targets"].append(torch.from_numpy(target_ri))
                    val_out["meta"].append(meta)
                else:
                    test_out[rec_name][nz][snr]["inputs"].append(torch.from_numpy(input_ri))
                    test_out[rec_name][nz][snr]["targets"].append(torch.from_numpy(target_ri))
                    test_out[rec_name][nz][snr]["meta"].append(meta)

    return train_out, val_out, test_out, f_ref, t_ref

# ===================== Save =====================
def save_split(filename: str, split: Dict, f, t):
    ensure_dir(os.path.dirname(filename) or ".")
    pkg = {"inputs": split["inputs"], "targets": split["targets"], "meta": split["meta"],
           "frequencies": torch.from_numpy(f), "times": torch.from_numpy(t),
           "fs": FS, "seg_len": SEG_LEN}
    torch.save(pkg, filename)

def save_test_bucket(root: str, buckets, f, t):
    ensure_dir(root)
    for rec_name, nz_dict in buckets.items():
        for nz, snr_dict in nz_dict.items():
            for snr, data in snr_dict.items():
                if len(data["inputs"]) == 0:
                    continue
                fname = f"{rec_name}_{nz}_{str(snr).replace('.', 'p')}db.pt"
                path = os.path.join(root, fname)
                pkg = {"inputs": data["inputs"], "targets": data["targets"], "meta": data["meta"],
                       "frequencies": torch.from_numpy(f), "times": torch.from_numpy(t),
                       "fs": FS, "seg_len": SEG_LEN}
                torch.save(pkg, path)

# ===================== Main =====================
def main():
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "test"))

    noises = read_nstdb(NOISE_DIR)
    all_train, all_val, all_test, f_ref, t_ref = {"inputs": [], "targets": [], "meta": []}, {"inputs": [], "targets": [], "meta": []}, {}, None, None

    for rec in RECORDS:
        clean, _ = read_mitbih_record(DATA_DIR, rec)
        result = process_record(rec, clean, noises)
        if result is None: 
            continue
        tr, va, te, f, t = result
        all_train["inputs"].extend(tr["inputs"])
        all_train["targets"].extend(tr["targets"])
        all_train["meta"].extend(tr["meta"])
        all_val["inputs"].extend(va["inputs"])
        all_val["targets"].extend(va["targets"])
        all_val["meta"].extend(va["meta"])
        all_test.update(te)
        if f_ref is None: f_ref, t_ref = f, t
        print(f"✅ Done record {rec}")

    save_split(os.path.join(OUT_DIR, "train.pt"), all_train, f_ref, t_ref)
    save_split(os.path.join(OUT_DIR, "val.pt"), all_val, f_ref, t_ref)
    save_test_bucket(os.path.join(OUT_DIR, "test"), all_test, f_ref, t_ref)

    with open(os.path.join(OUT_DIR, "preprocess_info.json"), "w") as f:
        json.dump({
            "records": RECORDS,
            "noise_types": NOISE_TYPES,
            "snr_levels_db": SNR_LEVELS,
            "fs": FS,
            "seg_len": SEG_LEN,
            "stft": {"nperseg": STFT_NPERSEG, "noverlap": STFT_NOVERLAP, "window": STFT_WINDOW}
        }, f, indent=2)

    print(f"✅ train.pt, val.pt, test saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()