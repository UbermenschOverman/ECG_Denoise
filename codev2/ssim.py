# eval_batch_all_tests_metrics_and_plots.py
# for evaluation stage
import os
import csv
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.signal import istft, butter, filtfilt, find_peaks
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from stft_dataset import STFTDataset
from model import HybridSTFT_LIRA

# ===== Config =====
TEST_DIR   = "./codev2/noise/noisedb"
MODEL_PATH = "codev2/checkpointsv2/mit22.pth"
RES_DIR    = "codev2/ssim2"
FIG_DIR    = os.path.join(RES_DIR, "figs")
FS        = 360
SEG_LEN   = 4096
NPERSEG   = 8
NOVERLAP  = 7
WINDOW    = "boxcar"

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ===== Helpers: STFT, chuẩn hoá, ảnh =====
def split_ri(stft_img_2F_T: np.ndarray):
    Fdim = stft_img_2F_T.shape[0] // 2
    return stft_img_2F_T[:Fdim, :], stft_img_2F_T[Fdim:, :]

def istft_from_ri(real_FT: np.ndarray, imag_FT: np.ndarray):
    Zxx = real_FT + 1j * imag_FT
    _, x = istft(Zxx, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW)
    if len(x) < SEG_LEN:
        x = np.pad(x, (0, SEG_LEN - len(x)), mode="edge")
    else:
        x = x[:SEG_LEN]
    return x

def to_mag01(ri_2F_T: np.ndarray):
    r, i = split_ri(ri_2F_T)
    mag = np.sqrt(r**2 + i**2)
    mn, mx = mag.min(), mag.max()
    if mx > mn:
        mag = (mag - mn) / (mx - mn)
    else:
        mag = np.zeros_like(mag)
    return mag

# ===== Ảnh: SSIM, MAE, PSNR =====
def mae_img(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean(np.abs(a - b)))

def psnr_img(a, b, data_range=1.0):
    a = np.asarray(a); b = np.asarray(b)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-15:
        return 100.0
    return 10.0 * math.log10((data_range ** 2) / mse)

def ssim_img(a, b, data_range=1.0):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h < 3 or w < 3:
        return float('nan')
    k = min(7, h, w)
    if k % 2 == 0:
        k -= 1
    return float(structural_similarity(a[:h, :w], b[:h, :w],
                                       data_range=data_range,
                                       win_size=3,
                                       channel_axis=None))

# ===== ECG: lọc, R-peak, chỉ số =====
def bandpass(signal, fs, low=5.0, high=15.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(sig, fs):
    x = bandpass(sig, fs, 5, 15, order=3)
    dx = np.diff(x, prepend=x[0])
    energy = (dx ** 2)
    win = max(1, int(0.15 * fs))
    kernel = np.ones(win) / win
    env = np.convolve(energy, kernel, mode='same')
    height = 0.3 * np.max(env) if np.max(env) > 0 else None
    distance = max(1, int(0.2 * fs))
    peaks, _ = find_peaks(env, height=height, distance=distance)
    return peaks

def match_peaks(ref_peaks, est_peaks, fs, tol_ms=50):
    tol = int((tol_ms/1000.0) * fs)
    ref_used = np.zeros(len(ref_peaks), dtype=bool)
    tp = 0
    fp = 0
    for p in est_peaks:
        idx = np.where(np.abs(ref_peaks - p) <= tol)[0]
        if len(idx) > 0:
            d = np.abs(ref_peaks[idx] - p)
            j = idx[np.argmin(d)]
            if not ref_used[j]:
                ref_used[j] = True
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = np.sum(~ref_used)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1

def rr_hr_errors(ref_peaks, est_peaks, fs):
    n = min(len(ref_peaks), len(est_peaks))
    if n < 3:
        return float('nan'), float('nan')
    ref_rr = np.diff(ref_peaks[:n]) / fs
    est_rr = np.diff(est_peaks[:n]) / fs
    m = min(len(ref_rr), len(est_rr))
    if m == 0:
        return float('nan'), float('nan')
    rr_mae = float(np.mean(np.abs(ref_rr[:m] - est_rr[:m])) * 1000.0)
    ref_hr = 60.0 / np.clip(ref_rr[:m], 1e-6, None)
    est_hr = 60.0 / np.clip(est_rr[:m], 1e-6, None)
    hr_mae = float(np.mean(np.abs(ref_hr - est_hr)))
    return rr_mae, hr_mae

def st_deviation_mae(ref_sig, est_sig, ref_peaks, est_peaks, fs):
    if len(ref_peaks) < 1 or len(est_peaks) < 1:
        return float('nan')
    n = min(len(ref_peaks), len(est_peaks))
    j_off = int(0.04 * fs)
    pr_l = int(0.28 * fs)
    pr_r = int(0.20 * fs)
    vals = []
    for i in range(n):
        r_ref = ref_peaks[i]
        r_est = est_peaks[i]
        bl_l = max(0, r_ref - pr_l)
        bl_r = max(0, r_ref - pr_r)
        if bl_r <= bl_l:
            continue
        base_ref = float(np.median(ref_sig[bl_l:bl_r]))
        base_est = float(np.median(est_sig[max(0, r_est - pr_l):max(0, r_est - pr_r)]))
        j_ref = min(len(ref_sig)-1, r_ref + j_off)
        j_est = min(len(est_sig)-1, r_est + j_off)
        st_ref = float(ref_sig[j_ref] - base_ref)
        st_est = float(est_sig[j_est] - base_est)
        vals.append(abs(st_ref - st_est))
    if not vals:
        return float('nan')
    return float(np.mean(vals))

def prd_td(clean, rec):
    clean = np.asarray(clean); rec = np.asarray(rec)
    num = np.linalg.norm(clean - rec)
    den = np.linalg.norm(clean) + 1e-12
    return float(100.0 * num / den)

# ===== Meta =====
def parse_meta(meta, fname):
    rec = noise = None; snr_db = None
    if isinstance(meta, dict):
        rec = meta.get("record", None)
        noise = meta.get("noise", None)
        snr_db = meta.get("snr_db", None)
    if (rec is None or noise is None or snr_db is None) and isinstance(fname, str):
        base = os.path.basename(fname).replace(".pt", "")
        parts = base.split("_")
        if len(parts) >= 3:
            rec = parts[0]
            noise = parts[1]
            s = parts[2].replace("db", "").replace("p", ".")
            try:
                snr_db = float(s)
            except:
                snr_db = None
    return rec, noise, snr_db

# ===== Plot =====
def plot_segment(fig_path, clean_td, noisy_td, denoise_td, clean_mag, pred_mag, r_clean, r_pred):
    plt.figure(figsize=(12, 8))
    t = np.arange(len(clean_td)) / FS
    ax1 = plt.subplot(3,1,1)
    ax1.plot(t, noisy_td, label="Noisy", linewidth=0.7)
    ax1.plot(t, denoise_td, label="Denoised", linewidth=0.8)
    ax1.plot(t, clean_td, label="Clean", linewidth=0.8)
    ax1.scatter(r_clean/FS, clean_td[r_clean], marker='o', s=10, label="R clean")
    ax1.scatter(r_pred/FS, denoise_td[r_pred], marker='x', s=12, label="R denoised")
    ax1.set_title("Time-domain overlay")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amp"); ax1.legend(loc="upper right", fontsize=8)

    ax2 = plt.subplot(3,2,3)
    ax2.imshow(clean_mag, aspect='auto', origin='lower')
    ax2.set_title("STFT |Clean|"); ax2.set_xlabel("Frames"); ax2.set_ylabel("Freq bins")

    ax3 = plt.subplot(3,2,4)
    ax3.imshow(pred_mag, aspect='auto', origin='lower')
    ax3.set_title("STFT |Denoised|"); ax3.set_xlabel("Frames"); ax3.set_ylabel("Freq bins")

    ax4 = plt.subplot(3,2,5)
    diff = np.abs(clean_mag - pred_mag)
    ax4.imshow(diff, aspect='auto', origin='lower')
    ax4.set_title("|Clean|-|Denoised|"); ax4.set_xlabel("Frames"); ax4.set_ylabel("Freq bins")

    ax5 = plt.subplot(3,2,6)
    ax5.plot(t, clean_td - denoise_td, linewidth=0.8)
    ax5.set_title("Time error"); ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Error")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

# ===== Eval một file =====
def eval_file(pt_path, model, make_plots=True):
    dataset = STFTDataset(pt_path)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    per_seg_rows = []
    saved_imgs = 0

    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            x = batch["input"].to(DEVICE).float()
            y = batch["target"].to(DEVICE).float()

            y_pred = model(x)
            if y_pred.shape[-2:] != x.shape[-2:]:
                y_pred = F.interpolate(y_pred, size=x.shape[-2:], mode="bilinear", align_corners=False)

            x_np     = x[0,0].cpu().numpy()
            y_np     = y[0,0].cpu().numpy()
            ypred_np = y_pred[0,0].cpu().numpy()

            y_mag  = to_mag01(y_np)
            yp_mag = to_mag01(ypred_np)

            ssim_v = ssim_img(y_mag, yp_mag, data_range=1.0)
            mae_v  = mae_img(y_mag, yp_mag)
            psnr_v = psnr_img(y_mag, yp_mag, data_range=1.0)

            xr, xi = split_ri(x_np)
            yr, yi = split_ri(y_np)
            pr, pi = split_ri(ypred_np)

            noisy_td   = istft_from_ri(xr, xi)
            clean_td   = istft_from_ri(yr, yi)
            denoise_td = istft_from_ri(pr, pi)

            r_clean = detect_r_peaks(clean_td, FS)
            r_pred  = detect_r_peaks(denoise_td, FS)
            tp, fp, fn, prec, rec, f1 = match_peaks(r_clean, r_pred, FS, tol_ms=50)
            rr_mae_ms, hr_mae_bpm = rr_hr_errors(r_clean, r_pred, FS)
            st_mae = st_deviation_mae(clean_td, denoise_td, r_clean, r_pred, FS)
            prd_v  = prd_td(clean_td, denoise_td)

            row = {
                "seg_idx": i-1,
                "ssim": ssim_v,
                "mae":  mae_v,
                "psnr": psnr_v,
                "prd":  prd_v,
                "r_tp": tp,
                "r_fp": fp,
                "r_fn": fn,
                "r_precision": prec,
                "r_recall":    rec,
                "r_f1":        f1,
                "rr_mae_ms":   rr_mae_ms,
                "hr_mae_bpm":  hr_mae_bpm,
                "stdev_mae":   st_mae,
            }

            meta = batch.get("meta", None)
            if isinstance(meta, list) and len(meta) == 1 and isinstance(meta[0], dict):
                rec, nz, snr_db = parse_meta(meta[0], pt_path)
            else:
                rec, nz, snr_db = parse_meta(None, pt_path)
            row.update({"record": rec, "noise": nz, "snr_db": snr_db})
            per_seg_rows.append(row)

            if make_plots and saved_imgs < 5:
                base = os.path.splitext(os.path.basename(pt_path))[0]
                fig_path = os.path.join(FIG_DIR, f"{base}_seg{i-1}.png")
                plot_segment(fig_path, clean_td, noisy_td, denoise_td, y_mag, yp_mag, r_clean, r_pred)
                saved_imgs += 1

    return per_seg_rows

# ===== CSV =====
def write_csv(rows, csv_path):
    if not rows:
        return
    keys = [
        "record","noise","snr_db","seg_idx",
        "ssim","mae","psnr","prd",
        "r_tp","r_fp","r_fn","r_precision","r_recall","r_f1",
        "rr_mae_ms","hr_mae_bpm","stdev_mae"
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def mean_or_nan(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")

def summarize_rows(rows):
    out = {
        "n_segments": len(rows),
        "ssim":   mean_or_nan([r["ssim"]   for r in rows]),
        "mae":    mean_or_nan([r["mae"]    for r in rows]),
        "psnr":   mean_or_nan([r["psnr"]   for r in rows]),
        "prd":    mean_or_nan([r["prd"]    for r in rows]),
        "r_precision": mean_or_nan([r["r_precision"] for r in rows]),
        "r_recall":    mean_or_nan([r["r_recall"]    for r in rows]),
        "r_f1":        mean_or_nan([r["r_f1"]        for r in rows]),
        "rr_mae_ms":   mean_or_nan([r["rr_mae_ms"]   for r in rows]),
        "hr_mae_bpm":  mean_or_nan([r["hr_mae_bpm"]  for r in rows]),
        "stdev_mae":   mean_or_nan([r["stdev_mae"]   for r in rows]),
    }
    if rows:
        out["record"] = rows[0].get("record", "")
        out["noise"]  = rows[0].get("noise", "")
        out["snr_db"] = rows[0].get("snr_db", "")
    return out

# ===== Main =====
def main():
    model = HybridSTFT_LIRA(input_channels=1, output_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    files = [f for f in os.listdir(TEST_DIR) if f.endswith(".pt")]
    files.sort()

    summary_rows = []
    overall_rows = []

    for fn in files:
        pt_path = os.path.join(TEST_DIR, fn)
        try:
            rows = eval_file(pt_path, model, make_plots=True)
            if not rows:
                print(f"skip empty: {fn}")
                continue

            seg_csv = os.path.join(RES_DIR, f"{os.path.splitext(fn)[0]}_segments.csv")
            write_csv(rows, seg_csv)
            print(f"saved per-segment: {seg_csv}")

            summ = summarize_rows(rows)
            summ["file"] = fn
            summary_rows.append(summ)

            overall_rows.extend(rows)
        except Exception as e:
            print(f"error on {fn}: {e}")

    if summary_rows:
        keys = ["file","record","noise","snr_db","n_segments",
                "ssim","mae","psnr","prd",
                "r_precision","r_recall","r_f1",
                "rr_mae_ms","hr_mae_bpm","stdev_mae"]
        with open(os.path.join(RES_DIR, "summary_per_file.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"saved: {os.path.join(RES_DIR, 'summary_per_file.csv')}")

    if overall_rows:
        ov = summarize_rows(overall_rows)
        ov["file"] = "ALL"
        keys = ["file","record","noise","snr_db","n_segments",
                "ssim","mae","psnr","prd",
                "r_precision","r_recall","r_f1",
                "rr_mae_ms","hr_mae_bpm","stdev_mae"]
        with open(os.path.join(RES_DIR, "summary_overall.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerow({k: ov.get(k, "") for k in keys})
        print(f"saved: {os.path.join(RES_DIR, 'summary_overall.csv')}")

if __name__ == "__main__":
    main()
