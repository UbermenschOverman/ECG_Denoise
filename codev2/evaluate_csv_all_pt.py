# eval_batch_all_tests.py
import os
import csv
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.signal import istft

from stft_dataset import STFTDataset
from model import HybridSTFT_LIRA

# ===== Config =====
TEST_DIR   = "./datasetv2/test"
MODEL_PATH = "codev2/checkpointsv2/mit32.pth"
RES_DIR    = "noisedb_resual_model"            # nơi lưu CSV
FS        = 360
SEG_LEN   = 4096
NPERSEG   = 8
NOVERLAP  = 7
WINDOW    = "boxcar"

os.makedirs(RES_DIR, exist_ok=True)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ===== Helpers =====
def split_ri(stft_img_2F_T: np.ndarray):
    assert stft_img_2F_T.ndim == 2, f"expect (2F,T), got {stft_img_2F_T.shape}"
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

def mse(a, b):  return float(np.mean((np.asarray(a) - np.asarray(b))**2))
def rmse(a, b): return float(math.sqrt(mse(a, b)))
def snr(clean, test):
    clean = np.asarray(clean); test = np.asarray(test)
    n = test - clean
    Ps = np.mean(clean**2) + 1e-12
    Pn = np.mean(n**2) + 1e-12
    return 10.0 * np.log10(Ps/Pn)
def snri(noisy, denoised, clean): return snr(clean, denoised) - snr(clean, noisy)
def cosine_sim(a, b):
    a = np.asarray(a); b = np.asarray(b)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den
def prd(clean, rec):
    clean = np.asarray(clean); rec = np.asarray(rec)
    return float(100.0 * np.linalg.norm(clean - rec) / (np.linalg.norm(clean) + 1e-12))

def parse_meta(meta, fname):
    rec = noise = None; snr_db = None
    if isinstance(meta, dict):
        rec = meta.get("record", None)
        noise = meta.get("noise", None)
        snr_db = meta.get("snr_db", None)
    # fallback từ tên file: e.g., 103_bw_0p0db.pt
    if (rec is None or noise is None or snr_db is None) and isinstance(fname, str):
        base = os.path.basename(fname).replace(".pt", "")
        parts = base.split("_")
        if len(parts) >= 3:
            rec = parts[0]
            noise = parts[1]
            # "0p0db" -> "0.0"
            snr_db = parts[2].replace("db", "").replace("p", ".")
            try:
                snr_db = float(snr_db)
            except:
                snr_db = None
    return rec, noise, snr_db

def eval_file(pt_path, model):
    dataset = STFTDataset(pt_path)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    per_seg_rows = []
    # chạy
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            x = batch["input"].to(DEVICE).float()   # (B,1,2F,T)
            y = batch["target"].to(DEVICE).float()

            y_pred = model(x)
            if y_pred.shape[-2:] != x.shape[-2:]:
                y_pred = F.interpolate(y_pred, size=x.shape[-2:], mode="bilinear", align_corners=False)

            x_np     = x[0,0].cpu().numpy()
            y_np     = y[0,0].cpu().numpy()
            ypred_np = y_pred[0,0].cpu().numpy()

            xr, xi   = split_ri(x_np)
            yr, yi   = split_ri(y_np)
            pr, pi   = split_ri(ypred_np)

            noisy_td    = istft_from_ri(xr, xi)
            clean_td    = istft_from_ri(yr, yi)
            denoise_td  = istft_from_ri(pr, pi)

            row = {
                "seg_idx": i-1,
                "mse":   mse(clean_td, denoise_td),
                "rmse":  rmse(clean_td, denoise_td),
                "snri":  snri(noisy_td, denoise_td, clean_td),
                "cosine":cosine_sim(clean_td, denoise_td),
                "prd":   prd(clean_td, denoise_td),
            }
            # meta nếu có
            meta = batch.get("meta", None)
            if isinstance(meta, list) and len(meta) == 1 and isinstance(meta[0], dict):
                rec, nz, snr_db = parse_meta(meta[0], pt_path)
            else:
                rec, nz, snr_db = parse_meta(None, pt_path)
            row.update({"record": rec, "noise": nz, "snr_db": snr_db})
            per_seg_rows.append(row)

    return per_seg_rows

def write_csv(rows, csv_path):
    if not rows:
        return
    keys = ["record","noise","snr_db","seg_idx","mse","rmse","snri","cosine","prd"]
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
        "mse":   mean_or_nan([r["mse"]   for r in rows]),
        "rmse":  mean_or_nan([r["rmse"]  for r in rows]),
        "snri":  mean_or_nan([r["snri"]  for r in rows]),
        "cosine":mean_or_nan([r["cosine"]for r in rows]),
        "prd":   mean_or_nan([r["prd"]   for r in rows]),
    }
    # điền thông tin định danh nếu có
    if rows:
        out["record"] = rows[0].get("record", "")
        out["noise"]  = rows[0].get("noise", "")
        out["snr_db"] = rows[0].get("snr_db", "")
    return out

def main():
    # load model 1 lần
    model = HybridSTFT_LIRA(input_channels=1, output_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # liệt kê các file test
    files = [f for f in os.listdir(TEST_DIR) if f.endswith(".pt")]
    files.sort()

    summary_rows = []
    overall_rows = []

    for fn in files:
        pt_path = os.path.join(TEST_DIR, fn)
        try:
            rows = eval_file(pt_path, model)
            if not rows:
                print(f"skip empty: {fn}")
                continue
            # ghi per-segment
            seg_csv = os.path.join(RES_DIR, f"{os.path.splitext(fn)[0]}_segments.csv")
            write_csv(rows, seg_csv)
            print(f"saved per-segment: {seg_csv}")

            # tóm tắt theo file
            summ = summarize_rows(rows)
            summ["file"] = fn
            summary_rows.append(summ)

            # tích lũy cho overall
            overall_rows.extend(rows)
        except Exception as e:
            print(f"error on {fn}: {e}")

    # ghi summary per-file
    if summary_rows:
        keys = ["file","record","noise","snr_db","n_segments","mse","rmse","snri","cosine","prd"]
        with open(os.path.join(RES_DIR, "summary_per_file.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"saved: {os.path.join(RES_DIR, 'summary_per_file.csv')}")

    # ghi overall (một dòng chung)
    if overall_rows:
        ov = summarize_rows(overall_rows)
        ov["file"] = "ALL"
        keys = ["file","record","noise","snr_db","n_segments","mse","rmse","snri","cosine","prd"]
        with open(os.path.join(RES_DIR, "summary_overall.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerow({k: ov.get(k, "") for k in keys})
        print(f"saved: {os.path.join(RES_DIR, 'summary_overall.csv')}")

if __name__ == "__main__":
    main()
