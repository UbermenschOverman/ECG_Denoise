import os
import glob
import torch
import numpy as np
import pandas as pd
from scipy.signal import istft
from tqdm import tqdm
import torch.nn.functional as F
from model import HybridSTFT_LIRA

# ===== Config =====
MODEL_PATH = "trained_models/mit22.pth"
TEST_DIR = "preprocessed_mitdb/test"
OUTPUT_CSV = "denoising_results_summary_mitdb.csv"

FS = 360
SEG_LEN = 4096
STFT_NPERSEG = 8
STFT_NOVERLAP = 7
STFT_WINDOW = "boxcar"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Metrics =====
def calculate_snr(clean, noise):
    p_s = np.mean(clean**2) + 1e-12
    p_n = np.mean(noise**2) + 1e-12
    return 10 * np.log10(p_s / p_n)

def calculate_metrics(clean, noisy, denoised):
    # RMSE
    rmse = np.sqrt(np.mean((clean - denoised)**2))
    
    # PSNR
    max_val = np.max(clean) - np.min(clean)
    mse = np.mean((clean - denoised)**2) + 1e-12
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    # SNRI
    # SNR_in = SNR(clean, noisy-clean)
    # SNR_out = SNR(clean, denoised-clean)
    snr_in = calculate_snr(clean, noisy - clean)
    snr_out = calculate_snr(clean, denoised - clean)
    snri = snr_out - snr_in
    
    return snri, rmse, psnr

# ===== Helper: ISTFT =====
def istft_from_tensor(stft_tensor):
    # stft_tensor: (2F, T)
    stft_np = stft_tensor.cpu().numpy()
    F_dim = stft_np.shape[0] // 2
    real = stft_np[:F_dim, :]
    imag = stft_np[F_dim:, :]
    Zxx = real + 1j * imag
    _, x = istft(Zxx, fs=FS, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP, window=STFT_WINDOW)
    # Pad or crop to SEG_LEN
    if len(x) < SEG_LEN:
        x = np.pad(x, (0, SEG_LEN - len(x)), mode='edge')
    else:
        x = x[:SEG_LEN]
    return x

# ===== Main Evaluation Loop =====
def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory not found at {TEST_DIR}")
        return

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = HybridSTFT_LIRA(input_channels=1, output_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    results = []
    pt_files = glob.glob(os.path.join(TEST_DIR, "*.pt"))
    print(f"Found {len(pt_files)} test files.")

    for fpath in tqdm(pt_files, desc="Evaluating"):
        try:
            # Parse filename: rec_noise_snrdb.pt
            basename = os.path.basename(fpath)
            parts = basename.replace(".pt", "").split("_")
            if len(parts) >= 3:
                rec_id = parts[0]
                noise_type = parts[1]
                snr_str = parts[2].replace("db", "").replace("p", ".").replace("n", "-")
                try:
                    snr_level = float(snr_str)
                except:
                    snr_level = 0.0
            else:
                continue

            pkg = torch.load(fpath, map_location=DEVICE)
            inputs = pkg["inputs"]   # List of tensors (2F, T)
            targets = pkg["targets"] # List of tensors (2F, T)

            # Normalize to list
            if isinstance(inputs, torch.Tensor):
                input_list = [inputs[i] for i in range(inputs.shape[0])]
                target_list = [targets[i] for i in range(targets.shape[0])]
            else:
                input_list = inputs
                target_list = targets

            file_metrics = {"snri": [], "rmse": [], "psnr": []}

            for i in range(len(input_list)):
                inp_tensor = input_list[i].to(DEVICE) # (2F, T)
                tgt_tensor = target_list[i].to(DEVICE) # (2F, T)

                # Inference
                with torch.no_grad():
                    # Add batch and channel dims: (1, 1, 2F, T)
                    x = inp_tensor.unsqueeze(0).unsqueeze(0).float()
                    pred = model(x)
                    
                    # Interpolate if needed
                    if pred.shape[-1] != inp_tensor.shape[-1]:
                         pred = F.interpolate(pred, size=inp_tensor.shape, mode='bilinear', align_corners=False)
                    
                    pred_tensor = pred.squeeze() # (2F, T)

                # Convert to Time Domain
                noisy_sig = istft_from_tensor(inp_tensor)
                clean_sig = istft_from_tensor(tgt_tensor)
                denoised_sig = istft_from_tensor(pred_tensor)

                # Calculate Metrics
                snri, rmse, psnr = calculate_metrics(clean_sig, noisy_sig, denoised_sig)
                
                file_metrics["snri"].append(snri)
                file_metrics["rmse"].append(rmse)
                file_metrics["psnr"].append(psnr)

            # Average metrics for this file (Record + Noise + SNR)
            if file_metrics["snri"]:
                avg_snri = np.mean(file_metrics["snri"])
                avg_rmse = np.mean(file_metrics["rmse"])
                avg_psnr = np.mean(file_metrics["psnr"])

                results.append({
                    "Record": rec_id,
                    "NoiseType": noise_type,
                    "SNR_Level": snr_level,
                    "SNRI": avg_snri,
                    "RMSE": avg_rmse,
                    "PSNR": avg_psnr
                })

        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Evaluation complete. Results saved to {OUTPUT_CSV}")
    
    # Print Summary
    print("\n=== Summary by Noise Type ===")
    print(df.groupby("NoiseType")[["SNRI", "RMSE", "PSNR"]].mean())
    print("\n=== Summary by SNR Level ===")
    print(df.groupby("SNR_Level")[["SNRI", "RMSE", "PSNR"]].mean())

if __name__ == "__main__":
    evaluate()
