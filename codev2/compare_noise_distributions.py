import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, medfilt, istft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ===== Config =====
BKIC_CSV = "BKIC_noise_quantification_v2.csv"
MITDB_TEST_DIR = "preprocessed_mitdb/test"
MITDB_CSV_OUT = "MITDB_noise_quantification.csv"
PLOT_OUT = "noise_distribution_comparison.png"

FS = 360
SEG_LEN = 4096
STFT_NPERSEG = 8
STFT_NOVERLAP = 7
STFT_WINDOW = "boxcar"

# ===== Noise Quantification Logic (V2) =====
class NoiseQuantifierV2:
    def __init__(self, fs=FS):
        self.fs = fs
        self.bw_max_freq = 0.5
        self.ma_min_freq = 25.0

    def remove_baseline(self, x: np.ndarray):
        k = int(round(0.6 * self.fs))
        if k % 2 == 0: k += 1
        baseline = medfilt(x, kernel_size=k)
        x_detrend = x - baseline
        return x_detrend, baseline

    def calculate_bw(self, signal: np.ndarray) -> float:
        _, baseline_estimate = self.remove_baseline(signal)
        return np.sqrt(np.mean(baseline_estimate**2))

    def calculate_ma(self, signal: np.ndarray) -> float:
        signal_no_bw, _ = self.remove_baseline(signal)
        nyq = 0.5 * self.fs
        b, a = butter(4, self.ma_min_freq / nyq, btype='highpass')
        high_freq_component = filtfilt(b, a, signal_no_bw)
        return np.sqrt(np.mean(high_freq_component**2))

    def calculate_em(self, signal: np.ndarray) -> float:
        diff_sig = np.diff(signal)
        normalized_diff = diff_sig * self.fs
        return np.sqrt(np.mean(normalized_diff**2))

    def calculate_total_noise_index(self, signal: np.ndarray):
        nyq = 0.5 * self.fs
        b_bp, a_bp = butter(4, [0.67 / nyq, 35.0 / nyq], btype='bandpass')
        clean_estimate = filtfilt(b_bp, a_bp, signal)
        noise_estimate = signal - clean_estimate
        
        P_s = np.mean(clean_estimate ** 2) + 1e-12
        P_n = np.mean(noise_estimate ** 2) + 1e-12
        snr_db = 10 * np.log10(P_s / P_n)
        return snr_db

# ===== Helper: ISTFT from Tensor =====
def istft_from_tensor(stft_tensor):
    # stft_tensor: (2F, T)
    stft_np = stft_tensor.numpy()
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

# ===== Process MITDB =====
def process_mitdb(test_dir, output_csv):
    quantifier = NoiseQuantifierV2(FS)
    results = []
    
    pt_files = glob.glob(os.path.join(test_dir, "*.pt"))
    print(f"Found {len(pt_files)} test files in {test_dir}")

    for fpath in tqdm(pt_files, desc="Processing MITDB"):
        try:
            # Filename format: record_noise_snrdb.pt (e.g., 100_bw_0p0db.pt)
            basename = os.path.basename(fpath)
            parts = basename.replace(".pt", "").split("_")
            # Handle potential extra underscores in record name if any, but standard is rec_noise_snr
            # Assuming standard format: rec_noise_snrdb
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

            pkg = torch.load(fpath)
            inputs = pkg["inputs"] # List of tensors or Tensor
            
            # Normalize to list of tensors
            if isinstance(inputs, torch.Tensor):
                # (N, 2F, T)
                input_list = [inputs[i] for i in range(inputs.shape[0])]
            else:
                input_list = inputs

            for i, stft_tensor in enumerate(input_list):
                # Convert to Time Domain
                sig = istft_from_tensor(stft_tensor)
                
                # NORMALIZE Z-SCORE (Same as BKIC V2)
                if np.std(sig) > 1e-6:
                    sig = (sig - np.mean(sig)) / np.std(sig)
                else:
                    sig = sig - np.mean(sig)
                
                # Calculate Metrics
                bw = quantifier.calculate_bw(sig)
                ma = quantifier.calculate_ma(sig)
                em = quantifier.calculate_em(sig)
                snr_est = quantifier.calculate_total_noise_index(sig)
                
                results.append({
                    "record_id": rec_id,
                    "noise_type": noise_type,
                    "snr_level": snr_level,
                    "BW_RMS": bw,
                    "MA_RMS": ma,
                    "EM_RMS_deriv": em,
                    "SNR_estimate_dB": snr_est,
                    "Dataset": "MITDB"
                })
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved MITDB results to {output_csv}")
    return df

# ===== Main Visualization =====
def main():
    # 1. Load BKIC
    if not os.path.exists(BKIC_CSV):
        print(f"Error: {BKIC_CSV} not found. Run estimate_noise_components_v2.py first.")
        return
    
    df_bkic = pd.read_csv(BKIC_CSV)
    df_bkic["Dataset"] = "BKIC"
    # Ensure columns match
    df_bkic = df_bkic.rename(columns={
        "BW_RMS": "BW_RMS", 
        "MA_RMS": "MA_RMS", 
        "EM_RMS_deriv": "EM_RMS_deriv", 
        "SNR_estimate_dB": "SNR_estimate_dB"
    })
    # Add dummy columns for compatibility
    df_bkic["noise_type"] = "Real"
    df_bkic["snr_level"] = np.nan

    # 2. Process/Load MITDB
    # FORCE RE-PROCESS to apply normalization
    if os.path.exists(MITDB_CSV_OUT):
        print("Removing old MITDB CSV to re-process with normalization...")
        os.remove(MITDB_CSV_OUT)
        
    if not os.path.exists(MITDB_TEST_DIR):
            print(f"Error: {MITDB_TEST_DIR} not found.")
            return
    df_mit = process_mitdb(MITDB_TEST_DIR, MITDB_CSV_OUT)

    # 3. Combine
    # Select common columns
    cols = ["Dataset", "BW_RMS", "MA_RMS", "EM_RMS_deriv", "SNR_estimate_dB"]
    df_combined = pd.concat([df_bkic[cols], df_mit[cols]], ignore_index=True)
    
    # Convert to numeric
    for c in cols[1:]:
        df_combined[c] = pd.to_numeric(df_combined[c], errors='coerce')
    
    df_combined = df_combined.dropna()

    # 4. Plot
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3)

    # Histograms (Use stat="density" to handle sample size imbalance)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data=df_combined, x="SNR_estimate_dB", hue="Dataset", kde=True, ax=ax1, common_norm=False, stat="density")
    ax1.set_title("SNR Estimate Distribution (Density)")

    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=df_combined, x="BW_RMS", hue="Dataset", kde=True, ax=ax2, common_norm=False, log_scale=True, stat="density")
    ax2.set_title("BW RMS Distribution (Log Scale, Density)")

    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(data=df_combined, x="MA_RMS", hue="Dataset", kde=True, ax=ax3, common_norm=False, log_scale=True, stat="density")
    ax3.set_title("MA RMS Distribution (Log Scale, Density)")

    ax4 = fig.add_subplot(gs[1, 0])
    sns.histplot(data=df_combined, x="EM_RMS_deriv", hue="Dataset", kde=True, ax=ax4, common_norm=False, log_scale=True, stat="density")
    ax4.set_title("EM RMS Distribution (Log Scale, Density)")

    # PCA
    ax5 = fig.add_subplot(gs[1, 1:])
    features = ["BW_RMS", "MA_RMS", "EM_RMS_deriv"]
    x = df_combined[features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    df_pca = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    df_pca["Dataset"] = df_combined["Dataset"].values
    
    # Plot MITDB first (alpha=0.3), then BKIC (alpha=1.0) on top
    df_pca_mit = df_pca[df_pca["Dataset"] == "MITDB"]
    df_pca_bkic = df_pca[df_pca["Dataset"] == "BKIC"]
    
    sns.scatterplot(data=df_pca_mit, x="PC1", y="PC2", label="MITDB", alpha=0.3, ax=ax5, color='orange')
    sns.scatterplot(data=df_pca_bkic, x="PC1", y="PC2", label="BKIC", alpha=1.0, ax=ax5, color='blue', s=100, marker='X')
    
    ax5.set_title(f"PCA of Noise Metrics (Explained Var: {pca.explained_variance_ratio_.sum():.2f})")
    
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)
    print(f"✅ Comparison plot saved to {PLOT_OUT}")

if __name__ == "__main__":
    main()
