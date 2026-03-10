import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==============================
# Configuration
# ==============================

FS = 500
ADC_MAX = 2**23 - 1
ADC_MIN = -2**23


# ==============================
# Utility functions
# ==============================

def get_ecg_root():
    """Locate ECG project root."""
    try:
        this_file = os.path.abspath(__file__)
        code_dir = os.path.dirname(this_file)
        return os.path.dirname(code_dir)
    except NameError:
        cwd = os.getcwd()
        return os.path.join(cwd, "ECG")


def bandpass_filter(signal, fs, low=0.5, high=40, order=4):
    """Standard ECG bandpass filter."""
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def remove_saturation(x):
    """Replace ADC saturation with NaN."""
    x = x.copy()
    x[(x >= ADC_MAX) | (x <= ADC_MIN)] = np.nan
    return x


def detect_active_channel(channels):
    """
    Detect channel with largest variance.
    Channels: list of signals
    """
    vars_ = [np.nanvar(c) for c in channels]
    return int(np.argmax(vars_))


# ==============================
# Calibration (from MATLAB notes)
# ==============================

def calibrate_channels(amp1, amp2, amp3):

    f1 = ((5 / 2) / (2**23)) * amp1

    f2 = (10 * (amp2 - 2**24) / 2) / (2**24 - 1)

    f3 = ((5 - 5 / (2**24)) / (2**24 - 1)) * amp3

    return f1, f2, f3


# ==============================
# Main visualization
# ==============================

def visualize_BKIC3(input_dir=None, output_dir=None, clear_old=True):

    root = get_ecg_root()

    if input_dir is None:
        input_dir = os.path.join(root, "datasetBKIC3")

    if output_dir is None:
        output_dir = os.path.join(root, "datasetBKIC3_visualized")

    if not os.path.isdir(input_dir):
        print("Input directory not found:", input_dir)
        return

    if clear_old and os.path.exists(output_dir):
        print("Cleaning old visualizations...")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]

    if not txt_files:
        print("No txt files found.")
        return

    print(f"Processing {len(txt_files)} files...")

    for fname in sorted(txt_files):

        path = os.path.join(input_dir, fname)

        try:

            data = np.genfromtxt(path)

            if data.ndim == 1 or data.shape[1] < 9:
                print(f"Skip {fname}: invalid format")
                continue

            amp1 = data[:, 6]
            amp2 = data[:, 7]
            amp3 = data[:, 8]

            amp1 = remove_saturation(amp1)
            amp2 = remove_saturation(amp2)
            amp3 = remove_saturation(amp3)

            f1, f2, f3 = calibrate_channels(amp1, amp2, amp3)

            channels = [f1, f2, f3]

            ecg_idx = detect_active_channel(channels)
            ecg_signal = channels[ecg_idx]

            # filter ECG
            ecg_filtered = bandpass_filter(
                np.nan_to_num(ecg_signal),
                FS
            )

            n = len(ecg_signal)
            t = np.arange(n) / FS

            fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

            names = ["Channel 1", "Channel 2", "Channel 3"]

            for i in range(3):

                axs[i].plot(t, channels[i], linewidth=0.8)

                if i == ecg_idx:
                    axs[i].set_title(f"{names[i]}  (Detected ECG)")
                else:
                    axs[i].set_title(names[i])

                axs[i].grid(True, linestyle="--", alpha=0.5)

            axs[3].plot(t, ecg_filtered, color="red", linewidth=1)

            axs[3].set_title("Filtered ECG (0.5–40 Hz)")
            axs[3].set_xlabel("Time (seconds)")
            axs[3].grid(True)

            plt.suptitle(fname)

            plt.tight_layout()

            save_path = os.path.join(
                output_dir,
                os.path.splitext(fname)[0] + ".png"
            )

            plt.savefig(save_path, dpi=150)

            plt.close()

        except Exception as e:
            print("Error processing", fname, e)

    print("Visualization complete.")


if __name__ == "__main__":
    visualize_BKIC3()