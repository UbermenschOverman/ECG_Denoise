import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Configuration
# ==============================

FS = 500
ADC_MAX = 2**23 - 1
ADC_MIN = -2**23

# số giây hiển thị đầu tiên (giúp nhìn ECG dễ hơn)
DISPLAY_SECONDS = 10


# ==============================
# Utility
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


def detect_saturation_ratio(x):
    """Compute how many samples are saturated."""
    sat = np.sum((x >= ADC_MAX) | (x <= ADC_MIN))
    return sat / len(x)


# ==============================
# Visualization
# ==============================

def visualize_raw(input_dir=None, output_dir=None, clear_old=True):

    root = get_ecg_root()

    if input_dir is None:
        input_dir = os.path.join(root, "datasetBKIC3")

    if output_dir is None:
        output_dir = os.path.join(root, "datasetBKIC3_raw_visualized")

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

            channels = [amp1, amp2, amp3]

            # =====================
            # time axis
            # =====================

            n = len(amp1)
            t = np.arange(n) / FS

            # zoom first N seconds
            n_display = int(DISPLAY_SECONDS * FS)

            # =====================
            # saturation stats
            # =====================

            sat1 = detect_saturation_ratio(amp1)
            sat2 = detect_saturation_ratio(amp2)
            sat3 = detect_saturation_ratio(amp3)

            # =====================
            # plotting
            # =====================

            fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

            names = ["Amp1 (Column 7)", "Amp2 (Column 8)", "Amp3 (Column 9)"]
            sats = [sat1, sat2, sat3]

            for i in range(3):

                axs[i].plot(
                    t[:n_display],
                    channels[i][:n_display],
                    linewidth=0.8
                )

                axs[i].set_title(
                    f"{names[i]}  |  saturation={sats[i]*100:.2f}%"
                )

                axs[i].set_ylabel("ADC value")
                axs[i].grid(True, linestyle="--", alpha=0.5)

            axs[2].set_xlabel("Time (seconds)")

            plt.suptitle(fname)
            plt.tight_layout()

            save_path = os.path.join(
                output_dir,
                os.path.splitext(fname)[0] + "_raw.png"
            )

            plt.savefig(save_path, dpi=150)
            plt.close()

        except Exception as e:
            print("Error processing", fname, e)

    print("Raw visualization complete.")


# ==============================

if __name__ == "__main__":
    visualize_raw()