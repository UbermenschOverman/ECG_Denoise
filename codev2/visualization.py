import os
import wfdb
import matplotlib.pyplot as plt

def get_ecg_root():
    """Trả về thư mục gốc ECG (parent của 'codev2')."""
    try:
        this_file = os.path.abspath(__file__)
        codev2_dir = os.path.dirname(this_file)
        ecg_root = os.path.dirname(codev2_dir)
        return ecg_root
    except NameError:
        # fallback khi chạy trong Colab interactive
        cwd = os.path.abspath(os.getcwd())
        parts = cwd.split(os.sep)
        if "ECG" in parts:
            idx = parts.index("ECG")
            ecg_root = os.sep.join(parts[: idx + 1])
            return ecg_root
        return os.path.join(cwd, "ECG")

def visualize_dataset(dataset_dir, output_dir, num_samples=2000, show_annots=False):
    """Vẽ và lưu tín hiệu ECG của từng record trong một bộ dữ liệu."""
    if not os.path.isdir(dataset_dir):
        print(f"⚠️ Không tìm thấy thư mục: {dataset_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    records = [os.path.splitext(f)[0] for f in os.listdir(dataset_dir) if f.lower().endswith(".hea")]
    records = sorted(list(set(records)))

    if not records:
        print(f"⚠️ Không có file .hea trong {dataset_dir}")
        return

    print(f"\n📘 Đang xử lý dataset: {os.path.basename(dataset_dir)} ({len(records)} records)")
    print(f"   → Lưu hình vào: {output_dir}")

    for record_name in records:
        record_path = os.path.join(dataset_dir, record_name)
        try:
            record = wfdb.rdrecord(record_path)
            fs = getattr(record, "fs", 360)
            signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
            n_samples = min(num_samples, len(signal))
            time = [i / fs for i in range(n_samples)]

            plt.figure(figsize=(12, 4))
            plt.plot(time, signal[:n_samples], linewidth=1)
            plt.title(f"{os.path.basename(dataset_dir)} - Record: {record_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (mV)")
            plt.grid(True)

            # Annotation (nếu có và được bật)
            if show_annots:
                try:
                    ann = wfdb.rdann(record_path, "atr")
                    ann_samples = [s for s in ann.sample if s < n_samples]
                    if ann_samples:
                        yvals = [signal[s] for s in ann_samples]
                        plt.scatter([s / fs for s in ann_samples], yvals, color="red", marker="x", s=20)
                except Exception:
                    pass

            output_path = os.path.join(output_dir, f"{record_name}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   ✅ {record_name}.png")
        except Exception as e:
            print(f"   ❌ Lỗi với {record_name}: {e}")

def visualize_all_datasets(dataset_names=None, num_samples=2000, show_annots=False):
    """Tự động xử lý tất cả các dataset ECG con trong thư mục ECG."""
    ecg_root = get_ecg_root()
    print(f"📂 ECG root: {ecg_root}")

    # Nếu không truyền danh sách, tự tìm các thư mục có file .hea
    if dataset_names is None:
        subdirs = [d for d in os.listdir(ecg_root) if os.path.isdir(os.path.join(ecg_root, d))]
        dataset_names = [
            d for d in subdirs
            if any(f.endswith(".hea") for f in os.listdir(os.path.join(ecg_root, d)))
        ]

    if not dataset_names:
        print("❌ Không tìm thấy dataset nào (không có thư mục chứa .hea trong ECG/)")
        return

    print(f"🔍 Sẽ xử lý các dataset: {dataset_names}")

    for name in dataset_names:
        dataset_dir = os.path.join(ecg_root, name)
        output_dir = os.path.join(ecg_root, f"{name}_visualized")
        visualize_dataset(dataset_dir, output_dir, num_samples=num_samples, show_annots=show_annots)


if __name__ == "__main__":
    # Xử lý toàn bộ dataset trong thư mục ECG/
    visualize_all_datasets(show_annots=False, num_samples=2000)
