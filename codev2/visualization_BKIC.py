import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ===== Cấu hình =====
FS = 500  # Tần số lấy mẫu cố định 500 Hz


def get_ecg_root():
    """Tìm thư mục gốc ECG (parent của 'codev2')."""
    try:
        this_file = os.path.abspath(__file__)
        codev2_dir = os.path.dirname(this_file)
        ecg_root = os.path.dirname(codev2_dir)
        return ecg_root
    except NameError:
        cwd = os.path.abspath(os.getcwd())
        parts = cwd.split(os.sep)
        if "ECG" in parts:
            idx = parts.index("ECG")
            return os.sep.join(parts[: idx + 1])
        return os.path.join(cwd, "ECG")


def visualize_BKIC(input_dir=None, output_dir=None, clear_old=True):
    """
    Vẽ toàn bộ tín hiệu ECG (cột 9) từ các file .txt trong datasetBKIC.

    Parameters
    ----------
    input_dir : str or None
        Thư mục chứa các file .txt (mặc định: <ECG_root>/datasetBKIC)
    output_dir : str or None
        Thư mục lưu ảnh (mặc định: <ECG_root>/datasetBKIC_visualized)
    clear_old : bool
        Nếu True, xóa toàn bộ nội dung cũ trong thư mục output trước khi vẽ lại.
    """
    ecg_root = get_ecg_root()

    if input_dir is None:
        input_dir = os.path.join(ecg_root, "datasetBKIC")
    if output_dir is None:
        output_dir = os.path.join(ecg_root, "datasetBKIC_visualized")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục input: {input_dir}")

    # Xóa kết quả cũ nếu được yêu cầu
    if clear_old and os.path.exists(output_dir):
        print(f"🧹 Đang xóa dữ liệu cũ trong {output_dir} ...")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Lấy danh sách file .txt
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
    if not txt_files:
        print(f"⚠️ Không có file .txt trong {input_dir}")
        return

    print(f"📘 Đang xử lý datasetBKIC ({len(txt_files)} file)")
    print(f"   → Lưu hình vào: {output_dir}")

    for fname in sorted(txt_files):
        path = os.path.join(input_dir, fname)
        try:
            # Đọc toàn bộ dữ liệu
            data = np.loadtxt(path)

            if data.ndim == 1:
                print(f"   ⚠️ {fname}: chỉ có 1 cột → bỏ qua.")
                continue
            if data.shape[1] < 9:
                print(f"   ⚠️ {fname}: ít hơn 9 cột → bỏ qua.")
                continue

            # Lấy cột thứ 9 (index = 8)
            ecg_signal = data[:, 8]

            # Tạo trục thời gian (đơn vị giây)
            n_samples = len(ecg_signal)
            t = np.arange(n_samples) / FS

            # Vẽ toàn bộ tín hiệu
            plt.figure(figsize=(14, 4))
            plt.plot(t, ecg_signal, linewidth=0.8, color="tab:blue")
            plt.title(f"Dataset BKIC - {fname} (Channel 9, fs={FS} Hz)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude (a.u.)")
            plt.grid(True, linestyle="--", alpha=0.6)

            output_path = os.path.join(
                output_dir, f"{os.path.splitext(fname)[0]}_Ch9.png"
            )
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   ✅ {fname} → {output_path} ({n_samples} samples)")

        except Exception as e:
            print(f"   ❌ Lỗi khi xử lý {fname}: {e}")


if __name__ == "__main__":
    visualize_BKIC(clear_old=True)