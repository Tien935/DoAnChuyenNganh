import os
import sys
import glob
import torch
from ultralytics import YOLO

def find_latest_best_pt():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_pattern = os.path.join(base_dir, "runs", "**", "best.pt")
    best_files = glob.glob(search_pattern, recursive=True)
    if best_files:
        best_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return best_files[0]
    return os.path.join(base_dir, "yolo11n.pt")

def main():
    print("==================================================")
    print("   ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH YOLOV11 CHI TIẾT  ")
    print("==================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(base_dir, "dataset", "data.yaml")

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = find_latest_best_pt()

    print(f"Mô hình đang đánh giá: {model_path}")

    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file model '{model_path}'!")
        return

    model = YOLO(model_path)
    device = 0 if torch.cuda.is_available() else "cpu"

    print("\nĐang tính toán các chỉ số trên tập kiểm thử (Test Set)...")
    metrics = model.val(
        data=data_yaml_path,
        split="test",
        device=device,
        workers=2,
        plots=True
    )

    print("\n" + "=" * 65)
    print("              BẢNG CHỈ SỐ TỔNG QUAN TOÀN BỘ MÔ HÌNH")
    print("=" * 65)
    print(f"{'Precision (P - Độ chính xác)':<35} | {metrics.box.mp * 100:>14.2f}%")
    print(f"{'Recall (R - Độ nhạy / Bao phủ)':<35} | {metrics.box.mr * 100:>14.2f}%")
    print(f"{'mAP@50 (Độ chính xác chuẩn IoU=0.5)':<35} | {metrics.box.map50 * 100:>14.2f}%")
    print(f"{'mAP@50-95 (Độ chính xác nâng cao)':<35} | {metrics.box.map * 100:>14.2f}%")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("              ĐỘ CHÍNH XÁC CHI TIẾT TỪNG LOẠI RÁC")
    print("=" * 65)
    print(f"{'Tên loại rác (Class)':<25} | {'mAP@50 (%)':<15} | {'mAP@50-95 (%)':<15}")
    print("-" * 65)

    class_names = model.names
    maps_per_class = metrics.box.maps

    for idx, name in class_names.items():
        if idx < len(maps_per_class):
            map50_95_val = maps_per_class[idx] * 100
            map50_val = metrics.box.all_ap[idx, 0] * 100 if hasattr(metrics.box, 'all_ap') else 0.0
            print(f"{name:<25} | {map50_val:>14.2f}% | {map50_95_val:>14.2f}%")

    print("=" * 65)

if __name__ == "__main__":
    main()
