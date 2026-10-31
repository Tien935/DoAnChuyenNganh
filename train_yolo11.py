import os
import torch
from ultralytics import YOLO

def main():
    print("==================================================")
    print("  QUÁ TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH YOLOV11")
    print("==================================================")

    # 1. Kiểm tra phần cứng (GPU hay CPU)
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        batch_size = 8
        workers = 2
        print(f"[Thiết bị] Đang sử dụng GPU: {device_name}")
    else:
        device = "cpu"
        batch_size = 4
        workers = 0
        print("[Thiết bị] Không phát hiện GPU. Đang dùng CPU.")

    # Tự động xác định đường dẫn data.yaml theo vị trí file script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(base_dir, "dataset", "data.yaml")

    if not os.path.exists(data_yaml_path):
        print(f"Lỗi: Không tìm thấy file '{data_yaml_path}'!")
        return

    # 2. Khởi tạo mô hình YOLOv11 Nano pretrained
    print("\n[Bước 1/3] Tải mô hình YOLOv11 (yolo11n.pt)...")
    model = YOLO("yolo11n.pt")

    # 3. Bắt đầu quá trình Train
    epochs = 50

    print(f"\n[Bước 2/3] Bắt đầu huấn luyện {epochs} epochs trên bộ dữ liệu 16 loại rác...")
    train_results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        workers=workers,
        device=device,
        project=os.path.join(base_dir, "runs", "train_yolo11"),
        name="waste_yolo11",
        save=True,
        plots=True,
        verbose=True
    )

    save_dir = train_results.save_dir
    best_weight_path = os.path.join(save_dir, "weights", "best.pt")
    print("\n==================================================")
    print("           HUẤN LUYỆN THÀNH CÔNG!                 ")
    print("==================================================")
    print(f"File trọng số tốt nhất (Best Model): {best_weight_path}")

    # 4. Đánh giá chi tiết độ chính xác trên tập Test
    print("\n[Bước 3/3] Đang tính toán độ chính xác trên tập Test...")
    best_model = YOLO(best_weight_path)
    metrics = best_model.val(
        data=data_yaml_path,
        split="test",
        imgsz=640,
        device=device,
        workers=workers,
        plots=True
    )

    # 5. In bảng tổng kết
    print("\n" + "=" * 55)
    print("       BẢNG KẾT QUẢ ĐỘ CHÍNH XÁC MÔ HÌNH YOLOV11   ")
    print("=" * 55)
    print(f" * Precision (Độ chính xác)         : {metrics.box.mp * 100:.2f}%")
    print(f" * Recall (Độ nhạy / Độ bao phủ)    : {metrics.box.mr * 100:.2f}%")
    print(f" * mAP@50 (Độ chính xác mAP IoU=0.5): {metrics.box.map50 * 100:.2f}%")
    print(f" * mAP@50-95 (Độ chính xác chuẩn)   : {metrics.box.map * 100:.2f}%")
    print("=" * 55)
    print("\nBiểu đồ kết quả đã lưu tại:")
    print(f"{save_dir}")
    print(" 1. confusion_matrix.png")
    print(" 2. results.png")
    print(" 3. F1_curve.png / PR_curve.png")
    print(" 4. val_batch0_pred.jpg")
    print("=" * 55)

if __name__ == "__main__":
    main()
