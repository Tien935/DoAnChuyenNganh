import os
import torch
from ultralytics import YOLO

def main():
    print("==================================================")
    print("  HUẤN LUYỆN MÔ HÌNH YOLOv11s-seg (SEGMENTATION)")
    print("  Mô hình đối chứng so sánh với YOLOv12s-seg")
    print("==================================================")

    # 1. Kiểm tra GPU
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        batch_size = 8
        workers = 0
        print(f"[Thiết bị] Đang sử dụng GPU: {device_name}")
    else:
        device = "cpu"
        batch_size = 4
        workers = 0
        print("[Thiết bị] Không phát hiện GPU. Đang dùng CPU.")

    # Tự động xác định đường dẫn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(base_dir, "dataset", "data.yaml")

    if not os.path.exists(data_yaml_path):
        print(f"Lỗi: Không tìm thấy file '{data_yaml_path}'!")
        return

    # 2. Load YOLOv11s-seg (Small Segmentation — cùng cỡ với yolov12s-seg)
    print("\n[Bước 1/3] Tải mô hình YOLOv11s-seg (yolo11s-seg.pt)...")
    model = YOLO("yolo11s-seg.pt")   # Tự tải ~22MB nếu chưa có

    # 3. Train — giữ CÙNG điều kiện với YOLOv12s-seg để so sánh công bằng
    epochs = 100

    print(f"\n[Bước 2/3] Bắt đầu huấn luyện {epochs} epochs (Segmentation)...")
    train_results = model.train(
        data=data_yaml_path,
        epochs=epochs,          # Cùng 100 epochs với YOLOv12s-seg
        imgsz=640,              # Cùng kích thước ảnh
        batch=batch_size,       # batch=8 an toàn với VRAM 4GB
        workers=workers,
        optimizer="SGD",        # Cùng optimizer với YOLOv12s-seg
        lr0=0.01,               # Cùng learning rate
        weight_decay=0.0005,    # Cùng weight decay
        patience=100,           # Không early stop (cùng với YOLOv12s-seg)
        device=device,
        project=os.path.join(base_dir, "runs"),
        name="yolo11s_seg_100ep",
        save=True,
        plots=True,
        verbose=True
    )

    save_dir = train_results.save_dir
    best_weight_path = os.path.join(save_dir, "weights", "best.pt")

    print("\n==================================================")
    print("           HUẤN LUYỆN THÀNH CÔNG!")
    print("==================================================")
    print(f"Model tốt nhất: {best_weight_path}")

    # 4. Đánh giá trên tập Test
    print("\n[Bước 3/3] Đánh giá trên tập Test...")
    best_model = YOLO(best_weight_path)
    metrics = best_model.val(
        data=data_yaml_path,
        split="test",
        imgsz=640,
        device=device,
        workers=workers,
        plots=True
    )

    # 5. In bảng kết quả — đủ chỉ số để so sánh với YOLOv12s-seg
    print("\n" + "=" * 60)
    print("   BẢNG KẾT QUẢ YOLOv11s-seg (ĐỐI CHỨNG VỚI YOLOv12s-seg)")
    print("=" * 60)
    print(f" * Box Precision         : {metrics.box.mp * 100:.2f}%")
    print(f" * Box Recall            : {metrics.box.mr * 100:.2f}%")
    print(f" * Box mAP@50            : {metrics.box.map50 * 100:.2f}%")
    print(f" * Box mAP@50-95         : {metrics.box.map * 100:.2f}%")
    if hasattr(metrics, 'seg'):
        print(f" * Mask mAP@50           : {metrics.seg.map50 * 100:.2f}%")
        print(f" * Mask mAP@50-95        : {metrics.seg.map * 100:.2f}%")
    print("=" * 60)
    print("\nBiểu đồ kết quả lưu tại:")
    print(f"  {save_dir}")
    print(" - confusion_matrix.png")
    print(" - results.png (Loss, mAP qua từng epoch)")
    print(" - BoxF1_curve.png / MaskF1_curve.png")
    print(" - PR_curve.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
