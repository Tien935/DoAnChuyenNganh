from ultralytics import YOLO
import sys

def main():
    try:
        print("Đang đánh giá mô hình trên tập TEST...")
        # Load mô hình tốt nhất sau 100 epochs
        model = YOLO(r'd:\DoAnChuyenNganh\runs\train_100_epochs\weights\best.pt')
        
        # Đánh giá trên tập test
        metrics = model.val(data=r'd:\DoAnChuyenNganh\yolov12_dataset\data.yaml', split='test')
        
        # In ra các chỉ số (BBox và Mask)
        print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN TRÊN TẬP TEST ---")
        
        # Box metrics
        box = metrics.box
        print("Bounding Box (Object Detection):")
        print(f"Precision (B): {box.map50_95:.4f} (Lưu ý: đây là map, P/R được YOLO in trên console)")
        
        # Để lấy chi tiết per-class, ta sẽ đọc từ file CSV kết quả hoặc copy từ Terminal sau khi chạy lệnh này.
        
    except Exception as e:
        print(f"\n[LỖI]: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
