from ultralytics import YOLO
import sys

def main():
    try:
        print("Bắt đầu huấn luyện mô hình YOLOv12 Instance Segmentation (100 Epochs)...")
        model = YOLO('yolov12s-seg.pt') 
        
        results = model.train(
            data=r'd:\DoAnChuyenNganh\yolov12_dataset\data.yaml',
            epochs=100,           # Số chu kỳ học
            imgsz=640,            # Kích thước ảnh đầu vào
            batch=8,              # Batch size an toàn cho VRAM 6GB
            workers=0,            # Chống tràn RAM và treo máy
            optimizer='SGD',
            lr0=0.01,
            weight_decay=0.0005,
            patience=100,
            device=0,             # Chạy trên GPU
            project='runs',
            name='train_100_epochs'
        )
        print("\nQUÁ TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT THÀNH CÔNG!")
        
    except Exception as e:
        print(f"\n[LỖI TRONG QUÁ TRÌNH HUẤN LUYỆN]: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
