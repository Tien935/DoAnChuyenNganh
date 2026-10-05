from ultralytics import YOLO
import sys
import torch

def main():
    try:
        print("Bắt đầu Smoke Test (1 epoch)...")
        # Khởi tạo mô hình YOLO Segmentation. 
        model = YOLO('yolov12s-seg.pt') 
        
        # Bắt đầu quá trình huấn luyện
        results = model.train(
            data=r'd:\DoAnChuyenNganh\yolov12_dataset\data.yaml',
            epochs=1,
            imgsz=640,
            batch=8,
            workers=0,  
            optimizer='SGD',
            lr0=0.01,
            weight_decay=0.0005,
            patience=100,
            device=0,
            project='runs',
            name='smoke_test'
        )
        print("\nSmoke test hoàn tất thành công! Mọi thứ đều ổn định.")
        print(f"VRAM tối đa đã sử dụng: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"\n[LỖI TRONG QUÁ TRÌNH HUẤN LUYỆN]: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
