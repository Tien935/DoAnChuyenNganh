from ultralytics import YOLO

def main():
    print("Đang nạp mô hình best.pt để trích xuất điểm số...")
    # Khởi tạo mô hình
    model = YOLO(r'd:\DoAnChuyenNganh\runs\train_100_epochs\weights\best.pt')
    
    # Chạy hàm val() trên tập test để lấy ra object kết quả
    # Thêm workers=0 và batch=1 để tiết kiệm RAM tuyệt đối
    metrics = model.val(data=r'd:\DoAnChuyenNganh\yolov12_dataset\data.yaml', split='test', workers=0, batch=1)
    
    # Lấy thông số Bounding Box (Tương đương Object Detection của bài báo)
    # box = metrics.box
    # box_mp = box.mp      # Mean Precision
    # box_mr = box.mr      # Mean Recall
    # box_map50 = box.map50# mAP @ 0.5
    # # Công thức tính F1-Score: 2 * (P * R) / (P + R)
    # box_f1 = (2 * box_mp * box_mr) / (box_mp + box_mr + 1e-6)
    
    # Lấy thông số Segmentation Mask
    seg = metrics.seg
    seg_mp = seg.mp
    seg_mr = seg.mr
    seg_map50 = seg.map50
    seg_f1 = (2 * seg_mp * seg_mr) / (seg_mp + seg_mr + 1e-6)
    
    # In ra kết quả tổng hợp gọn gàng
    print("\n" + "="*50)
    print(" KẾT QUẢ ĐÁNH GIÁ (TRÊN TẬP TEST - 1.085 ẢNH)")
    print("="*50)
    
    print("\n1. KẾT QUẢ OBJECT DETECTION (Bounding Box):")
    print(f"   - Precision (P): {box_mp:.3f} ")
    print(f"   - Recall (R):    {box_mr:.3f} ")
    print(f"   - mAP@0.5:       {box_map50:.3f} ")
    print(f"   - F1-Score:      {box_f1:.3f} ")
    
    print("\n2. KẾT QUẢ INSTANCE SEGMENTATION (Mask):")
    print(f"   - Precision (P): {seg_mp:.3f} ")
    print(f"   - Recall (R):    {seg_mr:.3f} ")
    print(f"   - mAP@0.5:       {seg_map50:.3f} ")
    print(f"   - F1-Score:      {seg_f1:.3f} ")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
