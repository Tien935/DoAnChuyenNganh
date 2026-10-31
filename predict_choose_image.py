import os
import glob
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image

def find_latest_best_pt():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    best_files = glob.glob(os.path.join(base_dir, "runs", "**", "best.pt"), recursive=True)
    if best_files:
        best_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return best_files[0]
    return os.path.join(base_dir, "yolo11n.pt")

def main():
    model_name = find_latest_best_pt()
    print("==================================================")
    print("  CHỌN ẢNH ĐỂ NHẬN DIỆN BẰNG YOLOV11")
    print(f"  Mô hình sử dụng: {model_name}")
    print("==================================================")

    # 1. Mở cửa sổ Windows để chọn ảnh
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("Đang mở cửa sổ chọn file ảnh từ máy tính...")
    image_path = filedialog.askopenfilename(
        title="Chọn ảnh rác cần nhận diện",
        filetypes=[
            ("Tất cả định dạng ảnh", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )

    if not image_path:
        print("Bạn chưa chọn ảnh nào. Đã hủy chương trình.")
        return

    print(f"\nĐã chọn ảnh: {image_path}")

    # 2. Tải mô hình
    print(f"Đang nạp mô hình: {model_name}...")
    model = YOLO(model_name)

    # 3. Chạy nhận diện
    print("Đang thực hiện nhận diện...")
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True
    )

    result = results[0]

    # 4. In thông tin kết quả
    print("\n---------------- KẾT QUẢ ----------------")
    if len(result.boxes) == 0:
        print("Không phát hiện đối tượng nào trong ảnh.")
    else:
        print(f"Tìm thấy {len(result.boxes)} đối tượng:")
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            print(f"  + {name:<22}: {conf * 100:.2f}%")

    print("-----------------------------------------")
    print(f"Ảnh kết quả đã vẽ khung được lưu tại:")
    print(f"{result.save_dir}")

    # 5. Mở hiển thị trực tiếp ảnh kết quả lên màn hình
    try:
        saved_img_path = os.path.join(result.save_dir, os.path.basename(image_path))
        if os.path.exists(saved_img_path):
            img_result = Image.open(saved_img_path)
            img_result.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
