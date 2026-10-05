# DỰ ÁN TỐT NGHIỆP: NHẬN DẠNG VÀ PHÂN LOẠI RÁC THẢI VỚI YOLOv12 INSTANCE SEGMENTATION

## 1. Mục tiêu Dự án
Dự án nhằm mục đích xây dựng một hệ thống học sâu (Deep Learning) sử dụng kiến trúc YOLOv12 mới nhất để không chỉ **nhận diện (Detection)** mà còn **phân vùng điểm ảnh (Instance Segmentation)** chính xác 16 loại rác thải khác nhau. 

Dự án này là phiên bản nâng cấp và tối ưu hóa từ bài báo gốc *"Real-Time Waste Detection and Classification Using YOLOv12-Based Deep Learning Model"*.

## 2. Thông tin Dataset
- **Tổng số ảnh:** 7.288 ảnh.
- **Phân chia (Split):** 70% Train (5.121 ảnh), 15% Validation (1.082 ảnh), 15% Test (1.085 ảnh).
- **Số lớp (Classes):** 16 lớp (Battery, Glass-Bottle, Glove, Mask, Medicine, Newspaper, PET Bottle, Paper, Single-Use-Plastic, Single-layer Plastic, Squeeze Tube, syringe, v.v.).
- **Định dạng nhãn:** Toàn bộ được chuẩn hóa về dạng Polygon (Đa giác) để phục vụ huấn luyện Segmentation.

## 3. Cấu trúc Project (Sau khi thiết lập)
```text
D:\DoAnChuyenNganh\
├── yolov12_dataset/          # Dữ liệu ảnh và nhãn
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml             # File khai báo đường dẫn và classes
├── yolov12/                  # Mã nguồn gốc YOLOv12 (từ Github)
├── scripts/                  # Các đoạn code thực thi tự động
│   ├── check_dataset.py      # Kiểm tra và rà soát lỗi nhãn
│   ├── fix_mixed_labels.py   # Chuyển BBox thành Polygon
│   ├── train_smoke.py        # Test nhanh 1 epoch
│   ├── train.py              # Huấn luyện chính thức 100 epochs
│   └── evaluate.py           # Chấm điểm mô hình trên tập Test
├── runs/
│   └── train_100_epochs/     # Nơi lưu file trọng số best.pt và biểu đồ
├── venv/                     # Môi trường ảo Python
├── yolov12s-seg.pt           # File trọng số học trước (Pre-trained)
├── project_changes_log.txt   # Nhật ký quyết định kỹ thuật
├── so_sanh_thuc_nghiem_va_bai_bao.md # Báo cáo so sánh với bài báo
└── README.md                 # Hướng dẫn này
```

## 4. Hướng dẫn Cài đặt Môi trường (Installation)
*Mở Terminal (PowerShell) và chạy lần lượt các lệnh sau:*

**Bước 4.1: Tạo môi trường ảo và cập nhật pip**
```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

**Bước 4.2: Cài đặt PyTorch hỗ trợ GPU (Card rời NVIDIA)**
```powershell
.\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Bước 4.3: Cài đặt mã nguồn lõi YOLOv12 và thư viện phụ trợ**
*(Vì thư viện ultralytics mặc định chưa ổn định với YOLOv12, ta phải cài từ mã nguồn github gốc)*
```powershell
.\venv\Scripts\python.exe -m pip install opencv-python pandas matplotlib pyyaml
cd yolov12
..\venv\Scripts\python.exe -m pip install -e .
cd ..
.\venv\Scripts\python.exe -m pip install huggingface_hub "numpy<2.0.0"
```

**Bước 4.4: Tải trọng số Pre-trained**
```powershell
Invoke-WebRequest -Uri "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12s-seg.pt" -OutFile "yolov12s-seg.pt"
```

## 5. Tiền xử lý dữ liệu (Data Pre-processing)
Chạy script để chuyển đổi các nhãn cũ (Bounding Box 5 thông số) thành nhãn Segmentation (Polygon 9 thông số) để tránh lỗi huấn luyện:
```powershell
$env:PYTHONIOENCODING="utf-8"
.\venv\Scripts\python.exe scripts\fix_mixed_labels.py
```

## 6. Huấn luyện Mô hình (Training)
Mô hình được cấu hình để không bị treo máy trên Windows (`workers=0`) và tránh tràn RAM đồ họa 6GB (`batch=8`). Chạy lệnh sau để huấn luyện 100 chu kỳ:
```powershell
$env:PYTHONIOENCODING="utf-8"
.\venv\Scripts\python.exe scripts\train.py
```
*(Kết quả sẽ được lưu vào thư mục `runs/train_100_epochs/weights/best.pt`)*

## 7. Đánh giá Mô hình (Evaluation)
Để kiểm tra độ chính xác mAP, Precision, Recall trên tập TEST (1.085 ảnh chưa từng nhìn thấy):
```powershell
$env:PYTHONIOENCODING="utf-8"
.\venv\Scripts\python.exe scripts\evaluate.py
```

## 8. Kết quả Thực nghiệm (Segmentation Mask)
- **Precision:** 83.5%
- **Recall:** 76.9%
- **F1-Score:** 0.801
- **mAP@0.5:** 80.6%
- **Tốc độ (Inference Time):** ~19.6ms / ảnh (~51 FPS trên RTX 4050).
*(Đạt độ chính xác cao hơn bài báo gốc dù sử dụng số lượng phân lớp nhiều gấp đôi).*

## 9. Khắc phục sự cố thường gặp (Troubleshooting)
- **Lỗi `module 'numpy' has no attribute 'trapz'`:** Do dùng numpy bản 2.x. Cách sửa: `pip install "numpy<2.0.0"`.
- **Lỗi treo cứng máy / tràn Pagefile khi đang train:** Do cơ chế đa luồng của Windows ép tải thư viện PyTorch nhiều lần. Cách sửa: Bắt buộc thêm tham số `workers=0` vào hàm `model.train()`.
- **Lỗi `No such file or directory: 'yolov12s-seg.pt'`:** Thư viện không tự tải được file trọng số do YOLOv12 quá mới. Cách sửa: Chủ động download file `.pt` từ github thả vào thư mục gốc.
