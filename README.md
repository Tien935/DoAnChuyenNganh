# Nhận diện và phân loại rác bằng YOLOv12

Ứng dụng hỗ trợ nhận diện rác và phân vùng từng vật thể (instance segmentation) bằng YOLOv12. Giao diện web được xây dựng bằng Gradio, kết hợp FastAPI cho một số chức năng backend. Kết quả gồm lớp dự đoán, vùng mask và hướng dẫn phân loại rác; dự án cũng có các script để kiểm tra dữ liệu, huấn luyện và đánh giá mô hình.

## Tính năng

- Phân vùng vật thể rác trong ảnh bằng YOLOv12 segmentation.
- Giao diện Gradio hỗ trợ ảnh, video và webcam; ứng dụng khởi chạy backend webcam riêng.
- Phân loại rác theo nhóm và hiển thị gợi ý xử lý.
- Tùy chọn OpenCLIP hỗ trợ nhận diện zero-shot cho một số trường hợp được cấu hình trong ứng dụng.
- Script kiểm tra/chuyển đổi nhãn, huấn luyện và đánh giá mô hình.

## Cấu trúc dự án

```text
.
├── app.py                         # Ứng dụng Gradio/FastAPI
├── app_1.py                       # Bản ứng dụng phụ
├── yolov12/                       # Mã nguồn YOLOv12
├── yolov12s-seg.pt                # Trọng số khởi tạo segmentation
├── runs/train_100_epochs/         # Kết quả huấn luyện (nếu có)
├── yolov12_dataset/               # Dataset và data.yaml (nếu có)
├── sample_images/                 # Ảnh mẫu
├── scripts/
│   ├── check_dataset.py           # Kiểm tra dataset
│   ├── fix_mixed_labels.py        # Chuẩn hóa nhãn segmentation
│   ├── train.py                   # Huấn luyện
│   ├── train_smoke.py             # Chạy thử huấn luyện
│   └── evaluate.py                # Đánh giá trên tập test
└── YOLOv12_paper_evaluation_tools/ # Công cụ đánh giá/đối chiếu bổ sung
```

Một số thư mục dữ liệu, trọng số và kết quả chạy có thể không được đưa vào bản clone hoặc kho Git. Hãy kiểm tra chúng trước khi chạy các lệnh huấn luyện/đánh giá.

## Yêu cầu

- Windows 10/11 và Python 3.10 hoặc phiên bản tương thích với mã nguồn YOLOv12 trong thư mục `yolov12`.
- GPU NVIDIA và CUDA được khuyến nghị để huấn luyện, nhưng ứng dụng có thể chạy suy luận trên CPU (chậm hơn).
- Git nếu cần tải mã nguồn YOLOv12.

## Cài đặt

Mở PowerShell tại thư mục dự án:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

Cài PyTorch phù hợp với máy của bạn. Ví dụ dưới đây dùng CUDA 12.4; nếu không có GPU NVIDIA, hãy chọn lệnh cài đặt CPU từ trang chính thức của PyTorch.

```powershell
.\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Cài các thư viện ứng dụng và YOLOv12:

```powershell
.\venv\Scripts\python.exe -m pip install opencv-python gradio imageio imageio-ffmpeg numpy pandas matplotlib pyyaml pillow fastapi uvicorn open-clip-torch "numpy<2"
.\venv\Scripts\python.exe -m pip install huggingface_hub
.\venv\Scripts\python.exe -m pip install -e .\yolov12
```

Nếu cài đặt hoặc chạy gặp lỗi tương thích, hãy đối chiếu phiên bản Python/PyTorch với yêu cầu của mã nguồn YOLOv12 đang dùng.

## Trọng số mô hình

Ứng dụng ưu tiên trọng số đã huấn luyện tại `runs/train_100_epochs/weights/best.pt`. Nếu file này không tồn tại, `app.py` dùng `yolov12s-seg.pt` ở thư mục gốc. Vì vậy, cần có ít nhất một trong hai file trọng số trước khi chạy ứng dụng.

Nếu cần tải trọng số khởi tạo YOLOv12s segmentation:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12s-seg.pt" `
  -OutFile "yolov12s-seg.pt"
```

## Chạy ứng dụng

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe app.py
```

Ứng dụng mở giao diện tại `http://127.0.0.1:7860`. Có thể đổi cổng bằng biến môi trường `PORT`:

```powershell
$env:PORT = "7861"
.\venv\Scripts\python.exe app.py
```

## Kiểm tra dữ liệu và huấn luyện

Các script giả định dataset nằm tại `yolov12_dataset/` và có file `data.yaml`. Một số đường dẫn trong script hiện được ghi cố định theo máy phát triển; nếu chạy ở vị trí khác, hãy cập nhật đường dẫn dữ liệu/trọng số trong script tương ứng.

Kiểm tra dataset:

```powershell
.\venv\Scripts\python.exe scripts\check_dataset.py
```

Chuẩn hóa nhãn hỗn hợp sang định dạng segmentation (chỉ chạy khi dữ liệu cần chuyển đổi):

```powershell
.\venv\Scripts\python.exe scripts\fix_mixed_labels.py
```

Huấn luyện cấu hình 100 epoch:

```powershell
.\venv\Scripts\python.exe scripts\train.py
```

Đánh giá mô hình trên tập test:

```powershell
.\venv\Scripts\python.exe scripts\evaluate.py
```

Kết quả huấn luyện dự kiến được lưu trong `runs/train_100_epochs/`. Đánh giá cần có dataset, file cấu hình và trọng số huấn luyện phù hợp.

## Ghi chú

- Không sử dụng ảnh webcam/video làm bằng chứng đo hiệu năng nếu chưa cố định quy trình và cấu hình đo.
- Trọng số, dataset và thư mục `runs/` có thể chiếm nhiều dung lượng; không phải bản clone nào cũng bao gồm các tài nguyên này.
- Hướng dẫn phân loại trong ứng dụng mang tính hỗ trợ. Hãy tuân thủ hướng dẫn thu gom rác tại địa phương, đặc biệt với pin, vật sắc nhọn và rác y tế.
