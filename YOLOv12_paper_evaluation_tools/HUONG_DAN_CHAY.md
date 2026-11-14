# Chạy đánh giá để lấy số liệu viết bài báo

## 1. Cài thư viện

Mở Terminal hoặc Command Prompt trong thư mục chứa hai script rồi chạy:

```bash
python -m pip install ultralytics pyyaml pillow
```

Nên dùng đúng môi trường Python đã dùng để huấn luyện mô hình.

## 2. Kiểm tra dữ liệu (không sửa hoặc xóa tệp)

Ví dụ trên Windows:

```bash
python audit_dataset.py --data "D:\DoAnChuyenNganh\yolov12_dataset\data.yaml" --output "audit_results"
```

Các tệp quan trọng được tạo ra:

- `audit_summary.json`: số ảnh, số đối tượng và số lỗi của từng tập.
- `class_distribution.csv`: phân bố 16 lớp trong train/validation/test.
- `audit_errors.json`: danh sách nhãn thiếu hoặc không hợp lệ.
- `cross_split_exact_duplicates.json`: ảnh trùng hoàn toàn giữa các tập.

Script chỉ đọc dữ liệu. Không tự động sửa hoặc xóa ảnh/nhãn.

## 3. Đánh giá `best.pt`

```bash
python evaluate_paper.py ^
  --model "D:\DoAnChuyenNganh\runs\train_100_epochs\weights\best.pt" ^
  --data "D:\DoAnChuyenNganh\yolov12_dataset\data.yaml" ^
  --output "paper_results" ^
  --device 0 ^
  --imgsz 640 ^
  --batch 1 ^
  --workers 0 ^
  --iou 0.7
```

Nếu chạy trong PowerShell, có thể viết toàn bộ lệnh trên một dòng. Nếu máy không có GPU NVIDIA phù hợp, thay `--device 0` bằng `--device cpu` nhưng phải ghi đúng phần cứng này trong bài báo.

Script thực hiện đúng hai bước:

1. Chọn ngưỡng confidence theo Mask F1 cao nhất trên tập validation.
2. Đánh giá tập test với PR curve đầy đủ, không dùng test để chọn ngưỡng.

Các tệp cần gửi lại để viết bài:

- `paper_results/test_metrics.json`
- `paper_results/per_class_metrics.csv`
- Toàn bộ thư mục `paper_results/yolo_runs/test_final`
- `audit_results/audit_summary.json`
- `audit_results/class_distribution.csv`
- `audit_results/cross_split_exact_duplicates.json`

## 4. Chỉ số chính

Chỉ số chính của bài là **Mask mAP@0.5:0.95**. Precision, Recall và F1 được báo cáo tại ngưỡng confidence chọn trên validation. Kết quả bounding box, Mask mAP@0.5, kết quả theo lớp và tốc độ suy luận là các chỉ số bổ trợ.

Không sử dụng lại các con số cũ nếu chúng không khớp với tệp `test_metrics.json` mới tạo.
