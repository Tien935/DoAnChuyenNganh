# Chạy EcoVision trên laptop và làm báo cáo STT 5

Bản tích hợp nằm trong nhánh `frontend`. Giao diện được phát triển từ `origin/frontend`; dataset được đọc từ `origin/Tu` và mã tham khảo từ `Tien-new`. Các nhánh nguồn không bị sửa. Chưa commit hoặc push GitHub.

## 1. Mở ứng dụng

Nhấp đúp `F:\DoAnChuyenNganh\start_local.cmd`, giữ cửa sổ đó mở. Hoặc chạy trong PowerShell:

```powershell
Set-Location F:\DoAnChuyenNganh
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Mở http://127.0.0.1:8000. Nhấn Ctrl+C trong cửa sổ chạy để dừng. Nếu báo cổng đang được dùng, kiểm tra cửa sổ ứng dụng đã mở; không khởi động thêm bản thứ hai.

## 2. Thử ngay

1. Vào **Nhận diện**, giữ CPU, kích thước 320; chọn một ảnh test mẫu hoặc ảnh JPG/PNG của bạn.
2. Nhấn **Chạy**, chờ **Hoàn tất**; tải ảnh, CSV dự đoán hoặc thông tin lần chạy JSON.
3. Vào **Video**; chọn **Video kỹ thuật 2 giây** hoặc video thực của bạn. Video mẫu được tạo từ ảnh tĩnh, chỉ thử luồng kỹ thuật.
4. Chọn YOLO hoặc YOLO + ByteTrack; nhấn Chạy. Có thể tạm dừng, tiếp tục hoặc hủy tác vụ.
5. Khi hoàn tất, xem video MP4 trong trang hoặc nhấn Xuất video. Video đầu ra không có âm thanh.
6. Mở **Lịch sử** để lấy lại kết quả. Dữ liệu được lưu trên ổ đĩa, vẫn còn sau khi khởi động lại.

Webcam hiện hỗ trợ mở camera và chụp một khung hình để nhận diện, chưa suy luận liên tục. Trình duyệt sẽ hỏi quyền camera; tự cấp hoặc từ chối theo nhu cầu.

## 3. Mô hình hiện có và file cần bổ sung

Hiện có `models/yolov12n-seg.pt`, tải từ bản phát hành chính thức của tác giả YOLOv12. Đây là trọng số COCO **80 lớp**, chưa huấn luyện cho 16 lớp rác. Nó có thể trả nhãn không liên quan hoặc nhận sai rác. Chỉ dùng kiểm tra API, giao diện, xuất video và đo kỹ thuật; chưa dùng chứng minh độ chính xác nhận diện rác.

Nhận checkpoint đã huấn luyện từ nhóm, chép vào:

```text
F:\DoAnChuyenNganh\models\best.pt
```

Trong giao diện, nhấn **Tải lại mô hình**, chọn `models/best.pt` và chạy lại. API đọc tên lớp thật từ checkpoint; không đổi nhãn COCO thành nhãn rác. Cần khớp đủ tên và thứ tự lớp trong `config/waste.yaml`. Tên lớp khớp chưa chứng minh nguồn hay chất lượng checkpoint; vẫn cần xác nhận với thành viên huấn luyện.

OpenCLIP/VLM chưa tích hợp: kết quả YOLO và YOLO+ByteTrack không đại diện cho hệ thống đầy đủ YOLO+ByteTrack+VLM.

## 4. Đầu ra phục vụ báo cáo tuần 8

- **Bảng tốc độ CSV:** tất cả lần chạy và cấu hình, gồm cả lần lỗi/hủy để truy vết.
- **Biểu đồ tốc độ:** chỉ các lần hoàn tất; ghi kèm loại trọng số và cấu hình khi đưa vào báo cáo.
- **Tải báo cáo tuần 8:** báo cáo Markdown từ số liệu thật hiện có.
- `local_runs/<run-id>/`: ảnh/video kết quả, `detections.csv`, `job.json`, tệp nguồn và log encoder.
- `docs/week8/`: hướng dẫn/báo cáo kiểm chứng và checklist hồi quy. Trạng thái chưa chạy vẫn được giữ nguyên.

FPS suy luận + tracker đo thời gian gọi mô hình, gồm tracking nếu bật. FPS pipeline còn gồm đọc khung, vẽ và ghi/xuất; không tính nạp model, warmup, upload hay tạm dừng. Thời gian encoder gồm ghi/đợi và hoàn tất tệp, không phải thời gian CPU thuần. Tốc độ tải tệp từ trình duyệt không được gọi là thời gian tạo video.

`unique_tracks` là số ID track duy nhất qua video; chưa phải số vật thể đi qua vạch, chưa có sai số đếm hay chỉ số đổi ID. Độ tin cậy trên từng vật thể không phải Precision của tập test.

Muốn đo so sánh: dùng cùng video thực, thiết bị, mô hình, kích thước và ngưỡng; chạy warmup rồi ít nhất 3 lượt mỗi cấu hình. Báo cáo trung bình/độ phân tán; tách video mẫu kỹ thuật khỏi video thực địa.

## 5. Confusion matrix, Precision, Recall, F1 và mAP

Sau khi có `best.pt` đúng 16 lớp và nhóm chốt tập test:

```powershell
.\venv\Scripts\python.exe scripts\evaluate.py --weights models\best.pt --device cpu --imgsz 320
```

Có thể dùng `--device 0 --imgsz 640` nếu máy đáp ứng và cấu hình này đã được nhóm thống nhất. Không so sánh các lần chạy khác cấu hình như cùng một phép thử.

Kết quả nằm trong `local_runs/evaluation/waste_test*`: confusion matrix, các đường PR/F1 do YOLO xuất, `summary.json`, `per_class.csv`. Chỉ số Box và Mask được tách riêng. Script từ chối checkpoint có tên/thứ tự lớp sai và nhãn test sai định dạng; không tự sửa tập test.

Một số nhãn polygon có thể là hình chữ nhật chuyển từ bbox. Điều này chỉ tạo mask hình chữ nhật, chưa phải nhãn phân vùng bám đường viền vật thể. Cần mô tả đúng giới hạn khi báo cáo Mask mAP.

## 6. Phần cứng và kiểm chứng cài đặt

Máy này có Quadro M1000M 4 GB, Python 3.12.7, PyTorch 2.6.0+cu124. Đã chạy thử ảnh trên CPU và CUDA; mặc định CPU/320 để bắt đầu nhẹ. Kết quả RTX 4050 trong README thuộc môi trường khác.

Để cài lại các phần còn thiếu trên máy khác, xem `setup_local.ps1` (cần mạng và quyền Git). Mã lõi YOLOv12 dùng commit `64d5ffc6db35a1bde446689ab8450ecd43475e00`. Danh sách gói đã cài được lưu ở `local_runs/installed_packages.txt`.

Kiểm tra lại khi có sửa code:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_local_api.py -q
```

Chạy máy chủ trước, rồi dùng cửa sổ PowerShell khác nếu muốn chạy lại kiểm thử thật. Lệnh này tạo thêm các tác vụ và video mẫu, cần chờ tác vụ giao diện đang chạy kết thúc:

```powershell
.\venv\Scripts\python.exe scripts\smoke_local.py
```

Không commit `venv`, `dataset`, `models`, `yolov12`, `local_runs` vì có dữ liệu lớn hoặc sinh tự động. Chỉ đưa mã và phần báo cáo cần thiết vào PR sau khi bạn rà soát.
