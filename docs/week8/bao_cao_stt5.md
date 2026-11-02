# Báo cáo công việc STT 5 — Tuần 8 (02–08/11/2026)

Xuất lúc: 2026-11-02T13:05:31+07:00

## Phạm vi

Tích hợp frontend với API YOLO và ByteTrack; xử lý ảnh/video, xuất MP4, lưu thông tin từng lần chạy và đo tốc độ trên laptop.
OpenCLIP chưa được tích hợp. Độ tin cậy dự đoán không phải Precision. ID track duy nhất không phải sai số đếm.

## Phần cứng và môi trường

```json
{
  "platform": "Windows-11-10.0.22000-SP0",
  "cpu": "Intel64 Family 6 Model 94 Stepping 3, GenuineIntel",
  "python": "3.12.7",
  "torch": "2.6.0+cu124",
  "cuda_available": true,
  "gpu": "Quadro M1000M",
  "gpu_memory_gb": 4.0
}
```

## Các lần chạy thực tế

| Run | Tệp | Model | Chế độ | Thiết bị | Khớp 16 lớp | Khung | FPS suy luận + tracker | FPS pipeline | Xử lý (s) | Ghi/hoàn tất tệp (s) |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 386a0fcc98a14881 | technical_fixture.mp4 | models/yolov12n-seg.pt | track | cpu | False | 24 | 10.64 | 9.83 | 2.441 | 0.037 |
| 258bb836b2ea4204 | sample_pet.jpg | models/yolov12n-seg.pt | predict | 0 | False | 1 | 6.16 | 4.73 | 0.212 | 0.022 |
| acf6f965d1574da5 | sample_pet.jpg | models/yolov12n-seg.pt | predict | cpu | False | 1 | 12.38 | 7.97 | 0.125 | 0.021 |
| aa1ac8e4ba814ac0 | technical_fixture.mp4 | models/yolov12n-seg.pt | track | cpu | False | 24 | 9.69 | 9.01 | 2.663 | 0.037 |
| a3c65c10b4cd489d | sample_pet.jpg | models/yolov12n-seg.pt | predict | cpu | False | 1 | 12.55 | 8.04 | 0.124 | 0.022 |
| 1b6f9ff387c24f42 | sample_pet.jpg | models/yolov12n-seg.pt | predict | 0 | False | 1 | 3.76 | 3.14 | 0.318 | 0.022 |
| 0366aa1a8f754295 | sample_pet.jpg | models/yolov12n-seg.pt | predict | cpu | False | 1 | 14.49 | 8.78 | 0.114 | 0.021 |
| 96fa9bc32a574a5e | technical_fixture.mp4 | models/yolov12n-seg.pt | track | cpu | False | 24 | 8.25 | 7.61 | 3.155 | 0.036 |
| 23924a0792dc46de | sample_pet.jpg | models/yolov12n-seg.pt | predict | cpu | False | 1 | 13.86 | 8.33 | 0.120 | 0.022 |
| 081a7042e856447f | -PP-5_jpg.rf.ec6b69f6f56399a3363a75bdd51580c4.jpg | models/yolov12n-seg.pt | track | cpu | False | 1 | 10.19 | 6.92 | 0.145 | 0.021 |
| d01ff2679b584b49 | technical_fixture.mp4 | models/yolov12n-seg.pt | track | cpu | False | 24 | 8.84 | 8.11 | 2.961 | 0.037 |
| 6c29191d55e44e5e | sample_pet.jpg | models/yolov12n-seg.pt | predict | cpu | False | 1 | 11.49 | 7.08 | 0.141 | 0.022 |

FPS pipeline gồm đọc khung, suy luận/tracking, vẽ và ghi/xuất; loại thời gian nạp model, warmup, upload và tạm dừng. Thời gian encoder là tổng thời gian ghi/đợi và hoàn tất tệp, không phải CPU time thuần. MP4 không giữ âm thanh.

## Độ chính xác và confusion matrix

Chỉ báo cáo P/R/F1/mAP và confusion matrix sau khi chạy scripts/evaluate.py với checkpoint rác đúng 16 lớp và tập test đã khóa. Không suy ra từ độ tin cậy hiển thị hoặc số liệu COCO.

Chưa tự động kết luận mô hình tốt hơn bài báo; cần cùng dữ liệu, loại chỉ số Box/Mask và điều kiện so sánh.

## Kiểm thử hồi quy

Xem docs/week8/regression_tests.csv và kết quả kiểm thử tự động trong docs/week8/. Các mục kiểm thử thủ công chưa có bằng chứng phải để Chưa chạy.

## Bằng chứng

- local_runs/<run-id>/job.json: cấu hình, SHA256 trọng số, phần cứng, thời gian và trạng thái.
- local_runs/<run-id>/detections.csv: dự đoán theo khung hình.
- local_runs/<run-id>/result.png hoặc result.mp4: đầu ra chú thích.
- /api/benchmark.csv và /api/benchmark.png: bảng và biểu đồ tốc độ thực đo.

Tổng tác vụ: 19; hoàn tất: 12; các lần lỗi/hủy không tính vào kết quả tốc độ hoàn tất.

## Kiểm chứng cài đặt và hồi quy ngày 02/11/2026

- Kiểm thử tự động: 19 ca đạt; bằng chứng `test_results.xml`.
- Kiểm thử thực với trọng số COCO: 6 kịch bản đạt; bằng chứng `smoke_results.json`.
- Đã sửa trạng thái tracker còn sót khi chuyển video sang ảnh và lỗi Windows khóa ảnh preview trong lúc xuất video.
- Đã sửa script đánh giá: đường dẫn tương đối, Precision tách khỏi mAP, xuất riêng Box/Mask và từ chối trọng số sai bộ lớp.
- Dataset lấy từ nhánh Tu: 5.121 train, 1.082 validation, 1.085 test; chưa khẳng định đây là tập test đã được nhóm khóa cho tuần 8.
- Lịch sử có cả các lần thử trước khi sửa lỗi; dùng ID trong `smoke_results.json` để đối chiếu bản kiểm chứng cuối.
- Video kỹ thuật được tạo từ ảnh tĩnh, không đại diện cho cảnh thực địa. Chưa có thống kê ba lượt video thực địa hoặc đánh giá chất lượng tracking/đếm.
- Chưa có checkpoint rác của nhóm, nên chưa hoàn tất confusion matrix hay đánh giá P/R/F1/mAP cho 16 lớp. Chưa tích hợp OpenCLIP.

| Kịch bản thực | Trạng thái | Run ID |
|---|---|---|
| real_image_cpu | PASS | a3c65c10b4cd489d |
| video_yolo_bytetrack_pause_resume_export | PASS | aa1ac8e4ba814ac0 |
| image_after_tracked_video_has_no_stale_tracker | PASS | acf6f965d1574da5 |
| corrupt_image_real_worker | PASS | 7f18fb2cecd74ae1 |
| cancel_real_worker | PASS | a1cb380a92b04569 |
| real_image_cuda | PASS | 258bb836b2ea4204 |

Thông tin phiên bản nguồn: `source_provenance.json`. Kết quả giao diện thủ công xem `regression_tests.csv`.

## Kiểm thử video trên trình duyệt

- Run ID: `386a0fcc98a14881`; xử lý đủ 24 khung, MP4 H.264 320×320, dài 2 giây.
- Trình duyệt đã giải mã và phát video; nút Xuất video đã tải MP4 xuống, mở lại đủ 24 khung.
- FPS suy luận + tracker: 10.64; FPS pipeline: 9.83; xử lý: 2.441 s; ghi/hoàn tất tệp: 0.037 s.
- Đây là video kỹ thuật tạo từ ảnh tĩnh và trọng số COCO, chưa phải kết quả nhận diện rác hay phép thử tracking thực địa.
- Ảnh bằng chứng: `frontend_video_test.jpg`; cấu hình đầy đủ: `browser_video_job.json`.
