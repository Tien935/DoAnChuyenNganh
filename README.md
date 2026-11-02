# DoAnChuyenNganh

## Chạy frontend tích hợp trên laptop

Xem [HUONG_DAN_CHAY_LOCAL.md](HUONG_DAN_CHAY_LOCAL.md). Mở bằng `start_local.cmd`, sau đó truy cập http://127.0.0.1:8000.

Bản tích hợp YOLOv12 + ByteTrack được phát triển trong nhánh `frontend`. Trọng số COCO hiện có chỉ dùng kiểm thử kỹ thuật; cần bổ sung `models/best.pt` đã huấn luyện của nhóm để nhận diện và đánh giá đúng 16 lớp rác. OpenCLIP chưa tích hợp.
