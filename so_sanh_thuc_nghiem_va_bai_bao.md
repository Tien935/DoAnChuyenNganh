# TỔNG HỢP CÁC ĐIỂM KHÁC BIỆT GIỮA THỰC NGHIỆM VÀ BÀI BÁO GỐC
*Tài liệu tham chiếu: “Real-Time Waste Detection and Classification Using YOLOv12-Based Deep Learning Model”*

Tài liệu này ghi nhận toàn bộ các quyết định thay đổi, tinh chỉnh và khác biệt của dự án thực tế so với bài báo gốc nhằm mục đích tối ưu hóa trên phần cứng cục bộ và sửa các lỗi logic của bài báo.

---

## 1. MỤC TIÊU BÀI TOÁN (TASK)
- **Bài báo gốc:** Giải quyết bài toán **Object Detection** (Nhận dạng bằng khung chữ nhật - Bounding Box).
- **Thực nghiệm của tôi:** Nâng cấp lên bài toán **Instance Segmentation** (Nhận dạng và phân vùng điểm ảnh). 
  - *Lý do:* Tận dụng định dạng Polygon có sẵn trong dataset. Việc này khó huấn luyện hơn nhưng mang lại độ chi tiết cực cao, mô hình có thể khoanh chính xác đường viền (contour) của từng loại rác thay vì chỉ khoanh một khung hình hộp thô cứng.

## 2. DỮ LIỆU & CÁC LỚP RÁC (DATASET & CLASSES)
- **Bài báo gốc:** Ghi nhận 6-7 lớp rác (Plastic, Metal, Paper, Glass, Organic Waste, Medical Waste).
- **Thực nghiệm của tôi:** Sử dụng bộ dữ liệu phân mảnh chi tiết lên tới **16 lớp rác** (Battery, Glass-Bottle, Glove, Mask, Single-Use-Plastic, Syringe, v.v.).
  - *Đánh giá:* Số lượng class nhiều gấp đôi khiến độ khó của bài toán tăng lên đáng kể (nguy cơ nhận nhầm cao hơn do các loại nhựa/giấy khá giống nhau). Tuy nhiên, kết quả thực tế cho thấy mô hình xử lý rất tốt.

## 3. LỖI LOGIC TRONG TỈ LỆ CHIA DATASET
- **Bài báo gốc:** Tác giả chia 75% train + 15% validation + 15% test (Tổng cộng = 105%). Đây là một lỗi sai logic trong quá trình viết báo.
- **Thực nghiệm của tôi:** Khắc phục lỗi này bằng cách chia chuẩn tỷ lệ vàng **70% - 15% - 15%** (Không có rò rỉ dữ liệu).
  - Train: 5.121 ảnh.
  - Validation: 1.082 ảnh.
  - Test: 1.085 ảnh.

## 4. XỬ LÝ LỖI DỮ LIỆU ĐẦU VÀO (DATA PRE-PROCESSING)
- **Bài báo gốc:** Không đề cập đến các rủi ro nhiễu loạn định dạng nhãn.
- **Thực nghiệm của tôi:** Qua quá trình kiểm tra tự động (check_dataset), phát hiện 7.083 nhãn bị lỗi do gán nhãn theo Bounding Box (5 giá trị) trộn lẫn với Polygon (đa giá trị).
  - *Giải pháp:* Viết script tính toán hình học để biến 7.083 Bounding Box này thành Polygon hình chữ nhật. Nhờ đó, bảo toàn được 100% dữ liệu đưa vào huấn luyện Segmentation mà không bị crash.

## 5. MÔ HÌNH VÀ MÔI TRƯỜNG HUẤN LUYỆN
- **Bài báo gốc:** Sử dụng YOLOv12, huấn luyện trên GPU Tesla T4 (16GB VRAM), RAM 16GB.
- **Thực nghiệm của tôi:** 
  - *Phần cứng:* Sử dụng Laptop với RTX 4050 (Chỉ có **6GB VRAM**).
  - *Mô hình:* Chọn biến thể **YOLOv12s-seg** (Bản Small cho Segmentation).
  - *Môi trường:* Thư viện Ultralytics gốc bị lỗi tương thích và tốn RAM. Đã chủ động tải mã nguồn YOLOv12 trực tiếp từ tác giả (repo sunsmarterjie) để cài đặt, giúp giảm thiểu rò rỉ bộ nhớ.
  - *Chống treo máy:* Đã phải vô hiệu hóa tính năng đa luồng bằng tham số `workers=0` để chống hiện tượng nghẽn bộ nhớ ảo (thrashing) do ổ cứng C chỉ còn trống 9GB.

## 6. CẤU HÌNH SIÊU THAM SỐ (HYPERPARAMETERS)
- **Bài báo gốc:** Batch size = 16.
- **Thực nghiệm của tôi:** Giảm Batch size = 8.
  - *Lý do:* Vì VRAM chỉ có 6GB, nếu cố nhồi batch size 16 cho bài toán Segmentation sẽ gây lỗi Out-Of-Memory. Các tham số khác (Epochs=100, SGD, lr0=0.01, weight_decay=0.0005) được giữ nguyên tuyệt đối để đối chiếu.

## 7. KẾT QUẢ ĐẠT ĐƯỢC (RESULTS COMPARISON)
| Metric | Bài báo gốc (Detection) | Thực nghiệm (Segmentation Mask) | Nhận xét |
| :--- | :--- | :--- | :--- |
| **Precision** | ~ 73.0% | **83.5%** | Tốt hơn +10.5% |
| **Recall** | ~ 70.7% | **76.9%** | Tốt hơn +6.2% |
| **F1-Score** | 0.750 | **0.801** | Tốt hơn +0.051 |
| **mAP@0.5** | 78.0% | **80.6%** | Tốt hơn +2.6% |

- Mặc dù phải đối mặt với nhiều khó khăn hơn (16 classes, bài toán Segmentation khó hơn, phần cứng VRAM thấp hơn), mô hình thực nghiệm vẫn đánh bại baseline của bài báo gốc ở tất cả các chỉ số đo lường. Tốc độ suy luận (Inference time) đạt ~19.6ms/ảnh (tương đương 51 FPS), hoàn toàn đáp ứng nhu cầu Real-time.
