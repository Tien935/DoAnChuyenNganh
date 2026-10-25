# BÁO CÁO KỸ THUẬT TUẦN 7 — MÔ HÌNH THỊ GIÁC - NGÔN NGỮ (VLM)
**Thành viên phụ trách:** Lê Đại Cương (Mã SV: 2023604315)  
**Vai trò:** Mô hình thị giác-ngôn ngữ (TV4)  
**Dự án:** Ứng dụng thị giác máy tính trong phát hiện và phân loại rác thải  
**Thời gian:** Tuần 7 (Giai đoạn Kiểm thử hệ thống)

---

## 1. MỤC TIÊU VÀ NHIỆM VỤ TUẦN 7
Theo kế hoạch dự án 10 tuần:
* **Công việc được giao:** Kiểm thử VLM: lớp gần giống, prompt nhạy cảm, ảnh không thuộc 6 lớp.
* **Mục tiêu cần đạt:** Báo cáo độ ổn định prompt và unknown threshold.

---

## 2. KIẾN TRÚC VÀ CÁC MODULE KIỂM THỬ TRIỂN KHAI
Thư mục `d:\DACN\week7` được tổ chức khoa học với các module kiểm thử chuyên biệt:

```text
week7/
├── vlm_eval_core.py             # Lớp cốt lõi Evaluator hỗ trợ tùy biến prompt, tính Margin Top1-Top2
├── test_similar_classes.py      # Module kiểm thử 6 lớp có nhãn & phân tích cặp lớp dễ nhầm lẫn
├── test_prompt_sensitivity.py   # Module kiểm thử 5 chiến lược Prompt & tính tỷ lệ nhất quán (Consistency)
├── test_ood_unknown.py          # Module kiểm thử Out-of-Distribution & quét dải ngưỡng Unknown Threshold
├── run_week7_eval.py            # Script điều phối thực thi toàn bộ kịch bản kiểm thử
├── generate_week7_figures.py    # Script xuất 4 biểu đồ phân tích 300 DPI
├── generate_report_docx.py      # Script xuất file Word báo cáo (.docx)
├── generate_report_html.py      # Script xuất file Web báo cáo (.html)
├── data/
│   └── ood_images/              # 59 ảnh OOD (ảnh tổng hợp, đồ vật thường nhật, rác ngoài 6 nhóm)
├── results/                     # 7 bảng số liệu thực nghiệm định dạng .csv
└── figures/                     # 4 biểu đồ phân tích trực quan định dạng .png
```

---

## 3. KẾT QUẢ THỰC NGHIỆM CHI TIẾT

### 3.1. Kiểm thử các lớp gần giống (Similar Classes Test)
* **Quy mô mẫu:** 595 ảnh có gán nhãn thực tế thuộc 6 nhóm từ `images/`.
* **Hiệu năng chung:** Độ chính xác Accuracy đạt **80.67%**, Weighted F1-score đạt **0.8132**.
* **Lớp vượt trội nhất:** `glass` (Thủy tinh) đạt Precision 0.9637, Recall 0.9300, F1 0.9466 (372/400 mẫu đúng).
* **Phân tích các cặp gây nhầm lẫn chính:**
  * **Plastic vs Glass:** 10/33 mẫu chai nhựa trong suốt (30.30%) bị nhầm sang thủy tinh do đặc trưng khúc xạ và phản xạ ánh sáng tương tự nhau trong ảnh chụp 2D.
  * **Cardboard vs Paper:** 4/24 mẫu carton (16.67%) nhầm sang giấy; 7/57 mẫu giấy in dày (12.28%) nhầm sang carton do cùng nguồn gốc vật liệu cellulose.
  * **Trash hỗn hợp:** 17/59 mẫu (28.81%) nhầm sang nhựa và 9/59 mẫu (15.25%) nhầm sang giấy do rác hỗn hợp chứa các mảng nilon hoặc bao bì chiếm diện tích bề mặt lớn.
* **Phân tích Margin $\Delta = s_{\text{top1}} - s_{\text{top2}}$:** Đa số các mẫu dự đoán sai có biên độ chênh lệch rất hẹp ($\Delta < 0.15$), mở ra khả năng phân xử phối hợp với YOLO ở Tuần 8.

---

### 3.2. Kiểm thử độ nhạy cảm và độ ổn định của Prompt (Prompt Sensitivity)
Thực nghiệm so sánh 5 chiến lược trên 166 ảnh chuẩn hóa (30 ảnh/lớp):

| Chiến lược Prompt | Độ chính xác (Accuracy %) | Confidence TB | Độ lệch chuẩn (Std) | Tỷ lệ nhất quán vs Ensemble | Số mẫu bị đảo nhãn |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P1 - Single Short** (Tên lớp đơn lẻ) | 56.02% | 0.6798 | 0.2076 | 74.70% | 42 |
| **P2 - Template Standard** (a photo of {c} waste) | 63.86% | 0.6744 | 0.2257 | 82.53% | 29 |
| **P3 - Descriptive 3** (Mô tả cơ lý 3 câu) | 63.25% | 0.7264 | 0.2149 | 75.30% | 41 |
| **P4 - Ensemble 5 Prompts** (Baseline Tuần 5-6) | **63.25%** | **0.6695** | **0.2170** | **100.00%** | **0** |
| **P5 - Contextual Distracted** (Nhiễu bối cảnh) | 67.47% | 0.7600 | 0.1955 | 57.83% | 70 |

* **Kết luận then chốt:** Single Short prompt quá mơ hồ, dẫn tới tỷ lệ đảo nhãn cao (25.3%). Bộ Prompt Ensemble (P4) giữ cho không gian vector ổn định nhất, phân tán độ tin cậy đều và triệt tiêu phương sai từ ngữ. Các prompt có bối cảnh phức tạp (P5) gây xáo trộn nhãn nghiêm trọng (70 mẫu bị đổi nhãn).

---

### 3.3. Kiểm thử ảnh Out-of-Distribution (OOD) và Quét ngưỡng Unknown
Thực nghiệm trên 166 ảnh In-Distribution (ID) và 59 ảnh OOD:

| Ngưỡng $\tau$ | True Rejection (OOD đúng %) | False Rejection (Bỏ sót rác thật %) | Tỷ lệ giữ lại ID (%) | Độ chính xác ID giữ lại (%) | Balanced F1 (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.30 | 40.68% | 2.41% | 97.59% | 63.58% | 57.42% |
| 0.35 | 54.24% | 8.43% | 91.57% | 67.11% | 68.12% |
| 0.40 | 72.88% | 13.86% | 86.14% | 67.83% | 78.96% |
| **0.45** | **81.36%** | **21.08%** | **78.92%** | **70.99%** | **80.12% (Tối ưu F1)** |
| **0.50** | **89.83%** | **29.52%** | **70.48%** | **74.36%** | **78.99% (An toàn)** |
| 0.55 | 91.53% | 34.34% | 65.66% | 76.15% | 76.47% |
| 0.60 | 94.92% | 38.55% | 61.45% | 77.45% | 74.60% |
| 0.70 | 98.31% | 52.41% | 47.59% | 86.08% | 64.13% |

* **Khuyến nghị ngưỡng hoạt động:**
  * **$\tau = 0.45$:** Điểm cân bằng F1 cao nhất (80.12%), chặn hơn 81% vật thể OOD trong khi bảo toàn 79% ảnh rác thật.
  * **$\tau = 0.50$:** Mức an toàn thực tế trên laptop, loại bỏ gần 90% (89.83%) ảnh ngoài phạm vi và nâng độ chính xác của tập dữ liệu giữ lại lên 74.36%.

---

## 4. KHÓ KHĂN VÀ BÀI HỌC KINH NGHIỆM
1. **Ranh giới vật liệu Plastic - Glass:** Tính chất trong suốt của nhựa và thủy tinh là thách thức cố hữu của mô hình thị giác thuần túy. Đề xuất kết hợp bounding box aspect ratio từ YOLO ở Tuần 8.
2. **Hiện tượng Over-rejection:** Không nên đặt $\tau > 0.55$ vì tỷ lệ loại bỏ nhầm rác thật sẽ vượt quá 35%, làm giảm độ nhạy phát hiện của hệ thống chung.

---

## 5. BÀN GIAO CHO CÁC THÀNH VIÊN
* **Bàn giao TV1 (Đỗ Đức Tiền):** Khuyến nghị cấu hình `threshold = 0.50` cho module VLM và bộ prompt Ensemble cố định P4.
* **Bàn giao TV3 (Nguyễn Xuân Tùng):** Dữ liệu phân tích ma trận nhầm lẫn để chuẩn bị kịch bản thực nghiệm so sánh mô hình độc lập vs kết hợp ở Tuần 8.
