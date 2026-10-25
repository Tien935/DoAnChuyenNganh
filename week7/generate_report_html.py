import base64
from pathlib import Path

def img_to_b64(path):
    if not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

fig1_b64 = img_to_b64(r"d:\DACN\week7\figures\fig1_similar_classes_cm.png")
fig2_b64 = img_to_b64(r"d:\DACN\week7\figures\fig2_confidence_margin.png")
fig3_b64 = img_to_b64(r"d:\DACN\week7\figures\fig3_prompt_stability.png")
fig4_b64 = img_to_b64(r"d:\DACN\week7\figures\fig4_ood_rejection_curves.png")

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo cáo Kỹ thuật Tuần 7 - Lê Đại Cương</title>
<style>
    body {{
        font-family: 'Times New Roman', Times, serif;
        line-height: 1.5;
        color: #222;
        max-width: 900px;
        margin: 40px auto;
        padding: 40px 50px;
        background: #fff;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-radius: 8px;
    }}
    @media print {{
        body {{
            box-shadow: none;
            margin: 0;
            padding: 0;
            max-width: 100%;
        }}
    }}
    .header {{
        text-align: center;
        margin-bottom: 25px;
    }}
    .header h1 {{
        color: #1b365d;
        font-size: 24px;
        margin: 0 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .header .sub {{
        color: #555;
        font-size: 14px;
        font-weight: bold;
        line-height: 1.4;
    }}
    .meta-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0 30px 0;
        font-size: 14px;
    }}
    .meta-table td {{
        padding: 8px 12px;
        border: 1px solid #dcdcdc;
    }}
    .meta-table tr:nth-child(even) {{
        background-color: #f9fafc;
    }}
    .meta-table td.label {{
        font-weight: bold;
        color: #1b365d;
        width: 25%;
    }}
    .status-badge {{
        color: #2e7d32;
        font-weight: bold;
    }}
    h2 {{
        color: #1b365d;
        font-size: 18px;
        border-bottom: 2px solid #1b365d;
        padding-bottom: 4px;
        margin-top: 30px;
        margin-bottom: 12px;
    }}
    h3 {{
        color: #2980b9;
        font-size: 16px;
        margin-top: 20px;
        margin-bottom: 8px;
    }}
    p, li {{
        font-size: 15px;
        text-align: justify;
    }}
    ul {{
        margin-top: 4px;
        padding-left: 24px;
    }}
    li {{
        margin-bottom: 6px;
    }}
    .data-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
    }}
    .data-table th {{
        background-color: #1b365d;
        color: #fff;
        padding: 9px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #1b365d;
    }}
    .data-table td {{
        padding: 8px 10px;
        border: 1px solid #ddd;
    }}
    .data-table tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
    .data-table td.highlight {{
        background-color: #d4edda;
        font-weight: bold;
    }}
    .data-table td.best {{
        background-color: #fff3cd;
        font-weight: bold;
    }}
    .figure-container {{
        text-align: center;
        margin: 25px 0;
    }}
    .figure-container img {{
        max-width: 90%;
        height: auto;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }}
    .figure-caption {{
        font-size: 13.5px;
        font-style: italic;
        color: #555;
        margin-top: 6px;
    }}
    .signatures {{
        display: flex;
        justify-content: space-between;
        margin-top: 40px;
        text-align: center;
        font-size: 15px;
    }}
    .signature-box {{
        width: 45%;
    }}
    .signature-space {{
        height: 70px;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>BÁO CÁO KỸ THUẬT TIẾN ĐỘ TUẦN 7</h1>
    <div class="sub">
        ĐỀ TÀI: ỨNG DỤNG THỊ GIÁC MÁY TÍNH TRONG PHÁT HIỆN VÀ PHÂN LOẠI RÁC THẢI<br>
        CHUYÊN TRÁCH: MÔ HÌNH THỊ GIÁC - NGÔN NGỮ (VLM)
    </div>
</div>

<table class="meta-table">
    <tr>
        <td class="label">Sinh viên thực hiện:</td>
        <td><strong>Lê Đại Cương</strong> — MSSV: <strong>2023604315</strong> (TV4: Mô hình thị giác-ngôn ngữ)</td>
    </tr>
    <tr>
        <td class="label">Nhiệm vụ tuần 7:</td>
        <td>Kiểm thử VLM: lớp gần giống, prompt nhạy cảm, ảnh không thuộc 6 lớp.</td>
    </tr>
    <tr>
        <td class="label">Mục tiêu nghiệm thu:</td>
        <td>Báo cáo độ ổn định prompt và unknown threshold.</td>
    </tr>
    <tr>
        <td class="label">Đánh giá tiến độ:</td>
        <td><span class="status-badge">Tiến độ: 100% — Trạng thái: Hoàn thành đúng hạn</span></td>
    </tr>
</table>

<h2>1. MỤC TIÊU VÀ PHẠM VI KIỂM THỬ TUẦN 7</h2>
<p>
    Trong giai đoạn Tuần 7 (Kiểm thử hệ thống), sau khi hoàn thành tích hợp sơ bộ tại Tuần 6, thành viên phụ trách VLM tiến hành kiểm thử chuyên sâu nhằm đánh giá độ tin cậy và các ranh giới suy luận của mô hình OpenCLIP (ViT-B-32-quickgelu). Trọng tâm kiểm thử gồm 3 trụ cột kỹ thuật:
</p>
<ul>
    <li><strong>Trụ cột 1 — Lớp gần giống:</strong> Đánh giá khả năng phân biệt của mô hình trên các cặp vật liệu có đặc trưng thị giác tương đồng (plastic vs glass, metal vs paper, cardboard vs paper, trash hỗn hợp).</li>
    <li><strong>Trụ cột 2 — Độ ổn định Prompt:</strong> Kiểm tra mức độ nhạy cảm và độ biến động dự đoán khi thay đổi cấu trúc câu nhắc từ ngắn đơn giản, mô tả chi tiết, prompt ensemble đến prompt nhiễu bối cảnh.</li>
    <li><strong>Trụ cột 3 — Out-of-Distribution (OOD):</strong> Kiểm thử khả năng từ chối đối với ảnh ngoài tập dữ liệu 6 lớp (đồ vật thường nhật, nhiễu tổng hợp, rác ngoài phạm vi) và xác lập dải ngưỡng unknown threshold tối ưu.</li>
</ul>

<h2>2. KẾT QUẢ KIỂM THỬ CÁC LỚP DỄ NHẦM LẪN (SIMILAR CLASSES)</h2>
<p>
    Thực nghiệm được triển khai trên toàn bộ <strong>595 ảnh rác thải có gán nhãn thực tế</strong> thuộc 6 nhóm. Kết quả chung đạt độ chính xác Accuracy <strong>80.67%</strong> (Weighted F1: 0.8132).
</p>

<h3>2.1. Ma trận nhầm lẫn và chỉ số phân loại chi tiết</h3>
<table class="data-table">
    <thead>
        <tr>
            <th>Lớp (Ground Truth)</th>
            <th>cardboard</th>
            <th>glass</th>
            <th>metal</th>
            <th>paper</th>
            <th>plastic</th>
            <th>trash</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>cardboard (24)</strong></td>
            <td class="highlight" style="text-align:right;">20</td>
            <td style="text-align:right;">0</td>
            <td style="text-align:right;">0</td>
            <td style="text-align:right;">4</td>
            <td style="text-align:right;">0</td>
            <td style="text-align:right;">0</td>
        </tr>
        <tr>
            <td><strong>glass (400)</strong></td>
            <td style="text-align:right;">0</td>
            <td class="highlight" style="text-align:right;">372</td>
            <td style="text-align:right;">14</td>
            <td style="text-align:right;">0</td>
            <td style="text-align:right;">14</td>
            <td style="text-align:right;">0</td>
        </tr>
        <tr>
            <td><strong>metal (22)</strong></td>
            <td style="text-align:right;">1</td>
            <td style="text-align:right;">0</td>
            <td class="highlight" style="text-align:right;">15</td>
            <td style="text-align:right;">0</td>
            <td style="text-align:right;">2</td>
            <td style="text-align:right;">4</td>
        </tr>
        <tr>
            <td><strong>paper (57)</strong></td>
            <td style="text-align:right;">7</td>
            <td style="text-align:right;">0</td>
            <td style="text-align:right;">1</td>
            <td class="highlight" style="text-align:right;">43</td>
            <td style="text-align:right;">0</td>
            <td style="text-align:right;">6</td>
        </tr>
        <tr>
            <td><strong>plastic (33)</strong></td>
            <td style="text-align:right;">1</td>
            <td style="text-align:right;">10</td>
            <td style="text-align:right;">13</td>
            <td style="text-align:right;">0</td>
            <td class="highlight" style="text-align:right;">9</td>
            <td style="text-align:right;">0</td>
        </tr>
        <tr>
            <td><strong>trash (59)</strong></td>
            <td style="text-align:right;">4</td>
            <td style="text-align:right;">4</td>
            <td style="text-align:right;">4</td>
            <td style="text-align:right;">9</td>
            <td style="text-align:right;">17</td>
            <td class="highlight" style="text-align:right;">21</td>
        </tr>
    </tbody>
</table>

<div class="figure-container">
    <img src="data:image/png;base64,{fig1_b64}" alt="Hình 1: Ma trận nhầm lẫn giữa 6 nhóm rác thải">
    <div class="figure-caption">Hình 1: Heatmap ma trận nhầm lẫn giữa 6 nhóm rác thải trên mô hình OpenCLIP</div>
</div>

<h3>2.2. Phân tích các cặp vật liệu gây nhầm lẫn chính</h3>
<ul>
    <li><strong>Nhựa vs Thủy tinh (Plastic -> Glass):</strong> 10/33 mẫu chai nhựa (30.30%) bị nhận diện nhầm sang thủy tinh. Do chai nhựa PET trong suốt phản quang và khúc xạ ánh sáng rất giống chai thủy tinh trong ảnh chụp tĩnh.</li>
    <li><strong>Carton vs Giấy (Cardboard &lt;-&gt; Paper):</strong> 4/24 mẫu bìa carton (16.67%) bị nhầm sang giấy, và 7/57 mẫu giấy in (12.28%) bị nhầm sang carton. Bản chất carton là sản phẩm tái chế từ giấy nhiều lớp nên phân bố embedding ngữ nghĩa rất sát nhau.</li>
    <li><strong>Rác tổng hợp (Trash):</strong> 17/59 mẫu (28.81%) bị nhầm sang nhựa và 9/59 mẫu (15.25%) nhầm sang giấy do rác sinh hoạt hỗn hợp chứa nhiều túi nilon và vỏ hộp, khiến mô hình thiên về vật liệu có diện tích bề mặt chiếm ưu thế.</li>
</ul>

<div class="figure-container">
    <img src="data:image/png;base64,{fig2_b64}" alt="Hình 2: Phân bố Margin">
    <div class="figure-caption">Hình 2: Phân bố biên độ chênh lệch độ tin cậy (Confidence Margin Δ = Top-1 - Top-2)</div>
</div>
<p>
    Phân tích biên độ Margin cho thấy: các mẫu dự đoán sai hầu hết có biên độ chênh lệch rất nhỏ (Δ &lt; 0.15). Đây là cơ sở then chốt để áp dụng quy tắc phân xử phối hợp với YOLO ở Tuần 8.
</p>

<h2>3. KẾT QUẢ KIỂM THỬ ĐỘ NHẠY CẢM VÀ ỔN ĐỊNH CỦA PROMPT</h2>
<p>
    Thực nghiệm tiến hành trên <strong>166 ảnh mẫu chuẩn hóa</strong> (30 ảnh/lớp) nhằm so sánh 5 chiến lược soạn thảo prompt:
</p>
<ul>
    <li><strong>Chiến lược 1 (P1 - Single Short):</strong> Độ chính xác chỉ đạt 56.02%, độ nhất quán so với Ensemble chỉ đạt 74.70% (42 mẫu bị đảo nhãn). Từ khóa đơn lẻ quá mơ hồ và dễ bị nhiễu ngữ cảnh.</li>
    <li><strong>Chiến lược 2 (P2 - Template Standard):</strong> Độ chính xác tăng lên 63.86%, độ nhất quán đạt 82.53% (29 mẫu bị đảo nhãn). Bổ sung từ khóa 'waste' định hướng vật thể vào ngữ cảnh phế liệu rõ rệt.</li>
    <li><strong>Chiến lược 3 (P3 - Descriptive 3):</strong> Độ chính xác đạt 63.25%, điểm tin cậy trung bình tăng lên 0.7264.</li>
    <li><strong>Chiến lược 4 (P4 - Ensemble 5 Prompts):</strong> Độ chính xác 63.25%, phân bố điểm tin cậy ổn định nhất (Std = 0.2170), đóng vai trò baseline tham chiếu chuẩn mực.</li>
    <li><strong>Chiến lược 5 (P5 - Contextual Distracted):</strong> Gây ra tới 70/166 trường hợp đảo nhãn (tỷ lệ nhất quán rớt xuống 57.83%). Từ khóa về nền nhà, bàn gỗ làm lệch không gian vector nghiêm trọng.</li>
</ul>

<div class="figure-container">
    <img src="data:image/png;base64,{fig3_b64}" alt="Hình 3: So sánh các chiến lược prompt">
    <div class="figure-caption">Hình 3: So sánh độ chính xác và tính nhất quán giữa 5 chiến lược Prompt</div>
</div>

<h2>4. KIỂM THỬ ẢNH NGOÀI TẬP (OOD) VÀ XÁC ĐỊNH NGƯỠNG UNKNOWN</h2>
<p>
    Tập thử nghiệm gồm 166 ảnh rác thực tế (In-Distribution - ID) và 59 ảnh ngoài tập rác (Out-of-Distribution - OOD) gồm các thiết bị điện tử, đồ gia dụng, cảnh sinh hoạt và nhiễu tổng hợp. Tiến hành quét dải ngưỡng τ ∈ [0.30, 0.80]:
</p>

<table class="data-table">
    <thead>
        <tr>
            <th>Ngưỡng τ</th>
            <th style="text-align:right;">True Rejection (OOD %)</th>
            <th style="text-align:right;">False Rejection (Mất ID %)</th>
            <th style="text-align:right;">Retained Accuracy (%)</th>
            <th style="text-align:right;">Balanced F1 (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0.30</td>
            <td style="text-align:right;">40.68%</td>
            <td style="text-align:right;">2.41%</td>
            <td style="text-align:right;">63.58%</td>
            <td style="text-align:right;">57.42%</td>
        </tr>
        <tr>
            <td>0.40</td>
            <td style="text-align:right;">72.88%</td>
            <td style="text-align:right;">13.86%</td>
            <td style="text-align:right;">67.83%</td>
            <td style="text-align:right;">78.96%</td>
        </tr>
        <tr class="best">
            <td><strong>0.45 (Tối ưu)</strong></td>
            <td style="text-align:right;"><strong>81.36%</strong></td>
            <td style="text-align:right;"><strong>21.08%</strong></td>
            <td style="text-align:right;"><strong>70.99%</strong></td>
            <td style="text-align:right;"><strong>80.12%</strong></td>
        </tr>
        <tr class="best">
            <td><strong>0.50 (An toàn)</strong></td>
            <td style="text-align:right;"><strong>89.83%</strong></td>
            <td style="text-align:right;"><strong>29.52%</strong></td>
            <td style="text-align:right;"><strong>74.36%</strong></td>
            <td style="text-align:right;"><strong>78.99%</strong></td>
        </tr>
        <tr>
            <td>0.60</td>
            <td style="text-align:right;">94.92%</td>
            <td style="text-align:right;">38.55%</td>
            <td style="text-align:right;">77.45%</td>
            <td style="text-align:right;">74.60%</td>
        </tr>
        <tr>
            <td>0.70</td>
            <td style="text-align:right;">98.31%</td>
            <td style="text-align:right;">52.41%</td>
            <td style="text-align:right;">86.08%</td>
            <td style="text-align:right;">64.13%</td>
        </tr>
    </tbody>
</table>

<div class="figure-container">
    <img src="data:image/png;base64,{fig4_b64}" alt="Hình 4: Đường cong OOD Rejection">
    <div class="figure-caption">Hình 4: Đường cong đánh đổi giữa tỷ lệ loại bỏ đúng OOD và bỏ sót rác thật theo ngưỡng τ</div>
</div>

<p><strong>Khuyến nghị chốt cấu hình ngưỡng cho hệ thống tích hợp:</strong></p>
<ul>
    <li><strong>Ngưỡng nhạy tối ưu (τ = 0.45):</strong> Điểm cân bằng F1 cao nhất (80.12%), giữ lại gần 80% mẫu rác thật trong khi chặn được hơn 81% vật thể lạ.</li>
    <li><strong>Ngưỡng an toàn thực tế (τ = 0.50):</strong> Chặn đứng 89.83% vật thể OOD, nâng độ chính xác của các mẫu giữ lại lên 74.36%. Đây là mức khuyến nghị cho phiên bản ứng dụng laptop.</li>
</ul>

<h2>5. KHÓ KHĂN PHÁT SINH VÀ ĐỀ XUẤT GIẢI PHÁP CHO TUẦN 8</h2>
<ul>
    <li><strong>1. Nhầm lẫn Plastic - Glass:</strong> Độ tương đồng ngữ nghĩa và thị giác giữa chai nhựa trong suốt và thủy tinh quá cao, dẫn tới 30.3% chai nhựa bị nhầm. Đề xuất: Phối hợp với YOLOv8 để bổ sung thông tin hình học bounding box (nhựa thường có nắp vặn hoặc vết méo bẹp).</li>
    <li><strong>2. Đánh đổi ngưỡng Unknown:</strong> Khi tăng ngưỡng τ lên trên 0.60, tỷ lệ bỏ sót rác thật (FRR) tăng vọt lên gần 40%, gây lãng phí dữ liệu phát hiện từ YOLO. Khẳng định ngưỡng 0.45 - 0.50 là vùng tối ưu không nên vượt quá.</li>
    <li><strong>3. Chuẩn hóa Prompt cố định:</strong> Các prompt có chứa thông tin bối cảnh (nền gỗ, sàn nhà) làm giảm mạnh độ ổn định. Do đó, toàn bộ pipeline tích hợp chỉ sử dụng bộ prompt thuần vật liệu Ensemble P4.</li>
</ul>

<h2>6. KẾT LUẬN VÀ BÀN GIAO</h2>
<p>
    Thành viên Lê Đại Cương đã hoàn thành 100% nhiệm vụ Tuần 7 đúng tiến độ, bàn giao đầy đủ mã nguồn kiểm thử và các kết luận thực nghiệm phục vụ việc đánh giá mô hình kết hợp ở Tuần 8.
</p>

<div class="signatures">
    <div class="signature-box">
        <strong>XÁC NHẬN CỦA TRƯỞNG NHÓM</strong><br>
        <em>(Ký và ghi rõ họ tên)</em>
        <div class="signature-space"></div>
        <strong>Đỗ Đức Tiền</strong>
    </div>
    <div class="signature-box">
        <em>Hà Nội, ngày 25 tháng 10 năm 2026</em><br>
        <strong>NGƯỜI LẬP BÁO CÁO</strong><br>
        <em>(Ký và ghi rõ họ tên)</em>
        <div class="signature-space"></div>
        <strong>Lê Đại Cương</strong>
    </div>
</div>

</body>
</html>
"""

out_html_dacn = Path(r"d:\DACN\week7\Bao_cao_tuan_7_Le_Dai_Cuong.html")
out_html_down = Path(r"c:\Users\ledai\Searches\Downloads\Bao_cao_tuan_7_Le_Dai_Cuong.html")

out_html_dacn.write_text(html_content, encoding="utf-8")
print("Saved HTML to:", out_html_dacn)
out_html_down.write_text(html_content, encoding="utf-8")
print("Saved HTML to Downloads:", out_html_down)
