import base64
from pathlib import Path

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

fig1_b64 = img_to_b64(r"d:\DACN\week6\figures\fig1_agreement_pie.png")
fig2_b64 = img_to_b64(r"d:\DACN\week6\figures\fig2_agreement_by_class.png")
fig3_b64 = img_to_b64(r"d:\DACN\week6\figures\fig3_alpha_distribution.png")

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo cáo Kỹ thuật Tuần 6 - Lê Đại Cương</title>
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
    .formula-box {{
        background: #fdf2f0;
        border-left: 4px solid #e74c3c;
        padding: 12px 20px;
        margin: 15px 0;
        font-family: Consolas, monospace;
        font-size: 15px;
        font-weight: bold;
        color: #c0392b;
        text-align: center;
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
        padding: 10px;
        font-weight: bold;
        text-align: left;
        border: 1px solid #1b365d;
    }}
    .data-table td {{
        padding: 8px 10px;
        border: 1px solid #ddd;
    }}
    .data-table tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
    .data-table tr.total {{
        font-weight: bold;
        background-color: #eef2f7;
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
    <h1>BÁO CÁO KỸ THUẬT TIẾN ĐỘ TUẦN 6</h1>
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
        <td class="label">Nhiệm vụ tuần 6:</td>
        <td>Đóng gói OpenCLIP inference; chuẩn hóa điểm; đề xuất công thức kết hợp confidence.</td>
    </tr>
    <tr>
        <td class="label">Mục tiêu nghiệm thu:</td>
        <td>VLM module hoàn chỉnh, chuẩn hóa API và quy tắc hợp nhất điểm tin cậy phục vụ pipeline.</td>
    </tr>
    <tr>
        <td class="label">Đánh giá tiến độ:</td>
        <td><span class="status-badge">Tiến độ: 100% — Trạng thái: Hoàn thành đúng hạn</span></td>
    </tr>
</table>

<h2>1. MỤC TIÊU VÀ NHIỆM VỤ TUẦN 6</h2>
<p>
    Trong giai đoạn Tuần 6 (Giai đoạn Tích hợp hệ thống phát hiện & phân loại rác thải), thành viên phụ trách mảng Thị giác - Ngôn ngữ (VLM) có trách nhiệm chuyển hóa các kết quả nghiên cứu độc lập từ Tuần 5 thành một mô-đun phần mềm hoàn chỉnh, có khả năng kết nối trực tiếp với mô hình phát hiện YOLO và giao diện ứng dụng:
</p>
<ul>
    <li><strong>Nhiệm vụ 1:</strong> Đóng gói tiến trình suy luận của OpenCLIP thành thư viện/API tiêu chuẩn (<code>vlm_inference.py</code>), có khả năng tái sử dụng mô hình tối ưu trên laptop.</li>
    <li><strong>Nhiệm vụ 2:</strong> Chuẩn hóa độ đo tương đồng cosine thành thang điểm xác suất Softmax và tích hợp cơ chế ngưỡng từ chối (Unknown Threshold).</li>
    <li><strong>Nhiệm vụ 3:</strong> Nghiên cứu và đề xuất công thức kết hợp độ tin cậy (Confidence Fusion) giữa YOLO và VLM (<code>confidence_fusion.py</code>), xây dựng quy tắc xử lý khi hai mô hình bất đồng ý kiến.</li>
</ul>

<h2>2. KẾT QUẢ TRIỂN KHAI KỸ THUẬT VÀ MÃ NGUỒN</h2>

<h3>2.1. Đóng gói mô-đun suy luận OpenCLIP (<code>vlm_inference.py</code>)</h3>
<p>
    Mô-đun được xây dựng xoay quanh lớp <code>VLMInference</code>, tích hợp đầy đủ quy trình từ tải trọng số, tiền xử lý ảnh đến phân loại có kiểm soát độ tin cậy:
</p>
<ul>
    <li><strong>Backbone thị giác:</strong> Mô hình OpenCLIP với kiến trúc <code>ViT-B-32-quickgelu</code> sử dụng trọng số tiền huấn luyện <code>openai</code>, cân bằng tối ưu giữa độ chính xác nhận diện ngữ cảnh và tốc độ suy luận trên laptop.</li>
    <li><strong>Prompt Ensemble 6 lớp:</strong> Mỗi lớp trong 6 nhóm rác (<code>cardboard</code>, <code>glass</code>, <code>metal</code>, <code>paper</code>, <code>plastic</code>, <code>trash</code>) sử dụng 5 prompt mô tả trực quan và đặc tính cơ lý để giảm thiểu độ nhạy cảm ngữ vựng.</li>
    <li><strong>Tối ưu hóa Text Feature Cache:</strong> Toàn bộ 30 câu prompt được tokenize và chuyển thành ma trận đặc trưng văn bản [6, 512] ngay khi nạp mô hình. Kỹ thuật này giúp loại bỏ hoàn toàn chi phí chạy Text Encoder trong quá trình dự đoán ảnh, chỉ cần chạy Vision Encoder trên ảnh crop.</li>
    <li><strong>Chuẩn hóa xác suất Softmax:</strong> Tính Cosine Similarity giữa vector ảnh và vector văn bản, nhân thang hệ số 100.0 rồi đưa qua hàm Softmax để đưa điểm số về phân bố xác suất [0, 1].</li>
    <li><strong>Unknown Rejection Threshold:</strong> Thiết lập ngưỡng mặc định <code>threshold = 0.50</code> (cho phép tùy chỉnh). Nếu <code>top_score &lt; threshold</code>, kết quả được đánh dấu trạng thái <code>UNKNOWN</code> nhằm ngăn ngừa dự đoán sai lệch đối với ảnh nhiễu, ảnh không rõ rác hoặc vật thể ngoài tập nhãn.</li>
</ul>

<h3>2.2. Đề xuất công thức và cơ chế kết hợp Confidence (<code>confidence_fusion.py</code>)</h3>
<p>
    Để tận dụng thế mạnh phát hiện vị trí chính xác của YOLO và khả năng hiểu đặc trưng vật liệu chuyên sâu của OpenCLIP, công thức kết hợp độ tin cậy tuyến tính có trọng số được đề xuất:
</p>

<div class="formula-box">
    C_final = α · C_yolo + (1 - α) · C_vlm &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (với α ∈ [0, 1])
</div>

<p>
    Trong đó <em>C_yolo</em> là confidence bounding box từ YOLO, <em>C_vlm</em> là xác suất Softmax của OpenCLIP, và <em>α</em> là trọng số điều phối (mặc định <em>α = 0.5</em>). Cơ chế phân xử và Fallback được đóng gói hoàn chỉnh trong hàm <code>fuse_prediction</code>:
</p>
<ul>
    <li><strong>Trường hợp 1 — Đồng thuận (<code>AGREEMENT</code>):</strong> Khi <code>yolo_class == vlm_class</code> và VLM đạt trạng thái <code>CONFIDENT</code>. Hai mô hình củng cố lẫn nhau, điểm tin cậy tổng thể được cộng hưởng bằng <em>C_final</em>.</li>
    <li><strong>Trường hợp 2 — VLM Unknown (<code>VLM_UNKNOWN_USE_YOLO</code>):</strong> Khi <code>top_score &lt; threshold</code>. Mô hình VLM không đủ tự tin xác nhận, hệ thống tự động giữ nguyên nhãn và điểm từ YOLO.</li>
    <li><strong>Trường hợp 3 — Bất đồng (<code>DISAGREEMENT_USE_YOLO</code>):</strong> Khi hai mô hình dự đoán khác lớp nhau. Theo quy ước tích hợp an toàn tuần 6, hệ thống ưu tiên giữ nhãn YOLO (do YOLO được huấn luyện có giám sát trực tiếp trên tập rác), đồng thời ghi lại nhật ký cảnh báo để phục vụ kiểm thử tuần 7.</li>
</ul>

<h2>3. THỰC NGHIỆM VÀ SỐ LIỆU ĐÁNH GIÁ TRÊN DỮ LIỆU THỰC TẾ</h2>
<p>
    Quá trình đánh giá được thực hiện tự động trên toàn bộ tập <strong>774 crop vật thể rác</strong> trích xuất từ mô hình YOLO baseline (<code>week4_crop_results.csv</code> ghép nối với <code>vlm_batch_results.csv</code>):
</p>

<table class="data-table">
    <thead>
        <tr>
            <th>Kịch bản tích hợp / Xử lý</th>
            <th style="text-align:right;">Số lượng crop</th>
            <th style="text-align:right;">Tỷ lệ (%)</th>
            <th>Phương án giải quyết</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Hai mô hình đồng thuận (Agreement)</strong></td>
            <td style="text-align:right;"><strong>428</strong></td>
            <td style="text-align:right;"><strong>55.30%</strong></td>
            <td>Kết hợp điểm tin cậy C_final</td>
        </tr>
        <tr>
            <td>Hai mô hình bất đồng (Disagreement)</td>
            <td style="text-align:right;">211</td>
            <td style="text-align:right;">27.26%</td>
            <td>Fallback giữ nhãn YOLO (yolo_priority)</td>
        </tr>
        <tr>
            <td>VLM rơi vào ngưỡng Unknown (&lt; 0.50)</td>
            <td style="text-align:right;">135</td>
            <td style="text-align:right;">17.44%</td>
            <td>Fallback giữ nhãn YOLO (VLM_UNKNOWN)</td>
        </tr>
        <tr>
            <td>VLM đạt ngưỡng tự tin (Confident)</td>
            <td style="text-align:right;">639</td>
            <td style="text-align:right;">82.56%</td>
            <td>Đủ điều kiện xét duyệt kết hợp</td>
        </tr>
        <tr class="total">
            <td>Tổng số crop thử nghiệm tích hợp</td>
            <td style="text-align:right;">774</td>
            <td style="text-align:right;">100.0%</td>
            <td>Bộ integration test thực tế</td>
        </tr>
    </tbody>
</table>

<div class="figure-container">
    <img src="data:image/png;base64,{fig1_b64}" alt="Hình 1: Tỷ lệ đồng thuận giữa YOLO và OpenCLIP">
    <div class="figure-caption">Hình 1: Tỷ lệ đồng thuận và phân bổ trạng thái xử lý giữa YOLO và OpenCLIP (774 crop)</div>
</div>

<h3>3.1. Phân tích tương quan theo từng lớp rác thải</h3>
<ul>
    <li><strong>Lớp Thủy tinh (Glass):</strong> Đạt mức tương đồng cao nhất <strong>87.68%</strong> (370/422 mẫu). Các chai, lọ thủy tinh có hình dạng và bề mặt quang học đặc trưng giúp cả hai mô hình nhận diện chính xác.</li>
    <li><strong>Lớp Giấy (Paper):</strong> Đạt tỷ lệ đồng thuận <strong>80.00%</strong> (28/35 mẫu). Sách báo, bìa carton phẳng có đặc trưng vân giấy rất tương thích với bộ từ khóa prompt của OpenCLIP.</li>
    <li><strong>Lớp Kim loại (Metal):</strong> Tỷ lệ đồng thuận chỉ đạt <strong>20.00%</strong> (11/55 mẫu). Nguyên nhân chính là do 32 mẫu lon kim loại có nhãn dán bằng giấy bên ngoài khiến OpenCLIP ưu tiên dự đoán nhãn paper.</li>
    <li><strong>Lớp Rác tổng hợp (Trash):</strong> Độ trùng khớp chỉ đạt <strong>11.01%</strong> do lớp rác tổng hợp chứa nhiều dạng tạp chất không thuần nhất, làm phân tán không gian embedding ngữ nghĩa.</li>
</ul>

<div class="figure-container">
    <img src="data:image/png;base64,{fig2_b64}" alt="Hình 2: Tỷ lệ đồng thuận theo lớp">
    <div class="figure-caption">Hình 2: Tỷ lệ đồng thuận giữa hai mô hình trên các lớp rác thải đại diện</div>
</div>

<h3>3.2. Khảo sát phân bố điểm kết hợp Confidence Fusion theo trọng số α</h3>
<ul>
    <li><strong>Trường hợp α = 0.3 (Ưu tiên VLM 70%):</strong> Điểm trung bình 0.7684, độ lệch chuẩn 0.1412. Điểm chịu ảnh hưởng mạnh bởi độ phân tán của xác suất Softmax VLM.</li>
    <li><strong>Trường hợp α = 0.5 (Cân bằng 50–50):</strong> Điểm trung bình 0.7751, độ lệch chuẩn 0.1260. Đây là cấu hình tối ưu khuyến nghị cho Tuần 6 vì cân bằng được ưu thế của cả hai mô hình.</li>
    <li><strong>Trường hợp α = 0.7 (Ưu tiên YOLO 70%):</strong> Điểm trung bình 0.7818, độ lệch chuẩn 0.1183. Phân bố điểm mượt mà và ổn định theo độ tin cậy của YOLO.</li>
</ul>

<div class="figure-container">
    <img src="data:image/png;base64,{fig3_b64}" alt="Hình 3: Biểu đồ hộp phân bố điểm kết hợp C_final">
    <div class="figure-caption">Hình 3: Biểu đồ hộp phân bố điểm kết hợp C_final ứng với các cấu hình α khác nhau</div>
</div>

<h2>4. KHÓ KHĂN VÀ VẤN ĐỀ PHÁT SINH</h2>
<ul>
    <li><strong>1. Hiện tượng Overconfident của Softmax:</strong> Hàm Softmax khi áp dụng trên nhiệt độ chuẩn (scale 100.0) có xu hướng đẩy điểm cao nhất lên rất sát 1.0 ngay cả khi ảnh crop bị mờ hoặc góc chụp thiếu sáng. Cần khảo sát thêm phương pháp Temperature Scaling hoặc hiệu chuẩn Platt Scaling ở Tuần 8.</li>
    <li><strong>2. Nhiễu bao bì và nhãn dán:</strong> Các vật thể làm từ vật liệu hỗn hợp (ví dụ lon kim loại dán nhãn giấy, chai nhựa bọc màng in màu) tạo ra sự giằng co giữa đặc trưng thị giác bao quát và ngữ nghĩa cục bộ, dẫn đến 211 ca bất đồng ý kiến giữa YOLO và VLM.</li>
    <li><strong>3. Cơ chế phân xử bất đồng:</strong> Cơ chế fallback hiện tại ưu tiên cố định cho YOLO (<code>yolo_priority</code>). Cần phối hợp với TV1 và TV3 để xây dựng luật phân xử động dựa trên độ chênh lệch biên độ tự tin (Confidence Margin).</li>
</ul>

<h2>5. KẾT LUẬN VÀ KẾ HOẠCH BÀN GIAO</h2>
<p>
    Thành viên Lê Đại Cương đã hoàn thành 100% mục tiêu của Tuần 6 đúng theo kế hoạch ban đầu:
</p>
<ul>
    <li><strong>Sản phẩm mã nguồn:</strong> Đã đóng gói hoàn chỉnh thư viện VLM tại <code>d:\DACN\week6\vlm_inference.py</code>.</li>
    <li><strong>Module hợp nhất:</strong> Cung cấp hàm tích hợp <code>fuse_prediction</code> trong <code>d:\DACN\week6\confidence_fusion.py</code> để Đỗ Đức Tiền (TV1) ráp nối vào main pipeline và Chu Tuấn Kiệt (TV5) hiển thị lên giao diện.</li>
    <li><strong>Cập nhật tiến độ:</strong> Đã cập nhật đầy đủ kết quả, tiến độ và khó khăn vào file kế hoạch 10 tuần <code>Ke_hoach_10_tuan_phan_loai_rac_thai.xlsx</code>.</li>
    <li><strong>Chuẩn bị Tuần 7:</strong> Sẵn sàng phối hợp kiểm thử các trường hợp dữ liệu biên (ảnh mờ, tối, vật thể nhỏ, nhãn lai) và tinh chỉnh ngưỡng theo phân công Tuần 7.</li>
</ul>

<div class="signatures">
    <div class="signature-box">
        <strong>XÁC NHẬN CỦA TRƯỞNG NHÓM</strong><br>
        <em>(Ký và ghi rõ họ tên)</em>
        <div class="signature-space"></div>
        <strong>Đỗ Đức Tiền</strong>
    </div>
    <div class="signature-box">
        <em>Hà Nội, ngày 19 tháng 10 năm 2026</em><br>
        <strong>NGƯỜI LẬP BÁO CÁO</strong><br>
        <em>(Ký và ghi rõ họ tên)</em>
        <div class="signature-space"></div>
        <strong>Lê Đại Cương</strong>
    </div>
</div>

</body>
</html>
"""

out_html_dacn = Path(r"d:\DACN\week6\Bao_cao_tuan_6_Le_Dai_Cuong.html")
out_html_down = Path(r"c:\Users\ledai\Searches\Downloads\Bao_cao_tuan_6_Le_Dai_Cuong.html")

out_html_dacn.write_text(html_content, encoding="utf-8")
print("Saved HTML to:", out_html_dacn)
out_html_down.write_text(html_content, encoding="utf-8")
print("Saved HTML to Downloads:", out_html_down)
