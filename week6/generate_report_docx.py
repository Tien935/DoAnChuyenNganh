import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_heading_1(doc, text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(13.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(27, 54, 93)
    return h

def add_heading_2(doc, text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(3)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = RGBColor(41, 128, 185)
    return h

def add_body_p(doc, text, bold_prefix=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    return p

def add_bullet_p(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    return p

def main():
    doc = docx.Document()

    # Margin setup (A4 standard: 1 inch = 2.54 cm)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # 1. Header & Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("BÁO CÁO KỸ THUẬT TIẾN ĐỘ TUẦN 6")
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(27, 54, 93)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(12)
    run_sub = sub_p.add_run("ĐỀ TÀI: ỨNG DỤNG THỊ GIÁC MÁY TÍNH TRONG PHÁT HIỆN VÀ PHÂN LOẠI RÁC THẢI\nCHUYÊN TRÁCH: MÔ HÌNH THỊ GIÁC - NGÔN NGỮ (VLM)")
    run_sub.font.name = "Times New Roman"
    run_sub.font.size = Pt(11)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(90, 90, 90)

    # 2. Metadata Box
    tbl_info = doc.add_table(rows=4, cols=2)
    tbl_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_info.autofit = False

    info_data = [
        ("Sinh viên thực hiện:", "Lê Đại Cương — MSSV: 2023604315 (TV4: Mô hình thị giác-ngôn ngữ)"),
        ("Nội dung tuần 6:", "Đóng gói OpenCLIP inference; chuẩn hóa điểm; đề xuất công thức kết hợp confidence."),
        ("Mục tiêu nghiệm thu:", "VLM module hoàn chỉnh, chuẩn hóa API và quy tắc hợp nhất điểm tin cậy phục vụ pipeline."),
        ("Đánh giá kết quả:", "Tiến độ: 100% — Trạng thái: Hoàn thành đúng kế hoạch")
    ]

    col_widths = [Inches(1.8), Inches(4.7)]
    for row_idx, (k, v) in enumerate(info_data):
        row = tbl_info.rows[row_idx]
        for c_idx, w in enumerate(col_widths):
            cell = row.cells[c_idx]
            cell.width = w
            set_cell_background(cell, "F4F6F9" if row_idx % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(k if c_idx == 0 else v)
            r.font.name = "Times New Roman"
            r.font.size = Pt(10.5)
            if c_idx == 0 or (row_idx == 3 and c_idx == 1):
                r.font.bold = True
            if row_idx == 3 and c_idx == 1:
                r.font.color.rgb = RGBColor(46, 125, 50)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3. Content Sections
    add_heading_1(doc, "1. MỤC TIÊU VÀ NHIỆM VỤ TUẦN 6")
    add_body_p(doc, "Trong giai đoạn Tuần 6 (Giai đoạn Tích hợp hệ thống phát hiện & phân loại rác thải), thành viên phụ trách mảng Thị giác - Ngôn ngữ (VLM) có trách nhiệm chuyển hóa các kết quả nghiên cứu độc lập từ Tuần 5 thành một mô-đun phần mềm hoàn chỉnh, có khả năng kết nối trực tiếp với mô hình phát hiện YOLO và giao diện người dùng.")
    add_bullet_p(doc, "Đóng gói tiến trình suy luận của OpenCLIP thành thư viện/API tiêu chuẩn, có khả năng tái sử dụng mô hình tối ưu trên thiết bị laptop.", "Nhiệm vụ 1: ")
    add_bullet_p(doc, "Chuẩn hóa độ đo tương đồng cosine thành thang điểm xác suất và tích hợp cơ chế ngưỡng từ chối (Unknown Threshold).", "Nhiệm vụ 2: ")
    add_bullet_p(doc, "Nghiên cứu và đề xuất công thức kết hợp độ tin cậy (Confidence Fusion) giữa YOLO và VLM, xây dựng quy tắc xử lý khi hai mô hình bất đồng ý kiến.", "Nhiệm vụ 3: ")

    add_heading_1(doc, "2. KẾT QUẢ TRIỂN KHAI KỸ THUẬT VÀ MÃ NGUỒN")

    add_heading_2(doc, "2.1. Đóng gói mô-đun suy luận OpenCLIP (vlm_inference.py)")
    add_body_p(doc, "Mô-đun được xây dựng xoay quanh lớp VLMInference, tích hợp đầy đủ quy trình từ tải trọng số, tiền xử lý ảnh đến phân loại có kiểm soát độ tin cậy:")
    add_bullet_p(doc, "Mô hình OpenCLIP với kiến trúc ViT-B-32-quickgelu sử dụng trọng số tiền huấn luyện openai, cân bằng tối ưu giữa độ chính xác nhận diện ngữ cảnh và tốc độ suy luận trên laptop.", "Backbone thị giác: ")
    add_bullet_p(doc, "Thay vì dùng một câu mô tả duy nhất, mỗi lớp trong 6 nhóm rác (cardboard, glass, metal, paper, plastic, trash) sử dụng 5 prompt mô tả trực quan và đặc tính cơ lý để giảm thiểu độ nhạy cảm từ vựng.", "Prompt Ensemble 6 lớp: ")
    add_bullet_p(doc, "Toàn bộ 30 câu prompt được tokenize và chuyển thành ma trận đặc trưng văn bản [6, 512] ngay khi nạp mô hình. Kỹ thuật này giúp loại bỏ 100% thời gian chạy Text Encoder trong quá trình dự đoán ảnh, chỉ cần chạy Vision Encoder trên ảnh crop.", "Tối ưu hóa Text Feature Cache: ")
    add_bullet_p(doc, "Tính tích vô hướng (Cosine Similarity) giữa vector ảnh và vector văn bản, nhân thang hệ số 100.0 rồi đưa qua hàm Softmax để đưa điểm số về phân bố xác suất [0, 1].", "Chuẩn hóa xác suất Softmax: ")
    add_bullet_p(doc, "Thiết lập ngưỡng mặc định threshold = 0.50 (cho phép tùy chỉnh). Nếu top_score < threshold, kết quả được đánh dấu trạng thái UNKNOWN nhằm ngăn ngừa dự đoán sai lệch đối với ảnh nhiễu, ảnh không rõ rác hoặc vật thể ngoài tập nhãn.", "Unknown Rejection Threshold: ")

    add_heading_2(doc, "2.2. Đề xuất công thức và cơ chế kết hợp Confidence (confidence_fusion.py)")
    add_body_p(doc, "Để tận dụng thế mạnh phát hiện vị trí chính xác của YOLO và khả năng hiểu đặc trưng vật liệu chuyên sâu của OpenCLIP, công thức kết hợp độ tin cậy tuyến tính có trọng số được đề xuất:")
    
    # Formula box
    p_formula = doc.add_paragraph()
    p_formula.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_formula.paragraph_format.space_before = Pt(4)
    p_formula.paragraph_format.space_after = Pt(4)
    r_f = p_formula.add_run("C_final = α · C_yolo + (1 - α) · C_vlm       (với α ∈ [0, 1])")
    r_f.font.name = "Consolas"
    r_f.font.size = Pt(11)
    r_f.font.bold = True
    r_f.font.color.rgb = RGBColor(180, 40, 20)

    add_body_p(doc, "Trong đó C_yolo là confidence bounding box từ YOLO, C_vlm là xác suất Softmax của OpenCLIP, và α là trọng số điều phối (mặc định α = 0.5 thể hiện sự tin cậy tương đương).")
    add_body_p(doc, "Cơ chế phân xử và Fallback được đóng gói hoàn chỉnh trong hàm fuse_prediction(yolo_class, yolo_conf, vlm_result, alpha, conflict_strategy):")
    add_bullet_p(doc, "Khi yolo_class == vlm_class và VLM đạt trạng thái CONFIDENT. Hai mô hình củng cố lẫn nhau, điểm tin cậy tổng thể được cập nhật bằng C_final.", "Trường hợp 1 — Đồng thuận (AGREEMENT): ")
    add_bullet_p(doc, "Khi top_score của VLM < threshold. Mô hình VLM không đủ tự tin xác nhận, hệ thống tự động giữ nguyên nhãn và điểm từ YOLO.", "Trường hợp 2 — VLM Unknown (VLM_UNKNOWN_USE_YOLO): ")
    add_bullet_p(doc, "Khi hai mô hình dự đoán 2 lớp rác khác nhau. Theo quy ước tích hợp an toàn tuần 6, hệ thống ưu tiên giữ nhãn YOLO (do YOLO được huấn luyện có giám sát trực tiếp trên tập rác), đồng thời ghi lại nhật ký cảnh báo để phục vụ kiểm thử tuần 7.", "Trường hợp 3 — Bất đồng (DISAGREEMENT_USE_YOLO): ")

    add_heading_1(doc, "3. THỰC NGHIỆM VÀ SỐ LIỆU ĐÁNH GIÁ TRÊN DỮ LIỆU THỰC TẾ")
    add_body_p(doc, "Quá trình đánh giá được thực hiện tự động trên toàn bộ tập 774 crop vật thể rác trích xuất từ mô hình YOLO baseline (week4_crop_results.csv ghép nối với vlm_batch_results.csv):")

    # Table of stats
    tbl_stats = doc.add_table(rows=6, cols=4)
    tbl_stats.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_stats.autofit = False

    headers = ["Kịch bản tích hợp / Xử lý", "Số lượng crop", "Tỷ lệ (%)", "Phương án giải quyết"]
    header_widths = [Inches(2.5), Inches(1.1), Inches(1.1), Inches(1.8)]

    stats_rows = [
        ("Hai mô hình đồng thuận (Agreement)", "428", "55.30%", "Kết hợp điểm C_final"),
        ("Hai mô hình bất đồng (Disagreement)", "211", "27.26%", "Fallback giữ nhãn YOLO"),
        ("VLM rơi vào ngưỡng Unknown (< 0.50)", "135", "17.44%", "Fallback giữ nhãn YOLO"),
        ("VLM đạt ngưỡng tự tin (Confident)", "639", "82.56%", "Đủ điều kiện xét duyệt"),
        ("Tổng số crop thử nghiệm tích hợp", "774", "100.0%", "Bộ integration test")
    ]

    for c_idx, (h_text, w) in enumerate(zip(headers, header_widths)):
        cell = tbl_stats.rows[0].cells[c_idx]
        cell.width = w
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row_vals in enumerate(stats_rows):
        row = tbl_stats.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = header_widths[c_idx]
            set_cell_background(cell, "F9FAFC" if r_idx % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            if c_idx in [1, 2]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            if r_idx == 4 or c_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Insert Figure 1
    fig1_path = Path(r"d:\DACN\week6\figures\fig1_agreement_pie.png")
    if fig1_path.exists():
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_before = Pt(4)
        p_img1.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(fig1_path), width=Inches(5.0))
        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap1.paragraph_format.space_after = Pt(8)
        r_c1 = p_cap1.add_run("Hình 1: Tỷ lệ đồng thuận và phân bổ trạng thái xử lý giữa YOLO và OpenCLIP (774 crop)")
        r_c1.font.name = "Times New Roman"
        r_c1.font.size = Pt(9.5)
        r_c1.font.italic = True

    add_heading_2(doc, "3.1. Phân tích tương quan theo từng lớp rác thải")
    add_body_p(doc, "Thống kê so sánh chi tiết cho thấy sự khác biệt rõ rệt giữa các nhóm vật liệu:")
    add_bullet_p(doc, "Đạt mức tương đồng cao nhất 87.68% (370/422 mẫu). Các chai, lọ thủy tinh có hình dạng và bề mặt quang học đặc trưng giúp cả hai mô hình nhận diện chính xác.", "Lớp Thủy tinh (Glass): ")
    add_bullet_p(doc, "Đạt tỷ lệ đồng thuận 80.00% (28/35 mẫu). Sách báo, thùng carton phẳng có đặc trưng vân giấy rất tương thích với bộ từ khóa prompt của OpenCLIP.", "Lớp Giấy (Paper): ")
    add_bullet_p(doc, "Tỷ lệ đồng thuận chỉ đạt 20.00% (11/55 mẫu). Nguyên nhân chính là do 32 mẫu lon kim loại có nhãn dán bằng giấy bên ngoài khiến OpenCLIP ưu tiên dự đoán nhãn paper.", "Lớp Kim loại (Metal): ")
    add_bullet_p(doc, "Độ trùng khớp chỉ đạt 11.01% do lớp rác tổng hợp chứa nhiều dạng tạp chất không thuần nhất, làm phân tán không gian embedding ngữ nghĩa.", "Lớp Rác tổng hợp (Trash): ")

    # Insert Figure 2
    fig2_path = Path(r"d:\DACN\week6\figures\fig2_agreement_by_class.png")
    if fig2_path.exists():
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.paragraph_format.space_before = Pt(4)
        p_img2.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(fig2_path), width=Inches(5.2))
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap2.paragraph_format.space_after = Pt(8)
        r_c2 = p_cap2.add_run("Hình 2: Tỷ lệ đồng thuận giữa hai mô hình trên các lớp rác thải đại diện")
        r_c2.font.name = "Times New Roman"
        r_c2.font.size = Pt(9.5)
        r_c2.font.italic = True

    add_heading_2(doc, "3.2. Khảo sát phân bố điểm kết hợp Confidence Fusion theo trọng số α")
    add_body_p(doc, "Trên 428 trường hợp đồng thuận, điểm kết hợp C_final được khảo sát với 3 giá trị trọng số α đại diện:")
    add_bullet_p(doc, "Điểm trung bình 0.7684, độ lệch chuẩn 0.1412. Điểm chịu ảnh hưởng mạnh bởi độ phân tán của xác suất Softmax VLM.", "Trường hợp α = 0.3 (Ưu tiên VLM 70%): ")
    add_bullet_p(doc, "Điểm trung bình 0.7751, độ lệch chuẩn 0.1260. Đây là cấu hình tối ưu khuyến nghị cho Tuần 6 vì cân bằng được ưu thế của cả hai mô hình.", "Trường hợp α = 0.5 (Cân bằng 50–50): ")
    add_bullet_p(doc, "Điểm trung bình 0.7818, độ lệch chuẩn 0.1183. Phân bố điểm mượt mà và ổn định theo độ tin cậy của YOLO.", "Trường hợp α = 0.7 (Ưu tiên YOLO 70%): ")

    # Insert Figure 3
    fig3_path = Path(r"d:\DACN\week6\figures\fig3_alpha_distribution.png")
    if fig3_path.exists():
        p_img3 = doc.add_paragraph()
        p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img3.paragraph_format.space_before = Pt(4)
        p_img3.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(fig3_path), width=Inches(5.2))
        p_cap3 = doc.add_paragraph()
        p_cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap3.paragraph_format.space_after = Pt(8)
        r_c3 = p_cap3.add_run("Hình 3: Biểu đồ hộp phân bố điểm kết hợp C_final ứng với các cấu hình α khác nhau")
        r_c3.font.name = "Times New Roman"
        r_c3.font.size = Pt(9.5)
        r_c3.font.italic = True

    add_heading_1(doc, "4. KHÓ KHĂN VÀ VẤN ĐỀ PHÁT SINH")
    add_bullet_p(doc, "Hàm Softmax khi áp dụng trên nhiệt độ chuẩn (scale 100.0) có xu hướng đẩy điểm cao nhất lên rất sát 1.0 (overconfident) ngay cả khi ảnh crop bị mờ hoặc góc chụp thiếu sáng. Cần khảo sát thêm phương pháp Temperature Scaling hoặc hiệu chuẩn Platt Scaling ở Tuần 8.", "1. Hiện tượng Overconfident của Softmax: ")
    add_bullet_p(doc, "Các vật thể làm từ vật liệu hỗn hợp (ví dụ hộp sữa giấy có nắp nhựa, chai thủy tinh dán nhãn giấy rộng) tạo ra sự giằng co giữa đặc trưng thị giác bao quát và ngữ nghĩa cục bộ, dẫn đến 211 ca bất đồng ý kiến giữa YOLO và VLM.", "2. Nhiễu bao bì và nhãn dán: ")
    add_bullet_p(doc, "Cơ chế fallback hiện tại ưu tiên cố định cho YOLO (yolo_priority). Cần phối hợp với TV1 và TV3 để xây dựng luật phân xử động dựa trên độ chênh lệch biên độ tự tin (Confidence Margin).", "3. Cơ chế phân xử bất đồng: ")

    add_heading_1(doc, "5. KẾT LUẬN VÀ KẾ HOẠCH BÀN GIAO")
    add_body_p(doc, "Thành viên Lê Đại Cương đã hoàn thành 100% mục tiêu của Tuần 6 đúng theo kế hoạch ban đầu:")
    add_bullet_p(doc, "Đã đóng gói hoàn chỉnh thư viện VLM và lưu trữ trong repository tại d:\\DACN\\week6\\vlm_inference.py.", "Sản phẩm mã nguồn: ")
    add_bullet_p(doc, "Cung cấp hàm tích hợp fuse_prediction trong d:\\DACN\\week6\\confidence_fusion.py để Đỗ Đức Tiền (TV1) ráp nối vào main orchestrator pipeline và Chu Tuấn Kiệt (TV5) hiển thị lên giao diện desktop.", "Module hợp nhất: ")
    add_bullet_p(doc, "Đã cập nhật đầy đủ kết quả, tiến độ và khó khăn vào file kế hoạch 10 tuần Ke_hoach_10_tuan_phan_loai_rac_thai.xlsx.", "Cập nhật tiến độ: ")
    add_bullet_p(doc, "Sẵn sàng phối hợp kiểm thử các trường hợp dữ liệu biên (ảnh mờ, tối, vật thể nhỏ, nhãn lai) và tinh chỉnh ngưỡng theo phân công Tuần 7.", "Chuẩn bị Tuần 7: ")

    # Signature section
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    tbl_sign = doc.add_table(rows=1, cols=2)
    tbl_sign.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_sign.autofit = False
    
    cell_l = tbl_sign.rows[0].cells[0]
    cell_l.width = Inches(3.2)
    p_l = cell_l.paragraphs[0]
    p_l.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_l1 = p_l.add_run("XÁC NHẬN CỦA TRƯỞNG NHÓM\n")
    r_l1.font.name = "Times New Roman"
    r_l1.font.size = Pt(11)
    r_l1.font.bold = True
    r_l2 = p_l.add_run("(Ký và ghi rõ họ tên)\n\n\n\nĐỗ Đức Tiền")
    r_l2.font.name = "Times New Roman"
    r_l2.font.size = Pt(11)

    cell_r = tbl_sign.rows[0].cells[1]
    cell_r.width = Inches(3.3)
    p_r = cell_r.paragraphs[0]
    p_r.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_r1 = p_r.add_run("Hà Nội, ngày 19 tháng 10 năm 2026\nNGƯỜI LẬP BÁO CÁO\n")
    r_r1.font.name = "Times New Roman"
    r_r1.font.size = Pt(11)
    r_r1.font.bold = True
    r_r2 = p_r.add_run("(Ký và ghi rõ họ tên)\n\n\n\nLê Đại Cương")
    r_r2.font.name = "Times New Roman"
    r_r2.font.size = Pt(11)

    # Save to workspace and Downloads
    out_docx_dacn = Path(r"d:\DACN\week6\Bao_cao_tuan_6_Le_Dai_Cuong.docx")
    out_docx_down = Path(r"c:\Users\ledai\Searches\Downloads\Bao_cao_tuan_6_Le_Dai_Cuong.docx")

    doc.save(str(out_docx_dacn))
    print("Saved DOCX to:", out_docx_dacn)
    try:
        doc.save(str(out_docx_down))
        print("Saved DOCX to Downloads:", out_docx_down)
    except Exception as e:
        print("Downloads copy note:", e)

if __name__ == "__main__":
    main()
