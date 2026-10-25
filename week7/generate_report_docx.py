import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
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

    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Header & Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("BÁO CÁO KỸ THUẬT TIẾN ĐỘ TUẦN 7")
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

    # Metadata Box
    tbl_info = doc.add_table(rows=4, cols=2)
    tbl_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_info.autofit = False

    info_data = [
        ("Sinh viên thực hiện:", "Lê Đại Cương — MSSV: 2023604315 (TV4: Mô hình thị giác-ngôn ngữ)"),
        ("Nội dung tuần 7:", "Kiểm thử VLM: lớp gần giống, prompt nhạy cảm, ảnh không thuộc 6 lớp."),
        ("Mục tiêu nghiệm thu:", "Báo cáo độ ổn định prompt và unknown threshold."),
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

    # 1. Mục tiêu
    add_heading_1(doc, "1. MỤC TIÊU VÀ PHẠM VI KIỂM THỬ TUẦN 7")
    add_body_p(doc, "Trong giai đoạn Tuần 7 (Kiểm thử hệ thống), sau khi hoàn thành tích hợp sơ bộ tại Tuần 6, thành viên phụ trách VLM tiến hành kiểm thử chuyên sâu nhằm đánh giá độ tin cậy và giới hạn hoạt động của mô hình OpenCLIP (ViT-B-32-quickgelu). Trọng tâm kiểm thử gồm 3 trụ cột kỹ thuật:")
    add_bullet_p(doc, "Đánh giá khả năng phân biệt của mô hình trên các cặp vật liệu có đặc trưng thị giác tương đồng (plastic vs glass, metal vs paper, cardboard vs paper, trash hỗn hợp).", "Trụ cột 1 — Lớp gần giống: ")
    add_bullet_p(doc, "Kiểm tra mức độ nhạy cảm và độ biến động dự đoán khi thay đổi cấu trúc câu nhắc từ ngắn đơn giản, mô tả chi tiết, prompt ensemble đến prompt nhiễu bối cảnh.", "Trụ cột 2 — Độ ổn định Prompt: ")
    add_bullet_p(doc, "Kiểm thử khả năng từ chối đối với ảnh ngoài tập dữ liệu 6 lớp (đồ vật thường nhật, nhiễu tổng hợp, rác ngoài phạm vi) và xác lập dải ngưỡng unknown threshold tối ưu.", "Trụ cột 3 — Out-of-Distribution (OOD): ")

    # 2. Kết quả Trụ cột 1
    add_heading_1(doc, "2. KẾT QUẢ KIỂM THỬ CÁC LỚP DỄ NHẦM LẪN (SIMILAR CLASSES)")
    add_body_p(doc, "Thực nghiệm được triển khai trên toàn bộ 595 ảnh rác thải có gán nhãn thực tế thuộc 6 nhóm. Kết quả chung đạt độ chính xác Accuracy 80.67% (Weighted F1: 0.8132).")

    # Bảng Confusion Matrix
    add_heading_2(doc, "2.1. Ma trận nhầm lẫn và chỉ số phân loại chi tiết")
    tbl_cm = doc.add_table(rows=7, cols=7)
    tbl_cm.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_cm.autofit = False

    cm_headers = ["Lớp (Ground Truth)", "cardboard", "glass", "metal", "paper", "plastic", "trash"]
    cm_widths = [Inches(1.8)] + [Inches(0.8)] * 6

    for c_idx, (h_text, w) in enumerate(zip(cm_headers, cm_widths)):
        cell = tbl_cm.rows[0].cells[c_idx]
        cell.width = w
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    cm_data = [
        ("cardboard (24)", "20", "0", "0", "4", "0", "0"),
        ("glass (400)", "0", "372", "14", "0", "14", "0"),
        ("metal (22)", "1", "0", "15", "0", "2", "4"),
        ("paper (57)", "7", "0", "1", "43", "0", "6"),
        ("plastic (33)", "1", "10", "13", "0", "9", "0"),
        ("trash (59)", "4", "4", "4", "9", "17", "21")
    ]

    for r_idx, row_vals in enumerate(cm_data):
        row = tbl_cm.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = cm_widths[c_idx]
            is_diag = (c_idx == r_idx + 1)
            set_cell_background(cell, "D4EDDA" if is_diag else ("F9FAFC" if r_idx % 2 == 0 else "FFFFFF"))
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            if c_idx > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            if is_diag or c_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Insert Fig 1
    fig1 = Path(r"d:\DACN\week7\figures\fig1_similar_classes_cm.png")
    if fig1.exists():
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(fig1), width=Inches(4.8))
        p_c1 = doc.add_paragraph()
        p_c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c1.paragraph_format.space_after = Pt(8)
        r = p_c1.add_run("Hình 1: Heatmap ma trận nhầm lẫn giữa 6 nhóm rác thải trên mô hình OpenCLIP")
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.italic = True

    add_heading_2(doc, "2.2. Phân tích các cặp vật liệu gây nhầm lẫn chính")
    add_bullet_p(doc, "10/33 mẫu chai nhựa (30.30%) bị nhận diện nhầm sang thủy tinh. Do chai nhựa PET trong suốt phản quang và khúc xạ ánh sáng rất giống chai thủy tinh trong ảnh chụp tĩnh.", "Nhựa vs Thủy tinh (Plastic -> Glass): ")
    add_bullet_p(doc, "4/24 mẫu bìa carton (16.67%) bị nhầm sang giấy, và 7/57 mẫu giấy in (12.28%) bị nhầm sang carton. Bản chất carton là sản phẩm tái chế từ giấy nhiều lớp nên phân bố embedding ngữ nghĩa rất sát nhau.", "Carton vs Giấy (Cardboard <-> Paper): ")
    add_bullet_p(doc, "17/59 mẫu (28.81%) bị nhầm sang nhựa và 9/59 mẫu (15.25%) nhầm sang giấy do rác sinh hoạt hỗn hợp chứa nhiều túi nilon và vỏ hộp, khiến mô hình thiên về vật liệu có diện tích bề mặt chiếm ưu thế.", "Rác tổng hợp (Trash): ")

    # Insert Fig 2
    fig2 = Path(r"d:\DACN\week7\figures\fig2_confidence_margin.png")
    if fig2.exists():
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(fig2), width=Inches(5.0))
        p_c2 = doc.add_paragraph()
        p_c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c2.paragraph_format.space_after = Pt(8)
        r = p_c2.add_run("Hình 2: Phân bố biên độ chênh lệch độ tin cậy (Margin Δ = Top-1 - Top-2)")
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.italic = True

    add_body_p(doc, "Phân tích biên độ Margin cho thấy: các mẫu dự đoán sai hầu hết có biên độ chênh lệch rất nhỏ (Δ < 0.15). Đây là cơ sở then chốt để áp dụng quy tắc phân xử phối hợp với YOLO ở Tuần 8.")

    # 3. Kết quả Trụ cột 2
    add_heading_1(doc, "3. KẾT QUẢ KIỂM THỬ ĐỘ NHẠY CẢM VÀ ỔN ĐỊNH CỦA PROMPT")
    add_body_p(doc, "Thực nghiệm tiến hành trên 166 ảnh mẫu chuẩn hóa (30 ảnh/lớp) nhằm so sánh 5 chiến lược soạn thảo prompt:")
    add_bullet_p(doc, "Độ chính xác chỉ đạt 56.02%, độ nhất quán so với Ensemble chỉ đạt 74.70% (42 mẫu bị đảo nhãn). Từ khóa đơn lẻ quá mơ hồ và dễ bị nhiễu ngữ cảnh.", "Chiến lược 1 (P1 - Single Short): ")
    add_bullet_p(doc, "Độ chính xác tăng lên 63.86%, độ nhất quán đạt 82.53% (29 mẫu bị đảo nhãn). Bổ sung từ khóa 'waste' định hướng vật thể vào ngữ cảnh phế liệu rõ rệt.", "Chiến lược 2 (P2 - Template Standard): ")
    add_bullet_p(doc, "Độ chính xác đạt 63.25%, điểm tin cậy trung bình tăng lên 0.7264.", "Chiến lược 3 (P3 - Descriptive 3): ")
    add_bullet_p(doc, "Độ chính xác 63.25%, phân bố điểm tin cậy ổn định nhất (Std = 0.2170), đóng vai trò baseline tham chiếu chuẩn mực.", "Chiến lược 4 (P4 - Ensemble 5 Prompts): ")
    add_bullet_p(doc, "Gây ra tới 70/166 trường hợp đảo nhãn (tỷ lệ nhất quán rớt xuống 57.83%). Từ khóa về nền nhà, bàn gỗ làm lệch không gian vector nghiêm trọng.", "Chiến lược 5 (P5 - Contextual Distracted): ")

    # Insert Fig 3
    fig3 = Path(r"d:\DACN\week7\figures\fig3_prompt_stability.png")
    if fig3.exists():
        p_img3 = doc.add_paragraph()
        p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(fig3), width=Inches(5.0))
        p_c3 = doc.add_paragraph()
        p_c3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c3.paragraph_format.space_after = Pt(8)
        r = p_c3.add_run("Hình 3: So sánh độ chính xác và tính nhất quán giữa 5 chiến lược Prompt")
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.italic = True

    # 4. Kết quả Trụ cột 3
    add_heading_1(doc, "4. KIỂM THỬ ẢNH NGOÀI TẬP (OOD) VÀ XÁC ĐỊNH NGƯỠNG UNKNOWN")
    add_body_p(doc, "Tập thử nghiệm gồm 166 ảnh rác thực tế (In-Distribution - ID) và 59 ảnh ngoài tập rác (Out-of-Distribution - OOD) gồm các thiết bị điện tử, đồ gia dụng, cảnh sinh hoạt và nhiễu tổng hợp. Tiến hành quét dải ngưỡng τ ∈ [0.30, 0.80]:")

    # Table sweep
    tbl_sw = doc.add_table(rows=7, cols=5)
    tbl_sw.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_sw.autofit = False

    sw_headers = ["Ngưỡng τ", "True Rejection (OOD %)", "False Rejection (Mất ID %)", "Retained Accuracy (%)", "Balanced F1 (%)"]
    sw_widths = [Inches(1.2), Inches(1.4), Inches(1.4), Inches(1.3), Inches(1.2)]

    for c_idx, (h_text, w) in enumerate(zip(sw_headers, sw_widths)):
        cell = tbl_sw.rows[0].cells[c_idx]
        cell.width = w
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    sw_rows = [
        ("0.30", "40.68%", "2.41%", "63.58%", "57.42%"),
        ("0.40", "72.88%", "13.86%", "67.83%", "78.96%"),
        ("0.45 (Tối ưu)", "81.36%", "21.08%", "70.99%", "80.12%"),
        ("0.50 (An toàn)", "89.83%", "29.52%", "74.36%", "78.99%"),
        ("0.60", "94.92%", "38.55%", "77.45%", "74.60%"),
        ("0.70", "98.31%", "52.41%", "86.08%", "64.13%")
    ]

    for r_idx, row_vals in enumerate(sw_rows):
        row = tbl_sw.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = sw_widths[c_idx]
            is_best = ("Tối ưu" in row_vals[0] or "An toàn" in row_vals[0])
            set_cell_background(cell, "FFF3CD" if is_best else ("F9FAFC" if r_idx % 2 == 0 else "FFFFFF"))
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            if c_idx > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            if is_best or c_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Insert Fig 4
    fig4 = Path(r"d:\DACN\week7\figures\fig4_ood_rejection_curves.png")
    if fig4.exists():
        p_img4 = doc.add_paragraph()
        p_img4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(fig4), width=Inches(5.0))
        p_c4 = doc.add_paragraph()
        p_c4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c4.paragraph_format.space_after = Pt(8)
        r = p_c4.add_run("Hình 4: Đường cong đánh đổi giữa tỷ lệ loại bỏ đúng OOD và bỏ sót rác thật theo ngưỡng τ")
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.italic = True

    add_body_p(doc, "Khuyến nghị chốt cấu hình ngưỡng cho hệ thống tích hợp:")
    add_bullet_p(doc, "Điểm cân bằng F1 cao nhất (80.12%), giữ lại gần 80% mẫu rác thật trong khi chặn được hơn 81% vật thể lạ.", "Ngưỡng nhạy tối ưu (τ = 0.45): ")
    add_bullet_p(doc, "Chặn đứng 89.83% vật thể OOD, nâng độ chính xác của các mẫu giữ lại lên 74.36%. Đây là mức khuyến nghị cho phiên bản ứng dụng laptop.", "Ngưỡng an toàn thực tế (τ = 0.50): ")

    # 5. Khó khăn & Kết luận
    add_heading_1(doc, "5. KHÓ KHĂN PHÁT SINH VÀ ĐỀ XUẤT GIẢI PHÁP CHO TUẦN 8")
    add_bullet_p(doc, "Độ tương đồng ngữ nghĩa và thị giác giữa chai nhựa trong suốt và thủy tinh quá cao, dẫn tới 30.3% chai nhựa bị nhầm. Đề xuất: Phối hợp với YOLOv8 để bổ sung thông tin hình học bounding box (nhựa thường có nắp vặn hoặc vết méo bẹp).", "1. Nhầm lẫn Plastic - Glass: ")
    add_bullet_p(doc, "Khi tăng ngưỡng τ lên trên 0.60, tỷ lệ bỏ sót rác thật (FRR) tăng vọt lên gần 40%, gây lãng phí dữ liệu phát hiện từ YOLO. Khẳng định ngưỡng 0.45 - 0.50 là vùng tối ưu không nên vượt quá.", "2. Đánh đổi ngưỡng Unknown: ")
    add_bullet_p(doc, "Các prompt có chứa thông tin bối cảnh (nền gỗ, sàn nhà) làm giảm mạnh độ ổn định. Do đó, toàn bộ pipeline tích hợp chỉ sử dụng bộ prompt thuần vật liệu Ensemble P4.", "3. Chuẩn hóa Prompt cố định: ")

    add_heading_1(doc, "6. KẾT LUẬN VÀ BÀN GIAO")
    add_body_p(doc, "Thành viên Lê Đại Cương đã hoàn thành 100% nhiệm vụ Tuần 7 đúng tiến độ, bàn giao đầy đủ mã nguồn kiểm thử và các kết luận thực nghiệm phục vụ việc đánh giá mô hình kết hợp ở Tuần 8.")

    # Signatures
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
    r_r1 = p_r.add_run("Hà Nội, ngày 25 tháng 10 năm 2026\nNGƯỜI LẬP BÁO CÁO\n")
    r_r1.font.name = "Times New Roman"
    r_r1.font.size = Pt(11)
    r_r1.font.bold = True
    r_r2 = p_r.add_run("(Ký và ghi rõ họ tên)\n\n\n\nLê Đại Cương")
    r_r2.font.name = "Times New Roman"
    r_r2.font.size = Pt(11)

    out_docx_dacn = Path(r"d:\DACN\week7\Bao_cao_tuan_7_Le_Dai_Cuong.docx")
    out_docx_down = Path(r"c:\Users\ledai\Searches\Downloads\Bao_cao_tuan_7_Le_Dai_Cuong.docx")

    doc.save(str(out_docx_dacn))
    print("Saved DOCX to:", out_docx_dacn)
    try:
        doc.save(str(out_docx_down))
        print("Saved DOCX to Downloads:", out_docx_down)
    except Exception as e:
        print("Downloads copy note:", e)

if __name__ == "__main__":
    main()
