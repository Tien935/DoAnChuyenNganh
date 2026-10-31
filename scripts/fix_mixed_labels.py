"""
fix_mixed_labels.py — Chuyển tất cả nhãn BBox (5 cột) sang Polygon (9 cột)
để dataset tương thích với YOLOv11s-seg / YOLOv12s-seg
"""
import os

# Tự động xác định đường dẫn dataset theo vị trí file script
base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "dataset")
splits = ['train', 'valid', 'test']

fixed_count = 0
file_count  = 0

print(f"Dataset path: {dataset_path}")

for split in splits:
    lbl_dir = os.path.join(dataset_path, split, "labels")
    if not os.path.exists(lbl_dir):
        print(f"[BỎ QUA] Không tìm thấy: {lbl_dir}")
        continue

    for lbl_name in os.listdir(lbl_dir):
        if not lbl_name.lower().endswith('.txt'):
            continue

        lbl_file = os.path.join(lbl_dir, lbl_name)

        with open(lbl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines  = []
        is_modified = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Nếu là BBox format (class_id xc yc w h) — 5 giá trị
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    # Chuyển thành 4 góc hình chữ nhật (polygon 4 điểm)
                    x1 = max(0.0, xc - w / 2)
                    y1 = max(0.0, yc - h / 2)
                    x2 = min(1.0, xc + w / 2)
                    y2 = max(0.0, yc - h / 2)
                    x3 = min(1.0, xc + w / 2)
                    y3 = min(1.0, yc + h / 2)
                    x4 = max(0.0, xc - w / 2)
                    y4 = min(1.0, yc + h / 2)

                    new_line = f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                    new_lines.append(new_line)
                    is_modified = True
                    fixed_count += 1
                except ValueError:
                    new_lines.append(line)
            else:
                # Đã là polygon format — giữ nguyên
                new_lines.append(line)

        if is_modified:
            file_count += 1
            with open(lbl_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines) + '\n')

print(f"\n✅ Hoàn tất!")
print(f"   Số nhãn đã chuyển đổi : {fixed_count}")
print(f"   Số file đã sửa        : {file_count}")
print(f"\nBây giờ có thể train YOLOv11s-seg!")
