import os

base_dir = r"d:\DoAnChuyenNganh"
dataset_path = os.path.join(base_dir, "yolov12_dataset")
splits = ['train', 'valid', 'test']

fixed_count = 0

for split in splits:
    lbl_dir = os.path.join(dataset_path, split, "labels")
    if not os.path.exists(lbl_dir):
        continue
        
    for lbl_name in os.listdir(lbl_dir):
        if not lbl_name.lower().endswith('.txt'):
            continue
            
        lbl_file = os.path.join(lbl_dir, lbl_name)
        
        with open(lbl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        is_modified = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            
            # If it's a Bounding Box format (class_id xc yc w h)
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    # Calculate 4 corners of the bounding box
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
                    # In case of invalid values, keep original line to not destroy data
                    new_lines.append(line)
            else:
                new_lines.append(line)
                
        if is_modified:
            with open(lbl_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines) + '\n')

print(f"Đã chuyển đổi thành công {fixed_count} nhãn Bounding Box sang Polygon.")
