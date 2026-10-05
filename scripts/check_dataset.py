import os
import yaml
from collections import defaultdict

# Paths
base_dir = r"d:\DoAnChuyenNganh"
dataset_path = os.path.join(base_dir, "yolov12_dataset")
yaml_path = os.path.join(dataset_path, "data.yaml")

# Load yaml
try:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading yaml: {e}")
    exit(1)

classes = data_yaml['names']
nc = data_yaml['nc']

splits = ['train', 'valid', 'test']
total_images = 0
total_labels = 0
missing_images = []
missing_labels = []
invalid_labels = []
object_distribution = defaultdict(int)

print(f"Đang kiểm tra dataset (Instance Segmentation) tại: {dataset_path}...\n")

for split in splits:
    img_dir = os.path.join(dataset_path, split, "images")
    lbl_dir = os.path.join(dataset_path, split, "labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Cảnh báo: Không tìm thấy thư mục {split}")
        continue
        
    imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')}
    
    total_images += len(imgs)
    total_labels += len(lbls)
    
    missing_labels.extend([f"{split}/images/{i}" for i in imgs - lbls])
    missing_images.extend([f"{split}/labels/{l}" for l in lbls - imgs])
    
    # Check labels for segmentation (Polygon)
    for lbl_name in lbls:
        lbl_file = os.path.join(lbl_dir, lbl_name + ".txt")
        try:
            with open(lbl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            invalid_labels.append(f"{lbl_file} (Read error)")
            continue
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # YOLO polygon requires at least class_id + 3 points (6 coords) = 7 values. Length must be odd.
            if len(parts) < 7 or len(parts) % 2 == 0:
                invalid_labels.append(f"{lbl_file} (line {idx+1}: không đủ số điểm Polygon hoặc format sai)")
                continue
                
            try:
                cls_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                if cls_id < 0 or cls_id >= nc:
                    invalid_labels.append(f"{lbl_file} (line {idx+1}: class {cls_id} out of bounds)")
                elif any(c < 0 or c > 1 for c in coords):
                    invalid_labels.append(f"{lbl_file} (line {idx+1}: có toạ độ Polygon nằm ngoài [0,1])")
                else:
                    object_distribution[cls_id] += 1
            except ValueError:
                invalid_labels.append(f"{lbl_file} (line {idx+1}: invalid number format)")

print("="*40)
print("--- KẾT QUẢ KIỂM TRA DATASET ---")
print("="*40)
print(f"Tổng số ảnh (total_images): {total_images}")
print(f"Tổng số nhãn (total_labels): {total_labels}")
print(f"Ảnh thiếu nhãn (missing_labels): {len(missing_labels)}")
print(f"Nhãn thiếu ảnh (missing_images): {len(missing_images)}")
print(f"Nhãn bị lỗi định dạng/toạ độ (invalid_labels): {len(invalid_labels)}")

print("\n--- Object Distribution (Phân bổ số lượng vật thể) ---")
for cls_id in range(nc):
    count = object_distribution.get(cls_id, 0)
    print(f"Class {cls_id:2d} ({classes[cls_id]:<20}): {count:5d} vật thể")

print("\n--- Split Distribution (Số lượng ảnh theo tập) ---")
for split in splits:
    img_dir = os.path.join(dataset_path, split, "images")
    if os.path.exists(img_dir):
        split_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"{split.upper():<5}: {split_imgs} ảnh")

if invalid_labels:
    print("\n[CẢNH BÁO] Top 5 nhãn bị lỗi:")
    for lbl in invalid_labels[:5]:
        print(f"  - {lbl}")
