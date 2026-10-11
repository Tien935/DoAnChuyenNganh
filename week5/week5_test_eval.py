
from pathlib import Path

import torch
import open_clip
import pandas as pd
from PIL import Image


# ==========================================
# 1. CAU HINH
# ==========================================

DATASET_DIR = Path(r"D:\DACN\Waste_Detection_v4_YOLO_BBox")
IMAGE_DIR = DATASET_DIR / "test" / "images"
LABEL_DIR = DATASET_DIR / "test" / "labels"

OUTPUT_DIR = Path(r"D:\DACN\week5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Chon tu ket qua valid, khong dieu chinh theo test
THRESHOLD = 0.60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. PROMPT NGAN
# ==========================================

PROMPTS = {
    "cardboard": ["cardboard"],
    "glass": ["glass"],
    "metal": ["metal"],
    "paper": ["paper"],
    "plastic": ["plastic"],
    "trash": ["trash"],
}


# ==========================================
# 3. KIEM TRA DUONG DAN
# ==========================================

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Khong tim thay thu muc anh test: {IMAGE_DIR}\n"
        "Hay sua DATASET_DIR cho dung voi noi ban giai nen ZIP."
    )

if not LABEL_DIR.is_dir():
    raise FileNotFoundError(
        f"Khong tim thay thu muc nhan test: {LABEL_DIR}"
    )

print("Device:", DEVICE)
print("Image directory:", IMAGE_DIR)
print("Label directory:", LABEL_DIR)
print("Unknown threshold:", THRESHOLD)


# ==========================================
# 4. KHOI TAO OPENCLIP
# ==========================================

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai",
)

tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

model = model.to(DEVICE)
model.eval()


# ==========================================
# 5. TAO TEXT FEATURES
# ==========================================

text_features_list = []

with torch.no_grad():
    for class_name in CLASSES:
        tokens = tokenizer(PROMPTS[class_name]).to(DEVICE)

        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

        # Moi lop hien chi co mot prompt
        class_feature = features.mean(dim=0)
        class_feature = class_feature / class_feature.norm()

        text_features_list.append(class_feature)

    text_features = torch.stack(text_features_list, dim=0)


# ==========================================
# 6. DOC NHAN GROUND TRUTH VA BBOX
# ==========================================

def load_objects(image_path):
    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if not label_path.is_file():
        print(f"CANH BAO: Thieu file nhan {label_path.name}")
        return None

    with Image.open(image_path) as image:
        width, height = image.size

    objects = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.strip().split()

            if not values:
                continue

            if len(values) < 5:
                print(
                    f"CANH BAO: Dong nhan sai dinh dang: "
                    f"{label_path.name}, dong {line_number}"
                )
                continue

            try:
                class_id = int(values[0])
                xc, yc, bw, bh = map(float, values[1:5])
            except ValueError:
                print(
                    f"CANH BAO: Khong doc duoc nhan "
                    f"{label_path.name}, dong {line_number}"
                )
                continue

            if class_id < 0 or class_id >= len(CLASSES):
                print(
                    f"CANH BAO: Ma lop khong hop le "
                    f"{label_path.name}, dong {line_number}"
                )
                continue

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))

            objects.append({
                "true_label": CLASSES[class_id],
                "bbox": (x1, y1, x2, y2),
            })

    return objects


# ==========================================
# 7. LAY DANH SACH ANH TEST
# ==========================================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in extensions
)

if not image_paths:
    raise RuntimeError(f"Khong tim thay anh trong {IMAGE_DIR}")

print("So anh test tim thay:", len(image_paths))


# ==========================================
# 8. DANH GIA OPENCLIP
# ==========================================

records = []
missing_label_count = 0

for image_path in image_paths:
    objects = load_objects(image_path)

    if objects is None:
        missing_label_count += 1
        continue

    with Image.open(image_path) as source:
        source = source.convert("RGB")

        for object_index, obj in enumerate(objects):
            x1, y1, x2, y2 = obj["bbox"]
            crop = source.crop((x1, y1, x2, y2))

            input_tensor = preprocess(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(input_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                logits = 100.0 * image_features @ text_features.T
                probabilities = logits.softmax(dim=-1)[0]

            pred_index = int(probabilities.argmax().item())
            raw_prediction = CLASSES[pred_index]
            confidence = float(probabilities[pred_index].item())

            # Neu duoi nguong thi tu choi gan nhan
            if confidence < THRESHOLD:
                final_prediction = "unknown"
                accepted = False
            else:
                final_prediction = raw_prediction
                accepted = True

            true_label = obj["true_label"]

            # Unknown duoc tinh la khong phan loai dung
            correct_after_threshold = (
                accepted and final_prediction == true_label
            )

            records.append({
                "image_name": image_path.name,
                "object_index": object_index,
                "true_label": true_label,
                "raw_prediction": raw_prediction,
                "confidence": confidence,
                "threshold": THRESHOLD,
                "final_prediction": final_prediction,
                "accepted": accepted,
                "correct_after_threshold": correct_after_threshold,
            })

    print(f"Da xu ly: {image_path.name}")


# ==========================================
# 9. LUU KET QUA CHI TIET
# ==========================================

if not records:
    raise RuntimeError(
        "Khong co doi tuong nao duoc danh gia. "
        "Hay kiem tra file labels va duong dan."
    )

df = pd.DataFrame(records)

detail_path = OUTPUT_DIR / "week5_test_details.csv"
df.to_csv(detail_path, index=False, encoding="utf-8-sig")


# ==========================================
# 10. TONG HOP KET QUA
# ==========================================

total = len(df)
accepted_df = df[df["accepted"]]
accepted_count = len(accepted_df)

correct_raw = (df["raw_prediction"] == df["true_label"]).sum()
correct_accepted = (
    accepted_df["final_prediction"] == accepted_df["true_label"]
).sum()

# Do chinh xac truoc khi ap dung unknown
raw_accuracy = correct_raw / total * 100

# Do chinh xac tren cac mau duoc chap nhan
accepted_accuracy = (
    correct_accepted / accepted_count * 100
    if accepted_count else 0.0
)

# Tinh unknown la khong phan loai dung
overall_accuracy = df["correct_after_threshold"].mean() * 100

coverage = accepted_count / total * 100

summary = pd.DataFrame([{
    "test_images_found": len(image_paths),
    "images_missing_labels": missing_label_count,
    "objects_evaluated": total,
    "threshold": THRESHOLD,
    "correct_before_threshold": int(correct_raw),
    "raw_accuracy_percent": raw_accuracy,
    "accepted_objects": accepted_count,
    "unknown_objects": total - accepted_count,
    "coverage_percent": coverage,
    "correct_accepted": int(correct_accepted),
    "accuracy_accepted_percent": accepted_accuracy,
    "overall_accuracy_with_unknown_wrong_percent": overall_accuracy,
}])

summary_path = OUTPUT_DIR / "week5_test_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


# ==========================================
# 11. KET QUA THEO TUNG LOP
# ==========================================

per_class_records = []

for class_name in CLASSES:
    part = df[df["true_label"] == class_name]
    part_accepted = part[part["accepted"]]

    count = len(part)
    accepted = len(part_accepted)

    correct_raw_class = (
        part["raw_prediction"] == part["true_label"]
    ).sum()

    correct_accepted_class = (
        part_accepted["final_prediction"] == part_accepted["true_label"]
    ).sum()

    per_class_records.append({
        "true_label": class_name,
        "total": count,
        "correct_before_threshold": int(correct_raw_class),
        "raw_accuracy_percent": (
            correct_raw_class / count * 100 if count else None
        ),
        "accepted": accepted,
        "unknown": count - accepted,
        "coverage_percent": accepted / count * 100 if count else None,
        "correct_accepted": int(correct_accepted_class),
        "accuracy_accepted_percent": (
            correct_accepted_class / accepted * 100
            if accepted else None
        ),
    })

per_class_df = pd.DataFrame(per_class_records)

per_class_path = OUTPUT_DIR / "week5_test_per_class.csv"
per_class_df.to_csv(
    per_class_path, index=False, encoding="utf-8-sig"
)


# ==========================================
# 12. IN KET QUA
# ==========================================

print("\n========== TEST SUMMARY ==========")
print(summary.to_string(index=False))

print("\n========== PER CLASS ==========")
print(per_class_df.to_string(index=False))

print("\nDa luu cac file:")
print(detail_path)
print(summary_path)
print(per_class_path)

print("\nHoan thanh danh gia OpenCLIP tren TEST.")
