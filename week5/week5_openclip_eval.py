
from pathlib import Path
from collections import defaultdict

import torch
import open_clip
import pandas as pd
from PIL import Image


# =========================
# 1. CAU HINH DU LIEU
# =========================

DATASET_DIR = Path(r"D:\DACN\Waste_Detection_v4_YOLO_BBox")
OUTPUT_DIR = Path(r"D:\DACN\week5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = DATASET_DIR / "valid" / "images"
LABEL_DIR = DATASET_DIR / "valid" / "labels"

CLASSES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. CAC BO PROMPT
# =========================

SHORT_PROMPTS = {
    "cardboard": ["cardboard"],
    "glass": ["glass"],
    "metal": ["metal"],
    "paper": ["paper"],
    "plastic": ["plastic"],
    "trash": ["trash"],
}

DESCRIPTIVE_PROMPTS = {
    "cardboard": [
        "a photo of cardboard waste",
        "a photo of a discarded cardboard box",
        "an image of recyclable cardboard material",
    ],
    "glass": [
        "a photo of glass waste",
        "a photo of a discarded glass bottle",
        "an image of recyclable glass material",
    ],
    "metal": [
        "a photo of metal waste",
        "a photo of a discarded metal can",
        "an image of recyclable metal material",
    ],
    "paper": [
        "a photo of paper waste",
        "a photo of discarded paper sheets",
        "an image of recyclable paper material",
    ],
    "plastic": [
        "a photo of plastic waste",
        "a photo of a discarded plastic bottle",
        "an image of recyclable plastic material",
    ],
    "trash": [
        "a photo of general waste",
        "a photo of mixed discarded garbage",
        "an image of miscellaneous waste",
    ],
}

ENSEMBLE_PROMPTS = {
    "cardboard": [
        "cardboard",
        "a photo of cardboard waste",
        "a photo of a cardboard box",
        "an image of discarded cardboard",
        "cardboard recyclable material",
    ],
    "glass": [
        "glass",
        "a photo of glass waste",
        "a photo of a glass bottle",
        "an image of discarded glass",
        "glass recyclable material",
    ],
    "metal": [
        "metal",
        "a photo of metal waste",
        "a photo of a metal can",
        "an image of discarded metal",
        "metal recyclable material",
    ],
    "paper": [
        "paper",
        "a photo of paper waste",
        "a photo of discarded paper",
        "an image of paper waste",
        "paper recyclable material",
    ],
    "plastic": [
        "plastic",
        "a photo of plastic waste",
        "a photo of a plastic bottle",
        "an image of discarded plastic",
        "plastic recyclable material",
    ],
    "trash": [
        "trash",
        "a photo of general waste",
        "an image of discarded trash",
        "a piece of miscellaneous waste",
        "a pile of general garbage",
    ],
}

PROMPT_SETS = {
    "short": SHORT_PROMPTS,
    "descriptive": DESCRIPTIVE_PROMPTS,
    "ensemble": ENSEMBLE_PROMPTS,
}


# =========================
# 3. KHOI TAO OPENCLIP
# =========================

print("Device:", DEVICE)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai",
)
tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

model = model.to(DEVICE)
model.eval()


# =========================
# 4. TAO TEXT FEATURES
# =========================

def create_text_features(prompt_dict):
    class_features = []

    with torch.no_grad():
        for class_name in CLASSES:
            prompts = prompt_dict[class_name]
            tokens = tokenizer(prompts).to(DEVICE)

            features = model.encode_text(tokens)
            features = features / features.norm(
                dim=-1, keepdim=True
            )

            # Trung binh feature cua cac prompt cung lop
            class_feature = features.mean(dim=0)
            class_feature = class_feature / class_feature.norm()

            class_features.append(class_feature)

    return torch.stack(class_features, dim=0)


TEXT_FEATURES = {
    name: create_text_features(prompts)
    for name, prompts in PROMPT_SETS.items()
}


# =========================
# 5. DOC NHAN YOLO GOC
# =========================

def load_ground_truth(image_path):
    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if not label_path.exists():
        return []

    with open(label_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    objects = []

    with Image.open(image_path) as image:
        width, height = image.size

    for line in lines:
        values = line.strip().split()

        if len(values) < 5:
            continue

        class_id = int(values[0])
        x_center, y_center, box_width, box_height = map(
            float, values[1:5]
        )

        if class_id < 0 or class_id >= len(CLASSES):
            continue

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        objects.append({
            "true_label": CLASSES[class_id],
            "bbox": (x1, y1, x2, y2),
        })

    return objects


# =========================
# 6. CAT VUNG DOI TUONG
# =========================

records = []

image_paths = sorted(
    list(IMAGE_DIR.glob("*.jpg"))
    + list(IMAGE_DIR.glob("*.jpeg"))
    + list(IMAGE_DIR.glob("*.png"))
)

print("So anh valid:", len(image_paths))

for image_path in image_paths:
    objects = load_ground_truth(image_path)

    if not objects:
        print("Bo qua anh khong co nhan:", image_path.name)
        continue

    with Image.open(image_path) as source:
        source = source.convert("RGB")

        for object_index, obj in enumerate(objects):
            crop = source.crop(obj["bbox"])

            input_tensor = preprocess(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(input_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                for prompt_name, text_features in TEXT_FEATURES.items():
                    # So sanh image voi feature cua 6 lop
                    logits = 100.0 * image_features @ text_features.T
                    probabilities = logits.softmax(dim=-1)[0]

                    pred_index = int(probabilities.argmax().item())
                    pred_label = CLASSES[pred_index]
                    confidence = float(probabilities[pred_index].item())

                    records.append({
                        "image_name": image_path.name,
                        "object_index": object_index,
                        "prompt_type": prompt_name,
                        "true_label": obj["true_label"],
                        "predicted_label": pred_label,
                        "confidence": confidence,
                    })

    print("Da xu ly:", image_path.name)


# =========================
# 7. LUU KET QUA
# =========================

if not records:
    raise RuntimeError(
        "Khong co crop nao duoc danh gia. "
        "Hay kiem tra lai DATASET_DIR va thu muc valid."
    )

df = pd.DataFrame(records)

detail_path = OUTPUT_DIR / "week5_valid_details.csv"
df.to_csv(detail_path, index=False, encoding="utf-8-sig")

# Do chinh xac cua tung cach prompt
summary = (
    df.assign(correct=df["true_label"] == df["predicted_label"])
      .groupby("prompt_type")
      .agg(
          total=("correct", "size"),
          correct=("correct", "sum"),
          accuracy=("correct", "mean"),
      )
      .reset_index()
)

summary["accuracy_percent"] = summary["accuracy"] * 100

summary_path = OUTPUT_DIR / "week5_prompt_comparison.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

# Ket qua theo tung lop
per_class = (
    df.assign(correct=df["true_label"] == df["predicted_label"])
      .groupby(["prompt_type", "true_label"])
      .agg(
          total=("correct", "size"),
          correct=("correct", "sum"),
          accuracy=("correct", "mean"),
      )
      .reset_index()
)

per_class["accuracy_percent"] = per_class["accuracy"] * 100
per_class_path = OUTPUT_DIR / "week5_per_class.csv"
per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")

# Thu nghiem nguong unknown tren cac confidence
threshold_rows = []

for prompt_name in PROMPT_SETS:
    part = df[df["prompt_type"] == prompt_name].copy()
    part["correct"] = part["true_label"] == part["predicted_label"]

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        accepted = part[part["confidence"] >= threshold]

        threshold_rows.append({
            "prompt_type": prompt_name,
            "threshold": threshold,
            "total_objects": len(part),
            "accepted_objects": len(accepted),
            "coverage_percent": (
                len(accepted) / len(part) * 100 if len(part) else 0
            ),
            "accuracy_accepted_percent": (
                accepted["correct"].mean() * 100
                if len(accepted) else None
            ),
        })

threshold_df = pd.DataFrame(threshold_rows)
threshold_path = OUTPUT_DIR / "week5_unknown_threshold.csv"
threshold_df.to_csv(
    threshold_path, index=False, encoding="utf-8-sig"
)

print("\n===== SO SANH PROMPT =====")
print(summary.to_string(index=False))

print("\nDa luu cac file:")
print(detail_path)
print(summary_path)
print(per_class_path)
print(threshold_path)
print("\nHoan thanh.")
