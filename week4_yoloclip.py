from pathlib import Path
import csv

import torch
import open_clip
from PIL import Image


# ============================================================
# 1. CẤU HÌNH
# ============================================================

PROJECT_DIR = Path(r"D:\DACN")

CROP_DIR = PROJECT_DIR / "week4" / "crops"

OUTPUT_CSV = PROJECT_DIR / "week4" / "week4_yolo_vlm_results.csv"


# ============================================================
# 2. DANH SÁCH CLASS
# ============================================================

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# ============================================================
# 3. PROMPT ENSEMBLE
# ============================================================

PROMPTS = {

    "cardboard": [
        "a photo of cardboard waste",
        "a photo of a cardboard box",
        "an image of discarded cardboard",
        "a piece of cardboard waste",
        "cardboard recyclable material"
    ],

    "glass": [
        "a photo of glass waste",
        "a photo of a glass bottle",
        "an image of discarded glass",
        "a piece of glass waste",
        "glass recyclable material"
    ],

    "metal": [
        "a photo of metal waste",
        "a photo of a metal can",
        "an image of discarded metal",
        "a piece of metal waste",
        "metal recyclable material"
    ],

    "paper": [
        "a photo of paper waste",
        "a photo of discarded paper",
        "an image of paper waste",
        "a piece of paper waste",
        "paper recyclable material"
    ],

    "plastic": [
        "a photo of plastic waste",
        "a photo of a plastic bottle",
        "an image of discarded plastic",
        "a piece of plastic waste",
        "plastic recyclable material"
    ],

    "trash": [
        "a photo of trash",
        "a photo of general waste",
        "an image of discarded trash",
        "a piece of miscellaneous waste",
        "a pile of general garbage"
    ]
}


# ============================================================
# 4. DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("OPENCLIP - TUẦN 4")
print("=" * 60)

print("Device:", device)


# ============================================================
# 5. LOAD OPENCLIP
# ============================================================

print()
print("Đang load OpenCLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer(
    "ViT-B-32-quickgelu"
)

model = model.to(device)

model.eval()

print("OpenCLIP loaded!")


# ============================================================
# 6. CHUẨN BỊ TEXT FEATURES
# ============================================================

print()
print("Đang tạo text features...")


# Mỗi class có 5 prompt
text_features_by_class = {}


with torch.no_grad():

    for class_name in CLASSES:

        prompts = PROMPTS[class_name]

        tokens = tokenizer(prompts).to(device)

        features = model.encode_text(tokens)

        # Chuẩn hóa từng prompt
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        # Trung bình 5 prompt
        class_feature = features.mean(dim=0)

        # Chuẩn hóa lại
        class_feature = class_feature / class_feature.norm()

        text_features_by_class[class_name] = class_feature


# Tạo tensor [6, embedding_dim]
text_features = torch.stack([
    text_features_by_class[class_name]
    for class_name in CLASSES
])


print("Đã tạo text features cho 6 class.")


# ============================================================
# 7. TÌM CÁC ẢNH CROP
# ============================================================

image_extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]


crop_files = []

for ext in image_extensions:

    crop_files.extend(
        CROP_DIR.glob(f"*{ext}")
    )


crop_files = sorted(crop_files)


print()
print("Số crop tìm thấy:", len(crop_files))


if len(crop_files) == 0:

    print()
    print("Không tìm thấy crop!")

    print("Kiểm tra:")
    print(CROP_DIR)

    exit()


# ============================================================
# 8. HÀM SUY LUẬN OPENCLIP
# ============================================================

def predict_vlm(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    image_input = preprocess(
        image
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        image_features = model.encode_image(
            image_input
        )

        # Normalize image
        image_features = (
            image_features /
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )


        # Cosine similarity
        similarities = (
            image_features @
            text_features.T
        )[0]


        # Chuyển sang xác suất
        probabilities = (
            similarities.softmax(dim=0)
        )


    # Tìm class cao nhất

    best_index = int(
        probabilities.argmax()
    )

    best_class = CLASSES[
        best_index
    ]

    best_confidence = float(
        probabilities[best_index]
    )


    # Lấy toàn bộ probability

    probability_dict = {}

    for i, class_name in enumerate(CLASSES):

        probability_dict[class_name] = float(
            probabilities[i]
        )


    return (
        best_class,
        best_confidence,
        probability_dict
    )


# ============================================================
# 9. PHÂN TÍCH TẤT CẢ CROP
# ============================================================

results = []


print()
print("=" * 60)
print("BẮT ĐẦU OPENCLIP")
print("=" * 60)


for index, crop_path in enumerate(
    crop_files,
    start=1
):

    print()
    print(
        f"[{index}/{len(crop_files)}]",
        crop_path.name
    )


    try:

        (
            vlm_class,
            vlm_confidence,
            probabilities
        ) = predict_vlm(
            crop_path
        )


        # ----------------------------------------------
        # Lấy YOLO class từ tên file
        # ----------------------------------------------

        filename_lower = crop_path.name.lower()

        yolo_class = "unknown"

        for class_name in CLASSES:

            if f"_{class_name}." in filename_lower:

                yolo_class = class_name

                break


            if f"_{class_name}_" in filename_lower:

                yolo_class = class_name

                break


        # ----------------------------------------------
        # So sánh YOLO và VLM
        # ----------------------------------------------

        if yolo_class == vlm_class:

            mapping_result = "MATCH"

        else:

            mapping_result = "MISMATCH"


        # ----------------------------------------------
        # Confidence status
        # ----------------------------------------------

        if vlm_confidence >= 0.50:

            vlm_status = "CONFIDENT"

        else:

            vlm_status = "UNCERTAIN"


        # ----------------------------------------------
        # In kết quả
        # ----------------------------------------------

        print(
            "YOLO:",
            yolo_class
        )

        print(
            "VLM:",
            vlm_class
        )

        print(
            "VLM confidence:",
            round(vlm_confidence, 4)
        )

        print(
            "Mapping:",
            mapping_result
        )


        # ----------------------------------------------
        # Lưu kết quả
        # ----------------------------------------------

        row = {

            "crop_name":
                crop_path.name,

            "crop_path":
                str(crop_path),

            "yolo_class":
                yolo_class,

            "vlm_class":
                vlm_class,

            "vlm_confidence":
                round(
                    vlm_confidence,
                    4
                ),

            "vlm_status":
                vlm_status,

            "mapping_result":
                mapping_result,

            "cardboard_prob":
                round(
                    probabilities["cardboard"],
                    4
                ),

            "glass_prob":
                round(
                    probabilities["glass"],
                    4
                ),

            "metal_prob":
                round(
                    probabilities["metal"],
                    4
                ),

            "paper_prob":
                round(
                    probabilities["paper"],
                    4
                ),

            "plastic_prob":
                round(
                    probabilities["plastic"],
                    4
                ),

            "trash_prob":
                round(
                    probabilities["trash"],
                    4
                )
        }


        results.append(row)


    except Exception as e:

        print(
            "Lỗi:",
            e
        )


# ============================================================
# 10. GHI CSV
# ============================================================

fieldnames = [

    "crop_name",
    "crop_path",

    "yolo_class",

    "vlm_class",

    "vlm_confidence",

    "vlm_status",

    "mapping_result",

    "cardboard_prob",
    "glass_prob",
    "metal_prob",
    "paper_prob",
    "plastic_prob",
    "trash_prob"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# 11. THỐNG KÊ
# ============================================================

match_count = sum(
    1
    for row in results
    if row["mapping_result"] == "MATCH"
)


mismatch_count = sum(
    1
    for row in results
    if row["mapping_result"] == "MISMATCH"
)


print()
print("=" * 60)
print("HOÀN THÀNH OPENCLIP")
print("=" * 60)

print("Tổng crop:", len(results))

print("MATCH:", match_count)

print("MISMATCH:", mismatch_count)

if len(results) > 0:

    mapping_accuracy = (
        match_count /
        len(results)
    ) * 100

    print(
        "Tỷ lệ MATCH:",
        round(mapping_accuracy, 2),
        "%"
    )


print()
print("File kết quả:")

print(OUTPUT_CSV)