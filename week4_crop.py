from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# ==================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==================================================

PROJECT_DIR = Path(r"D:\DACN")

# Model YOLO đã train
MODEL_PATH = PROJECT_DIR / "best.pt"

# Ảnh đầu vào để test
INPUT_IMAGE = PROJECT_DIR / "images" / "plastic"

# Thư mục lưu crop
CROP_DIR = PROJECT_DIR / "week4" / "crops"

# Thư mục lưu ảnh có Bounding Box
ANNOTATED_DIR = PROJECT_DIR / "week4" / "annotated"


# ==================================================
# 2. TẠO THƯ MỤC
# ==================================================

CROP_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. LOAD MODEL
# ==================================================

print("=" * 60)
print("LOAD YOLO MODEL")
print("=" * 60)

model = YOLO(str(MODEL_PATH))

print("Model:", MODEL_PATH)
print("Classes:", model.names)


# ==================================================
# 4. TÌM ẢNH
# ==================================================

image_extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

image_files = []

if INPUT_IMAGE.is_file():

    image_files.append(INPUT_IMAGE)

else:

    for ext in image_extensions:
        image_files.extend(INPUT_IMAGE.glob(f"*{ext}"))

print()
print("Số ảnh tìm thấy:", len(image_files))


if len(image_files) == 0:

    print("KHÔNG TÌM THẤY ẢNH!")

    print("Kiểm tra thư mục:")
    print(INPUT_IMAGE)

    exit()


# ==================================================
# 5. CHỈ TEST ẢNH ĐẦU TIÊN
# ==================================================

image_path = image_files[0]

print()
print("=" * 60)
print("ẢNH ĐANG TEST")
print("=" * 60)

print(image_path)


# ==================================================
# 6. YOLO DETECTION
# ==================================================

results = model.predict(
    source=str(image_path),
    conf=0.25,
    imgsz=640,
    save=False,
    verbose=False
)


result = results[0]


# ==================================================
# 7. KIỂM TRA BOUNDING BOX
# ==================================================

print()
print("=" * 60)
print("KẾT QUẢ YOLO")
print("=" * 60)

print("Số bounding box:", len(result.boxes))


if len(result.boxes) == 0:

    print()
    print("YOLO KHÔNG PHÁT HIỆN ĐỐI TƯỢNG.")

else:

    # Đọc ảnh gốc
    image = Image.open(image_path).convert("RGB")

    image_width, image_height = image.size

    print("Kích thước ảnh:", image_width, "x", image_height)


    # ==================================================
    # 8. DUYỆT TỪNG BOUNDING BOX
    # ==================================================

    for i, box in enumerate(result.boxes):

        # Class ID
        class_id = int(box.cls[0])

        # Tên class
        class_name = model.names[class_id]

        # Confidence
        confidence = float(box.conf[0])

        # Tọa độ Bounding Box
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Chuyển sang số nguyên
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # ==================================================
        # 9. GIỚI HẠN TỌA ĐỘ TRONG ẢNH
        # ==================================================

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(image_width, x2)
        y2 = min(image_height, y2)


        # ==================================================
        # 10. IN THÔNG TIN
        # ==================================================

        print()
        print("-" * 50)

        print("Đối tượng:", i + 1)
        print("Class ID:", class_id)
        print("Class:", class_name)
        print("Confidence:", round(confidence, 4))

        print(
            "Bounding Box:",
            f"({x1}, {y1}) -> ({x2}, {y2})"
        )


        # ==================================================
        # 11. CROP ẢNH
        # ==================================================

        crop = image.crop(
            (x1, y1, x2, y2)
        )


        # ==================================================
        # 12. TẠO TÊN FILE CROP
        # ==================================================

        original_name = image_path.stem

        crop_name = (
            f"{class_name}_"
            f"{original_name}_"
            f"crop_{i + 1:03d}.jpg"
        )

        crop_path = CROP_DIR / crop_name


        # ==================================================
        # 13. LƯU CROP
        # ==================================================

        crop.save(crop_path)

        print("Crop saved:", crop_path)


# ==================================================
# 14. LƯU ẢNH BOUNDING BOX
# ==================================================

annotated_image = result.plot()

annotated_name = (
    f"{image_path.stem}_annotated.jpg"
)

annotated_path = ANNOTATED_DIR / annotated_name

Image.fromarray(
    annotated_image[..., ::-1]
).save(annotated_path)


print()
print("=" * 60)
print("HOÀN THÀNH")
print("=" * 60)

print("Crop folder:")
print(CROP_DIR)

print()
print("Annotated folder:")
print(annotated_path)