from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import csv


# ==================================================
# 1. CẤU HÌNH
# ==================================================

PROJECT_DIR = Path(r"D:\DACN")

MODEL_PATH = PROJECT_DIR / "best.pt"

IMAGE_DIR = PROJECT_DIR / "images"

WEEK4_DIR = PROJECT_DIR / "week4"

CROP_DIR = WEEK4_DIR / "crops"

ANNOTATED_DIR = WEEK4_DIR / "annotated"

CSV_PATH = WEEK4_DIR / "week4_crop_results.csv"


# ==================================================
# 2. TẠO THƯ MỤC
# ==================================================

CROP_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. LOAD YOLO
# ==================================================

print("=" * 60)
print("LOAD YOLO MODEL")
print("=" * 60)

model = YOLO(str(MODEL_PATH))

print("Model:", MODEL_PATH)

print("Classes:", model.names)


# ==================================================
# 4. LẤY TOÀN BỘ ẢNH
# ==================================================

extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

image_files = []

for ext in extensions:

    image_files.extend(
        IMAGE_DIR.rglob(f"*{ext}")
    )


print()
print("Tổng số ảnh:", len(image_files))


if len(image_files) == 0:

    print("Không tìm thấy ảnh!")

    exit()


# ==================================================
# 5. TẠO CSV
# ==================================================

csv_fields = [
    "image_path",
    "image_name",
    "dataset_class",
    "detection_index",
    "yolo_class_id",
    "yolo_class",
    "yolo_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "crop_path"
]


csv_file = open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
)

writer = csv.DictWriter(
    csv_file,
    fieldnames=csv_fields
)

writer.writeheader()


# ==================================================
# 6. XỬ LÝ TỪNG ẢNH
# ==================================================

total_images = 0

total_boxes = 0


for image_index, image_path in enumerate(
    image_files,
    start=1
):

    print()
    print("=" * 60)

    print(
        f"[{image_index}/{len(image_files)}]",
        image_path.name
    )


    # ------------------------------------------------
    # Class thật theo thư mục
    # ------------------------------------------------

    dataset_class = image_path.parent.name


    # ------------------------------------------------
    # YOLO prediction
    # ------------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=640,
        save=False,
        verbose=False
    )


    result = results[0]

    total_images += 1


    # ------------------------------------------------
    # Nếu không phát hiện
    # ------------------------------------------------

    if len(result.boxes) == 0:

        print("Không phát hiện đối tượng.")

        continue


    # ------------------------------------------------
    # Đọc ảnh
    # ------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    width, height = image.size


    # ------------------------------------------------
    # Xử lý từng Bounding Box
    # ------------------------------------------------

    for detection_index, box in enumerate(
        result.boxes,
        start=1
    ):

        class_id = int(
            box.cls[0]
        )

        class_name = model.names[
            class_id
        ]

        confidence = float(
            box.conf[0]
        )


        # Bounding Box

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )


        x1 = int(max(0, x1))

        y1 = int(max(0, y1))

        x2 = int(min(width, x2))

        y2 = int(min(height, y2))


        # ------------------------------------------------
        # Crop
        # ------------------------------------------------

        crop = image.crop(
            (x1, y1, x2, y2)
        )


        # ------------------------------------------------
        # Tên file crop
        # ------------------------------------------------

        crop_name = (
            f"{image_path.stem}"
            f"_det_{detection_index:03d}"
            f"_{class_name}"
            f".jpg"
        )


        crop_path = (
            CROP_DIR / crop_name
        )


        crop.save(
            crop_path
        )


        # ------------------------------------------------
        # Ghi CSV
        # ------------------------------------------------

        writer.writerow({

            "image_path":
                str(image_path),

            "image_name":
                image_path.name,

            "dataset_class":
                dataset_class,

            "detection_index":
                detection_index,

            "yolo_class_id":
                class_id,

            "yolo_class":
                class_name,

            "yolo_confidence":
                round(
                    confidence,
                    4
                ),

            "x1": x1,

            "y1": y1,

            "x2": x2,

            "y2": y2,

            "crop_path":
                str(crop_path)
        })


        total_boxes += 1


        print(
            f"  {detection_index}. "
            f"{class_name} "
            f"conf={confidence:.4f}"
        )


# ==================================================
# 7. ĐÓNG CSV
# ==================================================

csv_file.close()


# ==================================================
# 8. TỔNG KẾT
# ==================================================

print()
print("=" * 60)

print("HOÀN THÀNH TUẦN 4 - PHẦN CROP")

print("=" * 60)

print("Tổng số ảnh:", total_images)

print("Tổng Bounding Box:", total_boxes)

print()

print("Thư mục crop:")

print(CROP_DIR)

print()

print("File kết quả CSV:")

print(CSV_PATH)