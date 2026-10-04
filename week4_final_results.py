from pathlib import Path
import pandas as pd


# =====================================================
# 1. ĐƯỜNG DẪN
# =====================================================

PROJECT_DIR = Path(r"D:\DACN")
WEEK4_DIR = PROJECT_DIR / "week4"

YOLO_CSV = WEEK4_DIR / "week4_crop_results.csv"
VLM_CSV = WEEK4_DIR / "week4_yolo_vlm_results.csv"

FINAL_CSV = WEEK4_DIR / "week4_final_results.csv"


# =====================================================
# 2. ĐỌC DỮ LIỆU
# =====================================================

print("Đang đọc dữ liệu...")

yolo_df = pd.read_csv(YOLO_CSV)
vlm_df = pd.read_csv(VLM_CSV)

print("Số dòng YOLO:", len(yolo_df))
print("Số dòng VLM :", len(vlm_df))


# =====================================================
# 3. CHUẨN HÓA ĐƯỜNG DẪN
# =====================================================

def normalize_path(path):
    return str(Path(str(path))).replace("\\", "/").lower()


yolo_df["crop_key"] = yolo_df["crop_path"].apply(normalize_path)
vlm_df["crop_key"] = vlm_df["crop_path"].apply(normalize_path)


# =====================================================
# 4. KIỂM TRA TRÙNG CROP
# =====================================================

print("\nKiểm tra crop trùng...")

print(
    "YOLO crop_key duy nhất:",
    yolo_df["crop_key"].nunique()
)

print(
    "VLM crop_key duy nhất:",
    vlm_df["crop_key"].nunique()
)


# =====================================================
# 5. LOẠI CÁC CROP YOLO BỊ TRÙNG
# =====================================================

print("\nLoại crop YOLO bị trùng...")

before_dedup = len(yolo_df)

yolo_df = yolo_df.drop_duplicates(
    subset=["crop_key"],
    keep="first"
).copy()

after_dedup = len(yolo_df)

print("YOLO trước khi loại trùng:", before_dedup)
print("YOLO sau khi loại trùng:", after_dedup)
print("Số dòng bị loại:", before_dedup - after_dedup)


# =====================================================
# 6. LẤY THÔNG TIN YOLO
# =====================================================

yolo_info = yolo_df[
    [
        "crop_key",
        "image_name",
        "dataset_class",
        "yolo_class_id",
        "yolo_class",
        "yolo_confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_path"
    ]
].copy()


# =====================================================
# 7. LẤY THÔNG TIN OPENCLIP
# =====================================================

vlm_info = vlm_df[
    [
        "crop_key",
        "vlm_class",
        "vlm_confidence",
        "vlm_status",
        "cardboard_prob",
        "glass_prob",
        "metal_prob",
        "paper_prob",
        "plastic_prob",
        "trash_prob"
    ]
].copy()


# =====================================================
# 8. GHÉP YOLO + OPENCLIP
# =====================================================

df = pd.merge(
    yolo_info,
    vlm_info,
    on="crop_key",
    how="inner",
    validate="one_to_one"
)

print("\nSố crop ghép được:", len(df))

print(
    "Số crop OpenCLIP không tìm thấy YOLO:",
    len(vlm_info) - len(df)
)

print(
    "Số crop YOLO không tìm thấy OpenCLIP:",
    len(yolo_info) - len(df)
)


# =====================================================
# 9. TÍNH MATCH / MISMATCH
# =====================================================

df["mapping_result"] = df.apply(
    lambda row:
        "MATCH"
        if row["yolo_class"] == row["vlm_class"]
        else "MISMATCH",
    axis=1
)

# =====================================================
# 9. SẮP XẾP CỘT
# =====================================================

columns = [
    "image_name",
    "dataset_class",
    "yolo_class_id",
    "yolo_class",
    "yolo_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "crop_path",
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

df = df[columns]


# =====================================================
# 10. LƯU KẾT QUẢ
# =====================================================

df.to_csv(
    FINAL_CSV,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 11. THỐNG KÊ
# =====================================================

total = len(df)

match_count = (
    df["mapping_result"] == "MATCH"
).sum()

mismatch_count = (
    df["mapping_result"] == "MISMATCH"
).sum()

match_rate = (
    match_count / total * 100
    if total > 0
    else 0
)


# =====================================================
# 12. IN KẾT QUẢ
# =====================================================

print("\n======================================")
print("KẾT QUẢ WEEK 4 - YOLO + OPENCLIP")
print("======================================")

print("Tổng crop được ghép:", total)
print("MATCH:", match_count)
print("MISMATCH:", mismatch_count)
print("MATCH rate:", round(match_rate, 2), "%")


print("\n--------------------------------------")
print("PHÂN BỐ NHÃN YOLO")
print("--------------------------------------")

print(
    df["yolo_class"].value_counts()
)


print("\n--------------------------------------")
print("PHÂN BỐ NHÃN OPENCLIP")
print("--------------------------------------")

print(
    df["vlm_class"].value_counts()
)


print("\n--------------------------------------")
print("MA TRẬN YOLO -> OPENCLIP")
print("--------------------------------------")

matrix = pd.crosstab(
    df["yolo_class"],
    df["vlm_class"]
)

print(matrix)


print("\n--------------------------------------")
print("MATCH THEO TỪNG LỚP")
print("--------------------------------------")

for cls in sorted(df["yolo_class"].dropna().unique()):

    class_df = df[
        df["yolo_class"] == cls
    ]

    class_total = len(class_df)

    class_match = (
        class_df["mapping_result"] == "MATCH"
    ).sum()

    class_rate = (
        class_match / class_total * 100
        if class_total > 0
        else 0
    )

    print(
        f"{cls}: "
        f"{class_match}/{class_total} "
        f"= {class_rate:.2f}%"
    )


print("\n--------------------------------------")
print("FILE KẾT QUẢ")
print("--------------------------------------")

print(FINAL_CSV)

print("\nHoàn thành!")