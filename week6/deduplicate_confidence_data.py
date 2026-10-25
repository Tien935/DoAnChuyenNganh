
from pathlib import Path
import pandas as pd

INPUT_CSV = Path(
    r"D:\DACN\week6\week6_yolo_vlm_confidence.csv"
)
OUTPUT_CSV = Path(
    r"D:\DACN\week6\week6_yolo_vlm_confidence_unique.csv"
)

df = pd.read_csv(INPUT_CSV)

print("Số dòng ban đầu:", len(df))
print("Số tên crop duy nhất:", df["crop_name"].nunique())

# Kiểm tra các dòng trùng trên toàn bộ cột
duplicate_rows = df.duplicated(keep="first")
print("Số dòng trùng hoàn toàn:", int(duplicate_rows.sum()))

# Chỉ loại các dòng giống nhau trên toàn bộ cột
unique_df = df.drop_duplicates(keep="first").copy()

print("Số dòng sau khi loại trùng hoàn toàn:", len(unique_df))
print("Số tên crop duy nhất sau xử lý:", unique_df["crop_name"].nunique())

# Kiểm tra một tên crop có còn nhiều bản ghi khác nhau không
remaining_duplicate_names = unique_df["crop_name"].duplicated().sum()
print("Số dòng vẫn trùng tên crop:", int(remaining_duplicate_names))

unique_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã lưu file mới:")
print(OUTPUT_CSV)
