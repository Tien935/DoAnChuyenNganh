
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Đường dẫn dữ liệu
YOLO_CSV = Path(r"D:\DACN\week4\week4_yolo_vlm_results.csv")
VLM_CSV = Path(r"D:\DACN\week6\vlm_batch_results.csv")
OUTPUT_CSV = Path(r"D:\DACN\week6\week6_comparison.csv")

yolo = pd.read_csv(YOLO_CSV)
vlm = pd.read_csv(VLM_CSV)

# Chuẩn hóa tên ảnh để ghép dữ liệu
yolo["image_name"] = yolo["crop_name"].astype(str).str.strip()
vlm["image_name"] = vlm["image_name"].astype(str).str.strip()

# Chỉ lấy các cột cần thiết
yolo = yolo[["image_name", "yolo_class"]].drop_duplicates()
vlm = vlm[
    ["image_name", "top_class", "top_score", "status"]
].copy()

# Kiểm tra tên ảnh có bị trùng trước khi ghép
print("Số ảnh Week 4:", len(yolo))
print("Số ảnh Week 6:", len(vlm))
print("Tên ảnh trùng trong Week 4:", yolo["image_name"].duplicated().sum())
print("Tên ảnh trùng trong Week 6:", vlm["image_name"].duplicated().sum())

# Ghép theo tên ảnh
df = vlm.merge(yolo, on="image_name", how="inner", validate="many_to_one")

if df.empty:
    raise ValueError("Không ghép được ảnh. Hãy kiểm tra tên ảnh ở hai file.")

# So sánh dự đoán OpenCLIP với dự đoán YOLO
df["match_yolo"] = df["top_class"] == df["yolo_class"]

print("\n===== TỔNG KẾT =====")
print("Số ảnh ghép được:", len(df))
print("Số dự đoán trùng YOLO:", int(df["match_yolo"].sum()))
print("Tỷ lệ trùng YOLO (%):", round(df["match_yolo"].mean() * 100, 2))

print("\n===== TRẠNG THÁI OPENCLIP =====")
print(df["status"].value_counts(dropna=False))

print("\n===== TỶ LỆ TRÙNG THEO LỚP YOLO =====")
print(
    df.groupby("yolo_class")["match_yolo"]
      .agg(["count", "sum", "mean"])
      .assign(agreement_percent=lambda x: x["mean"] * 100)
      .drop(columns="mean")
      .round(2)
)

classes = sorted(set(df["yolo_class"]) | set(df["top_class"]))

print("\n===== BÁO CÁO SO SÁNH VỚI YOLO =====")
print(classification_report(
    df["yolo_class"],
    df["top_class"],
    labels=classes,
    zero_division=0
))

print("\n===== MA TRẬN SO SÁNH =====")
print(pd.DataFrame(
    confusion_matrix(df["yolo_class"], df["top_class"], labels=classes),
    index=classes,
    columns=classes
))

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("\nĐã lưu kết quả:", OUTPUT_CSV)
