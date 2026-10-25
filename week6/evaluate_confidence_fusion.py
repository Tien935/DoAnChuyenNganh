
from pathlib import Path
import pandas as pd

INPUT_CSV = Path(
    r"D:\DACN\week6\week6_yolo_vlm_confidence.csv"
)
OUTPUT_CSV = Path(
    r"D:\DACN\week6\confidence_fusion_scores.csv"
)

df = pd.read_csv(INPUT_CSV)

required = [
    "crop_name",
    "yolo_class",
    "yolo_confidence",
    "top_class",
    "top_score",
    "status",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Thiếu các cột: {missing}")

# Chuyển các cột điểm sang kiểu số
df["yolo_confidence"] = pd.to_numeric(
    df["yolo_confidence"], errors="coerce"
)
df["top_score"] = pd.to_numeric(
    df["top_score"], errors="coerce"
)

# Chuẩn hóa lớp để so sánh
same_class = (
    df["yolo_class"].astype(str).str.strip().str.lower()
    == df["top_class"].astype(str).str.strip().str.lower()
)
vlm_confident = df["status"].eq("CONFIDENT")

# Chỉ kết hợp điểm khi hai mô hình đồng ý lớp
# và OpenCLIP không trả UNKNOWN.
valid_fusion = (
    same_class
    & vlm_confident
    & df["yolo_confidence"].notna()
    & df["top_score"].notna()
)

df["fusion_eligible"] = valid_fusion
df["fusion_note"] = "DISAGREEMENT: không kết hợp điểm"
df.loc[
    same_class & ~vlm_confident,
    "fusion_note",
] = "VLM_UNKNOWN: không kết hợp điểm"
df.loc[
    valid_fusion,
    "fusion_note",
] = "AGREEMENT_CONFIDENT: đủ điều kiện thử nghiệm"

# Tính điểm kết hợp với 3 trọng số khác nhau
alphas = [0.3, 0.5, 0.7]

for alpha in alphas:
    col = f"fusion_alpha_{alpha:.1f}"
    df[col] = pd.NA

    df.loc[valid_fusion, col] = (
        alpha * df.loc[valid_fusion, "yolo_confidence"]
        + (1 - alpha) * df.loc[valid_fusion, "top_score"]
    )

# In thống kê
print("Tổng số dòng:", len(df))
print("Số crop_name trùng:", df["crop_name"].duplicated().sum())
print("Hai mô hình đồng ý lớp:", int(same_class.sum()))
print("VLM CONFIDENT:", int(vlm_confident.sum()))
print("Đủ điều kiện kết hợp điểm:", int(valid_fusion.sum()))
print("Không kết hợp điểm:", int((~valid_fusion).sum()))

print("\nThống kê điểm kết hợp:")
for alpha in alphas:
    col = f"fusion_alpha_{alpha:.1f}"
    scores = df[col].dropna()

    print(f"\nAlpha = {alpha}")
    print("Số điểm:", len(scores))
    if len(scores) > 0:
        print(scores.describe().round(4).to_string())

# Lưu thành file mới, không ghi đè file đầu vào
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\nĐã lưu kết quả tại:")
print(OUTPUT_CSV)
