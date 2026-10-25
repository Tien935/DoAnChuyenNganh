
from pathlib import Path
import pandas as pd

INPUT_CSV = Path(
    r"D:\DACN\week6\week6_yolo_vlm_confidence.csv"
)
OUTPUT_CSV = Path(
    r"D:\DACN\week6\confidence_fusion_experiment.csv"
)

df = pd.read_csv(INPUT_CSV)

required = [
    "yolo_class",
    "yolo_confidence",
    "top_class",
    "top_score",
    "status",
    "class_agreement",
]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Thiếu cột: {missing}")

# Cách 1: chỉ sử dụng YOLO
df["pred_yolo_only"] = df["yolo_class"]

# Cách 2: chỉ sử dụng OpenCLIP nếu không UNKNOWN
df["pred_vlm_only"] = df["top_class"].where(
    df["status"].eq("CONFIDENT"),
    "UNKNOWN",
)

# Cách 3: ưu tiên nhãn chung khi cả hai đồng ý
# Nếu bất đồng hoặc VLM UNKNOWN, tạm giữ nhãn YOLO.
agreement = (
    df["yolo_class"].astype(str).str.lower()
    == df["top_class"].astype(str).str.lower()
)
vlm_confident = df["status"].eq("CONFIDENT")

df["pred_agreement_first"] = df["yolo_class"]
df.loc[
    agreement & vlm_confident,
    "pred_agreement_first",
] = df.loc[
    agreement & vlm_confident,
    "top_class",
]

df["fusion_reason"] = "DISAGREEMENT_USE_YOLO"
df.loc[
    agreement & vlm_confident,
    "fusion_reason",
] = "AGREEMENT"
df.loc[
    ~vlm_confident,
    "fusion_reason",
] = "VLM_UNKNOWN_USE_YOLO"

# Đây là số liệu đồng thuận, không phải accuracy
print("Tổng số dòng:", len(df))
print("Số crop_name trùng:", df["crop_name"].duplicated().sum())
print("VLM CONFIDENT:", int(vlm_confident.sum()))
print("VLM UNKNOWN:", int((~vlm_confident).sum()))
print("Hai mô hình cùng lớp:", int(agreement.sum()))
print("Hai mô hình khác lớp:", int((~agreement).sum()))

print("\nLý do xử lý:")
print(df["fusion_reason"].value_counts())

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("\nĐã lưu file:", OUTPUT_CSV)
