
from pathlib import Path
import pandas as pd

# Các file đầu vào: chỉ đọc, không sửa
YOLO_CSV = Path(r"D:\DACN\week4\week4_crop_results.csv")
VLM_CSV = Path(r"D:\DACN\week6\vlm_batch_results.csv")

# File đầu ra mới
OUTPUT_CSV = Path(r"D:\DACN\week6\week6_yolo_vlm_confidence.csv")

# Đọc dữ liệu
yolo = pd.read_csv(YOLO_CSV)
vlm = pd.read_csv(VLM_CSV)

# Lấy tên crop từ đường dẫn YOLO
yolo["crop_name"] = yolo["crop_path"].apply(
    lambda path: Path(str(path)).name
)

# Kiểm tra các cột bắt buộc
required_yolo = {
    "crop_name", "yolo_class", "yolo_confidence", "crop_path"
}
required_vlm = {
    "image_name", "top_class", "top_score", "status"
}

if not required_yolo.issubset(yolo.columns):
    raise ValueError(
        f"Thiếu cột YOLO: {required_yolo - set(yolo.columns)}"
    )

if not required_vlm.issubset(vlm.columns):
    raise ValueError(
        f"Thiếu cột VLM: {required_vlm - set(vlm.columns)}"
    )

# Kiểm tra tên crop bị trùng trong CSV VLM
duplicate_vlm = vlm["image_name"].duplicated().sum()

if duplicate_vlm > 0:
    raise ValueError(
        f"CSV VLM có {duplicate_vlm} tên crop trùng. "
        "Cần kiểm tra trước khi ghép."
    )

# Chỉ giữ các cột cần thiết của VLM
vlm_selected = vlm[
    [
        "image_name",
        "predicted_class",
        "top_class",
        "top_score",
        "threshold",
        "status",
        "score_cardboard",
        "score_glass",
        "score_metal",
        "score_paper",
        "score_plastic",
        "score_trash",
    ]
].copy()

# Ghép bằng tên crop
merged = yolo.merge(
    vlm_selected,
    left_on="crop_name",
    right_on="image_name",
    how="left",
    validate="many_to_one",
    indicator=True,
)

# Đánh dấu trạng thái khớp lớp
merged["class_agreement"] = (
    merged["yolo_class"].astype(str).str.lower()
    == merged["top_class"].astype(str).str.lower()
)

# Đếm số dòng đã và chưa khớp dữ liệu VLM
matched = merged["_merge"] == "both"
unmatched = ~matched

print("Số dòng YOLO:", len(yolo))
print("Số dòng VLM:", len(vlm))
print("Tên crop YOLO trùng:", yolo["crop_name"].duplicated().sum())
print("Dòng ghép được:", int(matched.sum()))
print("Dòng không tìm thấy VLM:", int(unmatched.sum()))

if matched.any():
    print(
        "Hai mô hình cùng lớp:",
        int(merged.loc[matched, "class_agreement"].sum())
    )
    print(
        "Hai mô hình khác lớp:",
        int((~merged.loc[matched, "class_agreement"]).sum())
    )

# Bỏ cột kỹ thuật sau khi kiểm tra
merged.drop(columns=["_merge"], inplace=True)

# Chỉ ghi file đầu ra mới
merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\nĐã lưu file mới tại:")
print(OUTPUT_CSV)
