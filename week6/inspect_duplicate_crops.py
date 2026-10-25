
import pandas as pd
from pathlib import Path

csv_path = Path(r"D:\DACN\week6\week6_yolo_vlm_confidence.csv")
df = pd.read_csv(csv_path)

# Lấy một crop_name bị trùng để kiểm tra
duplicate_names = df.loc[
    df["crop_name"].duplicated(keep=False), "crop_name"
].unique()

if len(duplicate_names) == 0:
    print("Không tìm thấy crop_name bị trùng.")
else:
    crop_name = duplicate_names[0]
    group = df[df["crop_name"] == crop_name]

    print("CROP NAME:", crop_name)
    print("Số dòng:", len(group))
    print("\nCác cột có giá trị khác nhau giữa các dòng:")

    for col in df.columns:
        values = group[col].astype(str).unique()
        if len(values) > 1:
            print(f"\n{col}:")
            for value in values:
                print("  ", value)

    print("\nToàn bộ các cột của nhóm:")
    print(group.to_string(index=False))
