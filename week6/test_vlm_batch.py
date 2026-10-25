
from pathlib import Path
import csv
from vlm_inference import load_vlm

# Thu muc anh crop da tao tu Week 4
IMAGE_DIR = Path(r"D:\DACN\week4\crops")

# File ket qua
OUTPUT_CSV = Path(r"D:\DACN\week6\vlm_batch_results.csv")

# Nguong hien tai, giu nguyen de kiem thu
THRESHOLD = 0.50

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not IMAGE_DIR.is_dir():
        print(f"Khong tim thay thu muc anh: {IMAGE_DIR}")
        return

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print("Khong tim thay anh trong thu muc.")
        return

    print(f"So anh tim thay: {len(image_paths)}")
    print("Dang khoi tao OpenCLIP...")

    vlm = load_vlm(threshold=THRESHOLD)
    results = []

    for index, image_path in enumerate(image_paths, start=1):
        try:
            result = vlm.predict_vlm(image_path)
            scores = result["scores"]

            row = {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "predicted_class": result["predicted_class"],
                "top_class": result["top_class"],
                "top_score": result["top_score"],
                "threshold": result["threshold"],
                "status": result["status"],
            }

            for class_name, score in scores.items():
                row[f"score_{class_name}"] = score

            results.append(row)

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name} -> "
                f"{result['predicted_class']} "
                f"(top={result['top_score']:.4f})"
            )

        except Exception as error:
            print(f"Loi voi anh {image_path.name}: {error}")

    if not results:
        print("Khong co anh nao duoc xu ly thanh cong.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    unknown_count = sum(
        row["status"] == "UNKNOWN" for row in results
    )

    print("\n===== TONG KET =====")
    print(f"So anh xu ly thanh cong: {len(results)}")
    print(f"So anh UNKNOWN: {unknown_count}")
    print(f"So anh CONFIDENT: {len(results) - unknown_count}")
    print(f"File CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

