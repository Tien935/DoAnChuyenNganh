import argparse
import csv
import html
from datetime import datetime
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageOps
import imagehash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--threshold", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not 0 <= args.threshold <= 64:
        raise ValueError("Threshold phải nằm trong khoảng 0–64.")

    # Đúng tên thư mục trong dataset của bạn.
    folders = {
        "train": root / "train" / "images",
        "val": root / "valid" / "images",
        "test": root / "test" / "images",
    }

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = {}
    errors = []

    for split, folder in folders.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"Không tìm thấy: {folder}")

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
        if not files:
            raise RuntimeError(f"Không tìm thấy ảnh trong: {folder}")

        records[split] = []
        print(f"\nĐọc {split}: {len(files)} ảnh", flush=True)

        for index, path in enumerate(files, 1):
            try:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    fingerprint = imagehash.phash(image, hash_size=8)

                records[split].append(
                    (path, int(str(fingerprint), 16))
                )
            except Exception as exc:
                errors.append([split, str(path), str(exc)])

            if index % 500 == 0:
                print(f"  Đã xử lý {index}/{len(files)}", flush=True)

    candidates = []

    for first, second in combinations(folders, 2):
        count = 0
        print(f"\nSo sánh {first} với {second}...", flush=True)

        for path_a, hash_a in records[first]:
            for path_b, hash_b in records[second]:
                distance = (hash_a ^ hash_b).bit_count()

                if distance <= args.threshold:
                    candidates.append({
                        "split_a": first,
                        "split_b": second,
                        "distance": distance,
                        "image_a": str(path_a),
                        "image_b": str(path_b),
                        "review": "",
                    })
                    count += 1

        print(f"  Có {count} cặp nghi vấn.", flush=True)

    candidates.sort(key=lambda row: row["distance"])

    # Mỗi lần chạy tạo thư mục mới, không ghi đè báo cáo cũ.
    output = Path.cwd() / (
        "near_duplicate_review_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(exist_ok=False)

    with (output / "candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = [
            "split_a", "split_b", "distance",
            "image_a", "image_b", "review"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)

    with (output / "read_errors.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["split", "image", "error"])
        writer.writerows(errors)

    # Chia trang HTML, mỗi trang tối đa 100 cặp.
    chunks = [
        candidates[i:i + 100]
        for i in range(0, len(candidates), 100)
    ] or [[]]

    for page_number, chunk in enumerate(chunks, 1):
        blocks = [
            '<meta charset="utf-8">',
            "<title>Rà soát ảnh gần trùng</title>",
            "<style>"
            "body{font-family:Arial;margin:24px}"
            "section{border-bottom:1px solid #aaa;padding:16px 0}"
            "img{width:42%;height:280px;object-fit:contain}"
            "p{overflow-wrap:anywhere}"
            "</style>",
            "<h1>Các cặp nghi gần trùng — chưa xác nhận</h1>",
        ]

        for offset, row in enumerate(chunk, 1):
            pair_id = (page_number - 1) * 100 + offset
            a = html.escape(Path(row["image_a"]).as_uri(), quote=True)
            b = html.escape(Path(row["image_b"]).as_uri(), quote=True)

            blocks.append(
                f"<section><h3>Cặp {pair_id} | "
                f'{row["split_a"]} – {row["split_b"]} | '
                f'Khoảng cách: {row["distance"]}</h3>'
                f'<img src="{a}"> <img src="{b}">'
                f'<p>{html.escape(row["image_a"])}</p>'
                f'<p>{html.escape(row["image_b"])}</p></section>'
            )

        (output / f"review_{page_number:03d}.html").write_text(
            "\n".join(blocks), encoding="utf-8"
        )

    print(f"\nTổng số cặp nghi vấn: {len(candidates)}")
    print(f"Số ảnh đọc lỗi: {len(errors)}")
    print(f"Kết quả lưu tại: {output.resolve()}")
    print("Không có ảnh hoặc nhãn nào bị sửa, xóa hay di chuyển.")


if __name__ == "__main__":
    main()