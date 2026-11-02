"""Evaluate trained waste weights, keeping box and mask metrics separate."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "local_runs/config"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def class_names(data):
    names = data["names"]
    return [names[index] for index in sorted(names)] if isinstance(names, dict) else names


def check_names(model_names, dataset_names):
    actual = [model_names[i] for i in range(len(model_names))]
    if actual != dataset_names:
        raise ValueError(f"Checkpoint có {len(actual)} lớp và không khớp tên/thứ tự của {len(dataset_names)} lớp dataset. Cần best.pt đã huấn luyện của nhóm; không đánh giá trọng số COCO trên bộ rác.")


def metric_summary(metric):
    p, r = float(metric.mp), float(metric.mr)
    return {"precision": p, "recall": r, "f1_from_mean_precision_recall": 2*p*r/(p+r) if p+r else 0,
            "map50": float(metric.map50), "map50_95": float(metric.map)}


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "config/waste.yaml")
    parser.add_argument("--device", default="cpu", choices=["cpu", "0"])
    parser.add_argument("--imgsz", type=int, default=320, choices=[320, 480, 640])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--name", default="waste_test")
    args = parser.parse_args(argv)
    if not args.weights.is_file():
        parser.error(f"Chưa có {args.weights}. Chép checkpoint đã huấn luyện vào models/best.pt.")
    data_file = args.data.resolve()
    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    names = class_names(data)
    base = (data_file.parent / data.get("path", ".")).resolve()
    for split in ("train", "val", "test"):
        if split in data:
            data[split] = str((base / data[split]).resolve())
    data.pop("path", None)
    test_path = Path(data["test"])
    if not test_path.is_dir():
        parser.error(f"Không tìm thấy tập test: {test_path}")
    from ultralytics import YOLO
    model = YOLO(str(args.weights.resolve()))
    check_names(model.names, names)
    labels = test_path.parent / "labels"
    if not labels.is_dir():
        parser.error(f"Không tìm thấy nhãn: {labels}")
    malformed = []
    rectangular = total = 0
    for label in sorted(labels.glob("*.txt")):
        for number, line in enumerate(label.read_text().splitlines(), 1):
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            total += 1
            cls, coords = values[0], values[1:]
            if cls != int(cls) or not 0 <= cls < len(names) or any(not 0 <= v <= 1 for v in coords):
                malformed.append(f"{label.name}:{number}: class/coordinate")
            if model.task == "segment" and (len(values) < 7 or len(values) % 2 == 0):
                malformed.append(f"{label.name}:{number}: không phải polygon")
            if len(coords) == 8:
                points = list(zip(coords[::2], coords[1::2]))
                if len({p[0] for p in points}) == 2 and len({p[1] for p in points}) == 2:
                    rectangular += 1
    if malformed:
        parser.error(f"Có {len(malformed)} lỗi nhãn. Không tự đổi nhãn test; xác nhận với STT 2. Ví dụ: {malformed[:4]}")
    output = ROOT / "local_runs/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_data.yaml"
    resolved.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    metrics = model.val(data=str(resolved), split="test", imgsz=args.imgsz, batch=args.batch,
                        device=args.device, workers=0, plots=True, project=str(output), name=args.name)
    result = {"weights": str(args.weights.resolve()), "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "data": str(data_file), "split": "test", "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
              "classes": names, "labels": total, "rectangular_polygons": rectangular,
              "annotation_note": "Polygon hình chữ nhật có thể được chuyển từ bbox; không chứng minh có mask đường viền chính xác.",
              "box": metric_summary(metrics.box), "speed_ms": metrics.speed}
    if hasattr(metrics, "seg"):
        result["mask"] = metric_summary(metrics.seg)
    target = Path(metrics.save_dir)
    (target / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (target / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "class_id", "class_name", "precision", "recall", "f1", "map50", "map50_95"])
        for task, metric in [("box", metrics.box)] + ([("mask", metrics.seg)] if hasattr(metrics, "seg") else []):
            for row, cls in enumerate(metric.ap_class_index):
                p, r, ap50, ap = metric.class_result(row)
                writer.writerow([task, int(cls), names[int(cls)], p, r, 2*p*r/(p+r) if p+r else 0, ap50, ap])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Confusion matrix, đường PR/F1 và bảng kết quả: {target}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        sys.exit(1)
