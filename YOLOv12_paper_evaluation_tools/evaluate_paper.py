#!/usr/bin/env python3
"""Evaluate YOLOv12 segmentation for a scientific paper.

The script chooses the operating confidence threshold on the validation split,
then evaluates the untouched test split. AP is computed from the full PR curve
using a low confidence floor; P/R/F1 are read at the validation-selected
operating point, so the test split is not used for threshold tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--output", default="paper_results", help="Output directory")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold")
    parser.add_argument(
        "--ap-conf-floor",
        type=float,
        default=0.001,
        help="Low confidence floor used to build PR curves and calculate AP",
    )
    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def class_names(metrics: Any, model: Any) -> dict[int, str]:
    names = getattr(metrics, "names", None) or getattr(model, "names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def curve_threshold(metric: Any) -> tuple[float, str]:
    px = to_numpy(getattr(metric, "px", None)).reshape(-1)
    f1_curve = to_numpy(getattr(metric, "f1_curve", None))
    if px.size and f1_curve.size:
        if f1_curve.ndim == 1:
            f1_curve = f1_curve[None, :]
        mean_f1 = np.nanmean(f1_curve, axis=0)
        index = int(np.nanargmax(mean_f1))
        return float(px[index]), "validation mask macro-F1 curve"
    return 0.25, "fallback 0.25 because this Ultralytics version did not expose F1 curves"


def curve_at_threshold(metric: Any, threshold: float) -> tuple[np.ndarray, np.ndarray, float]:
    px = to_numpy(getattr(metric, "px", None)).reshape(-1)
    p_curve = to_numpy(getattr(metric, "p_curve", None))
    r_curve = to_numpy(getattr(metric, "r_curve", None))
    if not px.size or not p_curve.size or not r_curve.size:
        p = to_numpy(getattr(metric, "p", None)).reshape(-1)
        r = to_numpy(getattr(metric, "r", None)).reshape(-1)
        return p, r, float("nan")
    if p_curve.ndim == 1:
        p_curve = p_curve[None, :]
    if r_curve.ndim == 1:
        r_curve = r_curve[None, :]
    index = int(np.argmin(np.abs(px - threshold)))
    return p_curve[:, index], r_curve[:, index], float(px[index])


def safe_mean(values: np.ndarray) -> float | None:
    values = to_numpy(values).astype(float).reshape(-1)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def metric_summary(metric: Any, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p, r, actual_threshold = curve_at_threshold(metric, threshold)
    ap50 = to_numpy(getattr(metric, "ap50", None)).reshape(-1)
    ap5095 = to_numpy(getattr(metric, "ap", None)).reshape(-1)
    class_ids = to_numpy(getattr(metric, "ap_class_index", None)).astype(int).reshape(-1)

    available = max(len(p), len(r), len(ap50), len(ap5095), len(class_ids))
    if not class_ids.size and available:
        class_ids = np.arange(available, dtype=int)

    rows: list[dict[str, Any]] = []
    for pos, class_id in enumerate(class_ids):
        precision = float(p[pos]) if pos < len(p) else None
        recall = float(r[pos]) if pos < len(r) else None
        f1 = None
        if precision is not None and recall is not None:
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
        rows.append(
            {
                "class_id": int(class_id),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50": float(ap50[pos]) if pos < len(ap50) else None,
                "ap50_95": float(ap5095[pos]) if pos < len(ap5095) else None,
            }
        )

    summary = {
        "precision_macro": safe_mean(p),
        "recall_macro": safe_mean(r),
        "f1_macro": safe_mean(np.asarray([row["f1"] for row in rows], dtype=float)) if rows else None,
        "map50": float(getattr(metric, "map50", np.nan)),
        "map50_95": float(getattr(metric, "map", np.nan)),
        "curve_threshold_used": None if np.isnan(actual_threshold) else actual_threshold,
    }
    return summary, rows


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def software_and_hardware(torch_module: Any, ultralytics_module: Any) -> dict[str, Any]:
    cuda_available = bool(torch_module.cuda.is_available())
    gpu = None
    if cuda_available:
        try:
            gpu = torch_module.cuda.get_device_name(0)
        except Exception:
            gpu = "CUDA device available"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch_module.__version__,
        "ultralytics": ultralytics_module.__version__,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "cudnn_version": torch_module.backends.cudnn.version() if cuda_available else None,
        "gpu": gpu,
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {data_path}")

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Thiếu thư viện. Hãy chạy: python -m pip install ultralytics pyyaml pillow"
        ) from exc

    model = YOLO(str(model_path))
    common = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "iou": args.iou,
        "conf": args.ap_conf_floor,
        "plots": True,
        "verbose": True,
        "project": str(output / "yolo_runs"),
    }

    print("\n[1/2] Hiệu chỉnh ngưỡng confidence trên VALIDATION...")
    val_metrics = model.val(split="val", name="validation_calibration", exist_ok=True, **common)
    if not hasattr(val_metrics, "seg"):
        raise RuntimeError("Model/kết quả không có metrics.seg; hãy kiểm tra đây có phải model segmentation không.")
    confidence, threshold_method = curve_threshold(val_metrics.seg)

    print(f"\nNgưỡng confidence đã chọn: {confidence:.4f} ({threshold_method})")
    print("\n[2/2] Đánh giá TEST một lần với PR curve đầy đủ...")
    test_metrics = model.val(split="test", name="test_final", exist_ok=True, **common)

    names = class_names(test_metrics, model)
    box_summary, box_rows = metric_summary(test_metrics.box, confidence)
    mask_summary, mask_rows = metric_summary(test_metrics.seg, confidence)
    box_by_id = {row["class_id"]: row for row in box_rows}
    mask_by_id = {row["class_id"]: row for row in mask_rows}

    per_class = []
    for class_id in sorted(set(box_by_id) | set(mask_by_id)):
        row = {"class_id": class_id, "class_name": names.get(class_id, str(class_id))}
        for prefix, source in (("box", box_by_id.get(class_id, {})), ("mask", mask_by_id.get(class_id, {}))):
            for key in ("precision", "recall", "f1", "ap50", "ap50_95"):
                row[f"{prefix}_{key}"] = source.get(key)
        per_class.append(row)

    speed = dict(getattr(test_metrics, "speed", {}) or {})
    inference_ms = speed.get("inference")
    fps = 1000.0 / inference_ms if inference_ms and inference_ms > 0 else None

    result = {
        "protocol": {
            "model": str(model_path),
            "data": str(data_path),
            "validation_split": "val",
            "test_split": "test",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "nms_iou": args.iou,
            "ap_confidence_floor": args.ap_conf_floor,
            "operating_confidence": confidence,
            "operating_confidence_method": threshold_method,
            "tta": False,
        },
        "box": box_summary,
        "mask": mask_summary,
        "speed_ms_per_image": speed,
        "fps_from_inference_only": fps,
        "environment": software_and_hardware(torch, ultralytics),
        "per_class": per_class,
        "notes": [
            "Mask mAP50-95 is the primary metric.",
            "AP uses the full precision-recall curve above the low confidence floor.",
            "Precision/Recall/F1 are macro averages at the confidence selected on validation.",
            "FPS is derived from inference time only; preprocessing and postprocessing are reported separately.",
        ],
    }

    json_path = output / "test_metrics.json"
    csv_path = output / "per_class_metrics.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(result), handle, ensure_ascii=False, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0].keys()) if per_class else ["class_id", "class_name"])
        writer.writeheader()
        writer.writerows(per_class)

    print("\n=== KẾT QUẢ TEST DÙNG CHO BÀI BÁO ===")
    for label, summary in (("BOX", box_summary), ("MASK", mask_summary)):
        print(
            f"{label}: P={summary['precision_macro']:.4f}, R={summary['recall_macro']:.4f}, "
            f"F1={summary['f1_macro']:.4f}, mAP50={summary['map50']:.4f}, "
            f"mAP50-95={summary['map50_95']:.4f}"
        )
    if inference_ms:
        print(f"Inference: {inference_ms:.3f} ms/ảnh; FPS suy luận: {fps:.2f}")
    print(f"\nĐã lưu: {json_path}")
    print(f"Đã lưu: {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"\n[LỖI] {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
