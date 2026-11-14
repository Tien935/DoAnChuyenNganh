#!/usr/bin/env python3
"""Read-only audit for a YOLO polygon-segmentation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--output", default="audit_results")
    return parser.parse_args()


def split_sources(data: dict, yaml_path: Path, key: str) -> list[Path]:
    value = data.get(key)
    if value is None and key == "val":
        value = data.get("valid")
    values = value if isinstance(value, list) else [value]
    base = Path(data.get("path", yaml_path.parent))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    result = []
    for item in values:
        if item is None:
            continue
        path = Path(str(item))
        candidate = path.resolve() if path.is_absolute() else (base / path).resolve()

        # Roboflow commonly exports paths such as ../train/images. Ultralytics
        # first resolves that path literally and, when it does not exist,
        # retries after removing the leading ../ relative to the YAML folder.
        # Mirror that behavior so the audit scans the same files as model.val().
        normalized = str(item).replace("\\", "/")
        if not candidate.exists() and normalized.startswith("../"):
            fallback = (base / normalized[3:]).resolve()
            if fallback.exists():
                candidate = fallback
        result.append(candidate)
    return result


def resolved_dataset_config(yaml_path: Path) -> dict:
    """Use Ultralytics' resolver when available, with a YAML-only fallback."""
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    try:
        from ultralytics.data.utils import check_det_dataset

        resolved = check_det_dataset(str(yaml_path), autodownload=False)
        if isinstance(resolved, dict):
            return resolved
    except Exception as error:
        print(f"[CẢNH BÁO] Không dùng được bộ phân giải Ultralytics: {error}")
    return raw


def images_from_source(source: Path) -> Iterable[Path]:
    if source.is_dir():
        yield from sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    elif source.is_file():
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            path = Path(line)
            yield path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    matches = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if matches:
        parts[matches[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def polygon_area(coords: list[float]) -> float:
    points = list(zip(coords[0::2], coords[1::2]))
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    yaml_path = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = resolved_dataset_config(yaml_path)
    names_raw = data.get("names", {})
    names = {int(k): str(v) for k, v in names_raw.items()} if isinstance(names_raw, dict) else {i: str(v) for i, v in enumerate(names_raw)}
    nc = int(data.get("nc", len(names)))

    errors: list[dict] = []
    split_stats: dict[str, dict] = {}
    distribution: dict[str, Counter] = {}
    hash_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in ("train", "val", "test"):
        images = []
        for source in split_sources(data, yaml_path, split):
            images.extend(images_from_source(source))
        images = sorted(set(images))
        counts: Counter = Counter()
        missing_labels = empty_labels = invalid_objects = objects = 0

        for image in images:
            if not image.is_file():
                errors.append({"split": split, "file": str(image), "issue": "missing_image"})
                continue
            hash_locations[sha256(image)].append((split, str(image)))
            label = label_for_image(image)
            if not label.is_file():
                missing_labels += 1
                errors.append({"split": split, "file": str(label), "issue": "missing_label"})
                continue
            lines = [line.strip() for line in label.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
                continue
            seen = set()
            for line_number, line in enumerate(lines, start=1):
                issue = None
                parts = line.split()
                if line in seen:
                    issue = "duplicate_annotation"
                seen.add(line)
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                    if len(parts) < 7 or len(coords) % 2:
                        issue = issue or "invalid_polygon_format"
                    elif not 0 <= class_id < nc:
                        issue = issue or "class_out_of_range"
                    elif any(value < 0 or value > 1 for value in coords):
                        issue = issue or "coordinate_out_of_range"
                    elif polygon_area(coords) <= 1e-10:
                        issue = issue or "zero_area_polygon"
                    if issue is None:
                        counts[class_id] += 1
                        objects += 1
                except (ValueError, IndexError):
                    issue = issue or "non_numeric_or_empty_annotation"
                if issue:
                    invalid_objects += 1
                    errors.append({"split": split, "file": str(label), "line": line_number, "issue": issue})

        distribution[split] = counts
        split_stats[split] = {
            "images": len(images),
            "objects": objects,
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "invalid_annotations": invalid_objects,
        }

    duplicates = []
    for digest, locations in hash_locations.items():
        involved_splits = sorted({split for split, _ in locations})
        if len(involved_splits) > 1:
            duplicates.append({"sha256": digest, "splits": involved_splits, "files": [path for _, path in locations]})

    summary = {
        "data_yaml": str(yaml_path),
        "number_of_classes": nc,
        "class_names": names,
        "splits": split_stats,
        "cross_split_exact_duplicate_groups": len(duplicates),
        "error_count": len(errors),
        "note": "This script is read-only. Near-duplicate image detection still requires a separate perceptual-hash review.",
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "audit_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "cross_split_exact_duplicates.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output / "class_distribution.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["class_id", "class_name", "train", "val", "test", "total"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for class_id in range(nc):
            values = {split: distribution.get(split, Counter()).get(class_id, 0) for split in ("train", "val", "test")}
            writer.writerow({"class_id": class_id, "class_name": names.get(class_id, str(class_id)), **values, "total": sum(values.values())})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nĐã lưu kết quả audit tại: {output}")


if __name__ == "__main__":
    main()
