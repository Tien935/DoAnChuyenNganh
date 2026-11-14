"""Sensitivity check on a filtered test subset; never trains or edits images/labels.

Place next to the original evaluate_paper.py. Use the SAME venv and YOLO source
as the original experiment. Defaults to preparation only; --evaluate runs one
test validation using the original validation-selected operating threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.dng', '.mpo', '.pfm'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_reviews(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'split_a', 'split_b', 'image_a', 'image_b', 'distance', 'review'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('CSV thiếu các cột bắt buộc.')
        rows = list(reader)
    for row_number, row in enumerate(rows, 2):
        row['review'] = row['review'].strip().lower()
        if row['review'] not in {'gan_trung', 'khac_anh', 'chua_ro'}:
            raise ValueError(f'Dòng {row_number}: trạng thái review không hợp lệ.')
        if {row['split_a'], row['split_b']} not in (
            {'train', 'val'}, {'train', 'test'}, {'val', 'test'}
        ):
            raise ValueError(f'Dòng {row_number}: cặp split không hợp lệ.')
        if not 0 <= int(row['distance']) <= 64:
            raise ValueError(f'Dòng {row_number}: khoảng cách hash không hợp lệ.')
    return rows


def select_exclusions(rows, exclude_unclear=False):
    """Exclude test endpoints only; train/val sets and thresholds never change."""
    accepted = {'gan_trung'} | ({'chua_ro'} if exclude_unclear else set())
    excluded = {}
    for row in rows:
        if row['review'] not in accepted or 'test' not in (row['split_a'], row['split_b']):
            continue
        side = 'a' if row['split_a'] == 'test' else 'b'
        # PureWindowsPath works with both Windows and forward-slash CSV paths.
        basename = PureWindowsPath(row['image_' + side]).name
        excluded.setdefault(basename, []).append({
            'review': row['review'],
            'distance': int(row['distance']),
            'other_split': row['split_b' if side == 'a' else 'split_a'],
            'other_image': row['image_b' if side == 'a' else 'image_a'],
        })
    return excluded


def label_counts(images, root, class_count):
    counts = Counter()
    for image in images:
        relative = image.relative_to(root / 'test' / 'images')
        label = root / 'test' / 'labels' / relative.with_suffix('.txt')
        if not label.is_file():
            raise FileNotFoundError(f'Thiếu nhãn: {label}')
        for number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            cid = int(parts[0])
            values = list(map(float, parts[1:]))
            if not 0 <= cid < class_count or len(parts) < 7 or len(parts) % 2 != 1:
                raise ValueError(f'Nhãn segmentation không hợp lệ: {label}:{number}')
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError(f'Tọa độ không hợp lệ: {label}:{number}')
            counts[cid] += 1
    return counts


def write_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    import yaml
    root = Path(args.root).resolve()
    data = Path(args.data).resolve()
    original_path = Path(args.original_metrics).resolve()
    original = json.loads(original_path.read_text(encoding='utf-8-sig'))
    protocol = original['protocol']
    threshold = float(protocol['operating_confidence'])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Ngưỡng gốc không hợp lệ.')
    if protocol['operating_confidence_method'] != 'validation mask macro-F1 curve':
        raise ValueError('Kết quả gốc không ghi nhận chọn ngưỡng từ validation.')
    source_yaml = yaml.safe_load(data.read_text(encoding='utf-8-sig'))
    raw_names = source_yaml['names']
    names = ({int(k): str(v) for k, v in raw_names.items()} if isinstance(raw_names, dict)
             else dict(enumerate(raw_names)))
    if set(names) != set(range(16)):
        raise ValueError('Yêu cầu đúng 16 lớp với ID liên tiếp 0–15.')
    folders = {'train': root / 'train' / 'images', 'val': root / 'valid' / 'images',
               'test': root / 'test' / 'images'}
    for folder in folders.values():
        if not folder.is_dir():
            raise FileNotFoundError(folder)
    images = sorted(p.resolve() for p in folders['test'].rglob('*')
                    if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(images) != 1085:
        raise ValueError(f'Test có {len(images)} ảnh, khác 1.085 ảnh gốc. Dừng để kiểm tra phiên bản.')
    if len({p.name for p in images}) != len(images):
        raise ValueError('Tên ảnh test trùng nhau trong các thư mục con; cần ánh xạ đầy đủ.')
    rows = load_reviews(args.candidates)
    exclusions = select_exclusions(rows, args.exclude_unclear)
    unknown = set(exclusions) - {p.name for p in images}
    if unknown:
        raise ValueError(f'CSV trỏ tới ảnh test không tồn tại: {sorted(unknown)[:5]}')
    retained = [p for p in images if p.name not in exclusions]
    removed = [p for p in images if p.name in exclusions]
    if not retained or not removed:
        raise ValueError('Tập giữ lại hoặc tập loại khỏi đánh giá rỗng; kiểm tra CSV.')
    before = label_counts(images, root, len(names))
    after = label_counts(retained, root, len(names))
    if sum(before.values()) != 3132:
        raise ValueError('Số đối tượng test không khớp 3.132 đối tượng gốc.')
    missing_classes = [names[c] for c in names if after[c] == 0]
    if missing_classes:
        raise ValueError(f'Tập lọc mất lớp {missing_classes}; cần thống nhất giao thức trước khi chạy.')
    parent = Path(args.output).resolve()
    output = parent / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    # Keep all new outputs outside the source dataset.
    if parent == root or root in parent.parents:
        raise ValueError('Thư mục output phải nằm ngoài dataset gốc.')
    output.mkdir(parents=True, exist_ok=False)
    test_list = output / 'filtered_test_images.txt'
    test_list.write_text(''.join(p.as_posix() + '\n' for p in retained), encoding='utf-8')
    write_csv(output / 'excluded_test_images.csv', ['image', 'reasons'], [
        {'image': str(p), 'reasons': json.dumps(exclusions[p.name], ensure_ascii=False)} for p in removed
    ])
    write_csv(output / 'class_support.csv', ['class_id', 'class_name', 'original', 'retained', 'excluded'], [
        {'class_id': c, 'class_name': names[c], 'original': before[c], 'retained': after[c],
         'excluded': before[c] - after[c]} for c in names
    ])
    filtered_yaml = output / 'filtered_data.yaml'
    filtered_yaml.write_text(yaml.safe_dump({
        'path': root.as_posix(), 'train': folders['train'].as_posix(),
        'val': folders['val'].as_posix(), 'test': test_list.as_posix(),
        'nc': len(names), 'names': names,
    }, allow_unicode=True, sort_keys=False), encoding='utf-8')
    manifest = {
        'analysis_type': 'post_hoc_test_subset_sensitivity_analysis_not_independent_test',
        'root': str(root), 'original_images': len(images), 'retained_images': len(retained),
        'excluded_images': len(removed), 'original_objects': sum(before.values()),
        'retained_objects': sum(after.values()), 'all_16_classes_present': True,
        'policy': 'direct reviewed cross-split pairs involving test; no deletion or retraining',
        'exclude_unclear': args.exclude_unclear,
        'review_counts': dict(Counter(r['review'] for r in rows)),
        'unnamed_csv_column': 'ignored; named review column is authoritative',
        'candidate_csv_sha256': digest(args.candidates),
        'original_metrics_sha256': digest(original_path),
        'source_yaml_sha256': digest(data),
        'filtered_image_list_sha256': digest(test_list),
        'fixed_operating_confidence': threshold,
        'original_protocol': protocol,
        'limitations': [
            'Only pairs found by the previous pHash screening and reviewed by the user are considered.',
            'Remaining near-duplicates, same-scene dependence and train-val overlap may persist.',
            'Changes in metrics are descriptive; sample size and class/scene composition also change.',
            'Existing weights and original validation-selected threshold are not reselected.',
            'Ultralytics may refresh its labels.cache; source images and label txt files are not edited.',
        ],
    }
    (output / 'subset_audit.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Ảnh test: {len(images)} -> {len(retained)}; loại khỏi danh sách: {len(removed)}')
    print(f'Đối tượng: {sum(before.values())} -> {sum(after.values())}; còn đủ 16 lớp.')
    print(f'Giữ nguyên confidence = {threshold}; không huấn luyện hoặc chọn lại ngưỡng.')
    print(f'Kết quả chuẩn bị: {output}')
    return output, filtered_yaml, original


def evaluate(args, output, filtered_yaml, original):
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from evaluate_paper import metric_summary, class_names, json_ready, software_and_hardware
    p = original['protocol']
    if ultralytics.__version__ != original['environment']['ultralytics']:
        raise RuntimeError('Phiên bản Ultralytics khác lần đo gốc. Hãy dùng đúng venv, không nâng cấp thư viện.')
    model_path = Path(args.model or p['model']).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f'Không tìm thấy best.pt: {model_path}')
    threshold = float(p['operating_confidence'])
    model = YOLO(str(model_path))
    if getattr(model, 'task', None) != 'segment':
        raise ValueError('Checkpoint phải thuộc nhiệm vụ segment.')
    metrics = model.val(
        data=str(filtered_yaml), split='test', imgsz=int(p['imgsz']),
        batch=int(p['batch']), workers=int(p['workers']),
        device=args.device if args.device is not None else p['device'],
        iou=float(p['nms_iou']), conf=float(p['ap_confidence_floor']),
        augment=bool(p.get('tta', False)), plots=True, verbose=True,
        project=str(output / 'yolo_runs'), name='filtered_test', exist_ok=False,
    )
    if not hasattr(metrics, 'seg'):
        raise RuntimeError('Không có kết quả segmentation.')
    # No fallback to metric.p / metric.r: those may be selected at a different point.
    for metric in [metrics.box, metrics.seg]:
        for attribute in ['px', 'p_curve', 'r_curve', 'ap_class_index']:
            if np.asarray(getattr(metric, attribute, [])).size == 0:
                raise RuntimeError(f'Thiếu {attribute}; dừng, không dùng chỉ số dự phòng.')
        if set(map(int, np.asarray(metric.ap_class_index).reshape(-1))) != set(range(16)):
            raise RuntimeError('Metric không bao phủ đủ 16 lớp; cần rà soát trước khi so sánh.')
    box, box_rows = metric_summary(metrics.box, threshold)
    mask, mask_rows = metric_summary(metrics.seg, threshold)
    for summary in [box, mask]:
        if not math.isclose(summary['curve_threshold_used'], threshold, abs_tol=1e-9):
            raise RuntimeError('Điểm đọc PR curve không trùng ngưỡng gốc; dừng để kiểm tra.')
    names = class_names(metrics, model)
    source_names = __import__('yaml').safe_load(filtered_yaml.read_text(encoding='utf-8'))['names']
    if names != source_names:
        raise RuntimeError('Tên lớp checkpoint không khớp data.yaml.')
    by_box = {r['class_id']: r for r in box_rows}
    by_mask = {r['class_id']: r for r in mask_rows}
    per_class = []
    for cid in sorted(names):
        row = {'class_id': cid, 'class_name': names[cid]}
        for prefix, source in [('box', by_box[cid]), ('mask', by_mask[cid])]:
            for key in ['precision', 'recall', 'f1', 'ap50', 'ap50_95']:
                row[f'{prefix}_{key}'] = source[key]
        per_class.append(row)
    speed = dict(metrics.speed)
    result = json_ready({
        'analysis_type': 'filtered_test_sensitivity_check_not_independent_test',
        'protocol': {**p, 'model': str(model_path), 'data': str(filtered_yaml),
                     'operating_confidence_method': 'reused_from_original_validation_no_recalibration'},
        'checkpoint_sha256': digest(model_path), 'box': box, 'mask': mask,
        'speed_ms_per_image': speed, 'per_class': per_class,
        'environment': software_and_hardware(torch, ultralytics),
    })
    (output / 'filtered_test_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(output / 'filtered_per_class_metrics.csv', list(per_class[0]), per_class)
    comparisons = []
    for head in ['box', 'mask']:
        for key in ['precision_macro', 'recall_macro', 'f1_macro', 'map50', 'map50_95']:
            a, b = float(original[head][key]), float(result[head][key])
            comparisons.append({'head': head, 'metric': key, 'original_full_test': a,
                                'filtered_test': b, 'difference': b - a})
    write_csv(output / 'comparison.csv', list(comparisons[0]), comparisons)
    print(f'HOÀN TẤT: {output}')
    print('Gửi lại subset_audit.json, class_support.csv, filtered_test_metrics.json và comparison.csv.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Thư mục chứa train, valid, test')
    parser.add_argument('--data', required=True, help='data.yaml gốc; chỉ đọc')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--original-metrics', required=True)
    parser.add_argument('--model', help='Mặc định dùng đường dẫn checkpoint trong kết quả gốc')
    parser.add_argument('--output', default='sensitivity_results')
    parser.add_argument('--device', default=None)
    parser.add_argument('--exclude-unclear', action='store_true', help='Lọc thêm cặp chua_ro; mặc định chỉ gan_trung')
    parser.add_argument('--evaluate', action='store_true', help='Đánh giá test một lần; không chọn lại ngưỡng')
    args = parser.parse_args()
    output, filtered_yaml, original = prepare(args)
    if args.evaluate:
        evaluate(args, output, filtered_yaml, original)
    else:
        print('Chỉ chuẩn bị tập con. Thêm --evaluate vào lệnh để chạy đánh giá.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit(f'[LỖI] {type(exc).__name__}: {exc}') from exc
