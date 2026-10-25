import csv
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from vlm_eval_core import VLMEvaluator, CLASSES

DATA_DIR = Path(r"d:\DACN\images")
RESULTS_DIR = Path(r"d:\DACN\week7\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "Cardboard": "cardboard",
    "Glass": "glass",
    "Metal": "metal",
    "Paper": "paper",
    "Plastic": "plastic",
    "Trash": "trash",
}

def run_similar_classes_test():
    print("==================================================")
    print("1. KIEM THU CAC LOP GAN GIOI / DE NHAM LAN (VLM)")
    print("==================================================")

    evaluator = VLMEvaluator()

    records = []
    # Collect all labeled images from images/<Class>
    for folder_name, class_name in CLASS_MAP.items():
        sub_dir = DATA_DIR / folder_name
        if not sub_dir.is_dir():
            continue
        img_paths = sorted([p for p in sub_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        for p in img_paths:
            try:
                res = evaluator.predict_image(p, threshold=0.50)
                row = {
                    "image_name": p.name,
                    "ground_truth": class_name,
                    "predicted_class": res["predicted_class"],
                    "top1_class": res["top1_class"],
                    "top1_score": res["top1_score"],
                    "top2_class": res["top2_class"],
                    "top2_score": res["top2_score"],
                    "margin": res["margin"],
                    "status": res["status"],
                    "is_correct": (class_name == res["top1_class"]),
                }
                for c in CLASSES:
                    row[f"score_{c}"] = res["scores"][c]
                records.append(row)
            except Exception as e:
                print(f"Loi doc anh {p.name}: {e}")

    df = pd.DataFrame(records)
    csv_details = RESULTS_DIR / "similar_classes_details.csv"
    df.to_csv(csv_details, index=False, encoding="utf-8-sig")
    print(f"Da xu ly {len(df)} anh co ground truth. Luu chi tiet tai: {csv_details}")

    # Compute Confusion Matrix
    y_true = df["ground_truth"]
    y_pred = df["top1_class"]

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    csv_cm = RESULTS_DIR / "similar_classes_confusion_matrix.csv"
    cm_df.to_csv(csv_cm, encoding="utf-8-sig")
    print("\nMa tran nham lan (Confusion Matrix):")
    print(cm_df)

    # Classification Report
    report_dict = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report_dict).transpose()
    csv_rep = RESULTS_DIR / "similar_classes_metrics.csv"
    rep_df.to_csv(csv_rep, encoding="utf-8-sig")
    print("\nBao cao chi so phan loai:")
    print(rep_df.round(4))

    # Confusion pairs analysis
    print("\n===== PHAN TICH CAC CAP LOP DE NHAM LAN =====")
    pairs = [
        ("plastic", "glass", "Chai/hop nhua trong suot bi nham sang thuy tinh"),
        ("glass", "plastic", "Thuy tinh bi nham sang nhua"),
        ("metal", "paper", "Lon kim loai co nhan giay bi nham sang giay"),
        ("cardboard", "paper", "Carton bi nham sang giay"),
        ("paper", "cardboard", "Giay day bi nham sang carton"),
        ("trash", "paper", "Rac hon hop bi nham sang giay"),
    ]
    pair_stats = []
    for c_true, c_pred, desc in pairs:
        subset = df[df["ground_truth"] == c_true]
        total = len(subset)
        confused = len(subset[subset["top1_class"] == c_pred])
        rate = (confused / total * 100) if total > 0 else 0.0
        avg_margin = subset[subset["top1_class"] == c_pred]["margin"].mean() if confused > 0 else 0.0
        print(f"[{c_true} -> {c_pred}] {desc}: {confused}/{total} ({rate:.2f}%), Margin TB: {avg_margin:.4f}")
        pair_stats.append({
            "true_class": c_true,
            "confused_with": c_pred,
            "description": desc,
            "confused_count": confused,
            "total_true": total,
            "confusion_rate_pct": round(rate, 2),
            "avg_margin": round(avg_margin, 4) if not np.isnan(avg_margin) else 0.0,
        })

    pd.DataFrame(pair_stats).to_csv(RESULTS_DIR / "confusion_pairs_analysis.csv", index=False, encoding="utf-8-sig")
    return df, cm_df, rep_df

if __name__ == "__main__":
    run_similar_classes_test()
