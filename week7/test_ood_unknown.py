from pathlib import Path
import pandas as pd
import numpy as np

from vlm_eval_core import VLMEvaluator, CLASSES

DATA_DIR = Path(r"d:\DACN\images")
OOD_DIR = Path(r"d:\DACN\week7\data\ood_images")
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

def run_ood_unknown_test():
    print("==================================================")
    print("3. KIEM THU ANH OOD (NGOAI 6 LOP) & NGUONG UNKNOWN")
    print("==================================================")

    evaluator = VLMEvaluator()

    # 1. Thu thap tap In-Distribution (ID) - Mau rac thuc te
    id_records = []
    for folder_name, class_name in CLASS_MAP.items():
        sub_dir = DATA_DIR / folder_name
        if not sub_dir.is_dir():
            continue
        imgs = sorted([p for p in sub_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])[:30]
        for p in imgs:
            res = evaluator.predict_image(p, threshold=0.0)  # Lay raw scores
            id_records.append({
                "image_name": p.name,
                "data_type": "IN_DISTRIBUTION",
                "ground_truth": class_name,
                "top1_class": res["top1_class"],
                "top1_score": res["top1_score"],
                "top2_class": res["top2_class"],
                "top2_score": res["top2_score"],
                "margin": res["margin"],
                "is_waste": True,
            })

    # 2. Thu thap tap Out-of-Distribution (OOD) - Anh khong thuoc 6 lop
    ood_records = []
    ood_imgs = sorted([p for p in OOD_DIR.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
    for p in ood_imgs:
        res = evaluator.predict_image(p, threshold=0.0)
        ood_records.append({
            "image_name": p.name,
            "data_type": "OUT_OF_DISTRIBUTION",
            "ground_truth": "non_waste_or_other",
            "top1_class": res["top1_class"],
            "top1_score": res["top1_score"],
            "top2_class": res["top2_class"],
            "top2_score": res["top2_score"],
            "margin": res["margin"],
            "is_waste": False,
        })

    all_df = pd.DataFrame(id_records + ood_records)
    csv_all = RESULTS_DIR / "ood_test_details.csv"
    all_df.to_csv(csv_all, index=False, encoding="utf-8-sig")

    print(f"Tong so mau ID (rac that): {len(id_records)}")
    print(f"Tong so mau OOD (khong phai 6 lop rac): {len(ood_records)}")

    # 3. Quet dai nguong Unknown Threshold tu 0.30 den 0.80
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep_results = []

    id_df = all_df[all_df["data_type"] == "IN_DISTRIBUTION"]
    ood_df = all_df[all_df["data_type"] == "OUT_OF_DISTRIBUTION"]

    for tau in thresholds:
        # Danh gia tren OOD: Ty le tu choi dung (True Rejection Rate - Specificity)
        ood_rejected = (ood_df["top1_score"] < tau).sum()
        trr = (ood_rejected / len(ood_df)) * 100

        # Danh gia tren ID: Ty le tu choi nham (False Rejection Rate)
        id_rejected = (id_df["top1_score"] < tau).sum()
        frr = (id_rejected / len(id_df)) * 100

        # Do chinh xac tren tap ID duoc giu lai (Retained Accuracy)
        id_retained = id_df[id_df["top1_score"] >= tau]
        if len(id_retained) > 0:
            retained_acc = (id_retained["top1_class"] == id_retained["ground_truth"]).sum() / len(id_retained) * 100
        else:
            retained_acc = 0.0

        # F1 score cho kha nang tu choi OOD vs giu ID
        # Precision = TRR / (TRR + (100 - FRR))
        retention_rate = 100 - frr
        if (trr + retention_rate) > 0:
            rejection_f1 = 2 * (trr * retention_rate) / (trr + retention_rate)
        else:
            rejection_f1 = 0.0

        sweep_results.append({
            "threshold": tau,
            "ood_total": len(ood_df),
            "ood_rejected_correctly": ood_rejected,
            "true_rejection_rate_pct": round(trr, 2),
            "id_total": len(id_df),
            "id_rejected_falsely": id_rejected,
            "false_rejection_rate_pct": round(frr, 2),
            "id_retention_rate_pct": round(retention_rate, 2),
            "id_retained_accuracy_pct": round(retained_acc, 2),
            "balanced_rejection_f1": round(rejection_f1, 2),
        })

    sweep_df = pd.DataFrame(sweep_results)
    csv_sweep = RESULTS_DIR / "ood_threshold_sweep.csv"
    sweep_df.to_csv(csv_sweep, index=False, encoding="utf-8-sig")

    print("\n===== KET QUA KHAO SAT NGUONG UNKNOWN THRESHOLD =====")
    print(sweep_df[["threshold", "true_rejection_rate_pct", "false_rejection_rate_pct", "id_retained_accuracy_pct", "balanced_rejection_f1"]].to_string(index=False))

    # Xac dinh nguong toi uu
    best_row = sweep_df.loc[sweep_df["balanced_rejection_f1"].idxmax()]
    print(f"\n-> Nguong toi uu khuyen nghi: tau = {best_row['threshold']:.2f} (F1 = {best_row['balanced_rejection_f1']:.2f}%, TRR = {best_row['true_rejection_rate_pct']}%, FRR = {best_row['false_rejection_rate_pct']}%)")

    return sweep_df, all_df

if __name__ == "__main__":
    run_ood_unknown_test()
