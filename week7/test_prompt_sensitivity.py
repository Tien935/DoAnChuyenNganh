from pathlib import Path
import pandas as pd
import numpy as np

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

# 5 Chien luoc Prompt khac nhau de kiem tra do nhay cam
PROMPT_STRATEGIES = {
    "P1_Single_Short": {
        "cardboard": ["cardboard"],
        "glass": ["glass"],
        "metal": ["metal"],
        "paper": ["paper"],
        "plastic": ["plastic"],
        "trash": ["trash"],
    },
    "P2_Template_Standard": {
        "cardboard": ["a photo of cardboard waste"],
        "glass": ["a photo of glass waste"],
        "metal": ["a photo of metal waste"],
        "paper": ["a photo of paper waste"],
        "plastic": ["a photo of plastic waste"],
        "trash": ["a photo of trash"],
    },
    "P3_Descriptive_3": {
        "cardboard": [
            "a photo of brown corrugated cardboard box",
            "an image of thick cardboard packaging material",
            "pieces of discarded cardboard paperboard",
        ],
        "glass": [
            "a photo of a transparent glass bottle",
            "an image of fragile glass jar and broken glass",
            "clear glass container recyclable waste",
        ],
        "metal": [
            "a photo of an aluminum beverage can",
            "an image of metallic tin can and metal scrap",
            "shiny metal object recyclable material",
        ],
        "paper": [
            "a photo of thin white paper sheet",
            "an image of printed newspaper and paper scraps",
            "flat paper document recyclable waste",
        ],
        "plastic": [
            "a photo of a clear plastic beverage bottle",
            "an image of molded plastic packaging container",
            "synthetic plastic bottle recyclable material",
        ],
        "trash": [
            "a photo of dirty mixed garbage waste",
            "an image of non-recyclable miscellaneous trash",
            "a pile of unsorted household garbage",
        ],
    },
    "P4_Ensemble_5": {  # Baseline chuẩn từ tuần 5 và tuần 6
        "cardboard": [
            "a photo of cardboard waste",
            "a photo of a cardboard box",
            "an image of discarded cardboard",
            "a piece of cardboard waste",
            "cardboard recyclable material",
        ],
        "glass": [
            "a photo of glass waste",
            "a photo of a glass bottle",
            "an image of discarded glass",
            "a piece of glass waste",
            "glass recyclable material",
        ],
        "metal": [
            "a photo of metal waste",
            "a photo of a metal can",
            "an image of discarded metal",
            "a piece of metal waste",
            "metal recyclable material",
        ],
        "paper": [
            "a photo of paper waste",
            "a photo of discarded paper",
            "an image of paper waste",
            "a piece of paper waste",
            "paper recyclable material",
        ],
        "plastic": [
            "a photo of plastic waste",
            "a photo of a plastic bottle",
            "an image of discarded plastic",
            "a piece of plastic waste",
            "plastic recyclable material",
        ],
        "trash": [
            "a photo of trash",
            "a photo of general waste",
            "an image of discarded trash",
            "a piece of miscellaneous waste",
            "a pile of general garbage",
        ],
    },
    "P5_Contextual_Distracted": {  # Prompt nhiễu bối cảnh / gây xao nhãng
        "cardboard": [
            "a photo of cardboard lying on a messy floor",
            "cardboard box with shipping labels and tape",
        ],
        "glass": [
            "a photo of a glass bottle with printed paper label",
            "glass beverage container under indoor lighting",
        ],
        "metal": [
            "a photo of a metal can with colorful paper wrapper",
            "beverage can sitting on a wooden table",
        ],
        "paper": [
            "a photo of paper scrap mixed with plastic wrapper",
            "printed office paper sheets on a desk",
        ],
        "plastic": [
            "a photo of a plastic bottle with plastic cap and brand label",
            "crushed plastic container on a dark background",
        ],
        "trash": [
            "a photo of general trash bag on the pavement",
            "messy trash pile in a waste disposal bin",
        ],
    },
}

def run_prompt_sensitivity_test(sample_per_class: int = 30):
    print("==================================================")
    print("2. KIEM THU DO NHAY CAM & DO ON DINH CUA PROMPT")
    print("==================================================")

    # Chon tap mau can bang de so sanh cong bang giua cac chien luoc
    sample_images = []
    for folder_name, class_name in CLASS_MAP.items():
        sub_dir = DATA_DIR / folder_name
        if not sub_dir.is_dir():
            continue
        all_imgs = sorted([p for p in sub_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        selected = all_imgs[:sample_per_class] if len(all_imgs) >= sample_per_class else all_imgs
        for p in selected:
            sample_images.append((p, class_name))

    print(f"Tong so mau thu nghiem do nhay prompt: {len(sample_images)} anh ({sample_per_class} anh/lop)")

    # Khoi tao Evaluator
    evaluator = VLMEvaluator(prompts=PROMPT_STRATEGIES["P4_Ensemble_5"])

    # Luu ket qua du doan cua tung chien luoc
    strategy_results = {}
    strategy_metrics = []

    for strat_name, prompt_dict in PROMPT_STRATEGIES.items():
        print(f"Dang danh gia chien luoc: {strat_name}...")
        evaluator.set_prompts(prompt_dict)

        preds = []
        correct_count = 0
        conf_scores = []

        for img_path, gt_class in sample_images:
            res = evaluator.predict_image(img_path)
            is_corr = (res["top1_class"] == gt_class)
            if is_corr:
                correct_count += 1
            conf_scores.append(res["top1_score"])
            preds.append({
                "image_name": img_path.name,
                "ground_truth": gt_class,
                "predicted_class": res["top1_class"],
                "score": res["top1_score"],
                "margin": res["margin"],
                "is_correct": is_corr,
            })

        acc = (correct_count / len(sample_images)) * 100
        avg_conf = float(np.mean(conf_scores))
        std_conf = float(np.std(conf_scores))

        strategy_results[strat_name] = preds
        strategy_metrics.append({
            "strategy": strat_name,
            "accuracy_pct": round(acc, 2),
            "avg_confidence": round(avg_conf, 4),
            "std_confidence": round(std_conf, 4),
            "num_samples": len(sample_images),
        })

    # Tinh toan tinh nhat quan (Consistency) so voi Ensemble baseline (P4)
    baseline_preds = strategy_results["P4_Ensemble_5"]
    for row in strategy_metrics:
        strat = row["strategy"]
        curr_preds = strategy_results[strat]
        agreements = sum(1 for b, c in zip(baseline_preds, curr_preds) if b["predicted_class"] == c["predicted_class"])
        consistency = (agreements / len(sample_images)) * 100
        row["consistency_with_baseline_pct"] = round(consistency, 2)
        row["flips_from_baseline"] = len(sample_images) - agreements

    summary_df = pd.DataFrame(strategy_metrics)
    csv_summary = RESULTS_DIR / "prompt_sensitivity_summary.csv"
    summary_df.to_csv(csv_summary, index=False, encoding="utf-8-sig")

    print("\n===== KET QUA SO SANH CAC CHIEN LUOC PROMPT =====")
    print(summary_df.to_string(index=False))

    # Bang chi tiet theo tung anh
    details = []
    for i, (img_path, gt) in enumerate(sample_images):
        row_det = {
            "image_name": img_path.name,
            "ground_truth": gt,
        }
        for strat in PROMPT_STRATEGIES.keys():
            row_det[f"{strat}_pred"] = strategy_results[strat][i]["predicted_class"]
            row_det[f"{strat}_score"] = strategy_results[strat][i]["score"]
        details.append(row_det)

    details_df = pd.DataFrame(details)
    csv_det = RESULTS_DIR / "prompt_sensitivity_details.csv"
    details_df.to_csv(csv_det, index=False, encoding="utf-8-sig")
    print(f"\nDa luu chi tiet tai: {csv_det}")

    return summary_df, details_df

if __name__ == "__main__":
    run_prompt_sensitivity_test()
