from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

RESULTS_DIR = Path(r"d:\DACN\week7\results")
FIGURES_DIR = Path(r"d:\DACN\week7\figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def generate_figures():
    print("Dang tao cac bieu do truc quan cho Tuan 7...")

    # 1. Figure 1: Heatmap Confusion Matrix
    cm_path = RESULTS_DIR / "similar_classes_confusion_matrix.csv"
    if cm_path.exists():
        cm_df = pd.read_csv(cm_path, index_col=0)
        plt.figure(figsize=(7.5, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True,
                    xticklabels=cm_df.columns, yticklabels=cm_df.index,
                    annot_kws={"size": 11, "weight": "bold"})
        plt.title("Ma trận nhầm lẫn giữa các lớp rác thải (OpenCLIP ViT-B/32)", fontsize=13, weight="bold", pad=14)
        plt.xlabel("Lớp dự đoán (Predicted Class)", fontsize=11, weight="bold")
        plt.ylabel("Lớp thực tế (Ground Truth)", fontsize=11, weight="bold")
        plt.tight_layout()
        out1 = FIGURES_DIR / "fig1_similar_classes_cm.png"
        plt.savefig(out1, dpi=300)
        plt.close()
        print(f"Saved: {out1}")

    # 2. Figure 2: Phân bố Margin (Top-1 - Top-2 score)
    det_path = RESULTS_DIR / "similar_classes_details.csv"
    if det_path.exists():
        df_det = pd.read_csv(det_path)
        correct_margins = df_det[df_det["is_correct"] == True]["margin"].dropna()
        wrong_margins = df_det[df_det["is_correct"] == False]["margin"].dropna()

        plt.figure(figsize=(8, 4.8))
        plt.hist(correct_margins, bins=25, alpha=0.65, label=f"Dự đoán đúng (N={len(correct_margins)})", color="#2ecc71", edgecolor="black")
        plt.hist(wrong_margins, bins=25, alpha=0.65, label=f"Dự đoán nhầm (N={len(wrong_margins)})", color="#e74c3c", edgecolor="black")
        plt.axvline(0.15, color="#d35400", linestyle="--", linewidth=1.8, label="Vùng ranh giới nhạy cảm (Δ < 0.15)")
        plt.title("Phân bố biên độ chênh lệch độ tin cậy Top-1 vs Top-2 (Confidence Margin Δ)", fontsize=12, weight="bold", pad=12)
        plt.xlabel("Biên độ Margin Δ = Top1_Score - Top2_Score", fontsize=11, weight="bold")
        plt.ylabel("Số lượng mẫu ảnh", fontsize=11, weight="bold")
        plt.legend(fontsize=10.5, loc="upper right")
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        out2 = FIGURES_DIR / "fig2_confidence_margin.png"
        plt.savefig(out2, dpi=300)
        plt.close()
        print(f"Saved: {out2}")

    # 3. Figure 3: So sánh độ chính xác & nhất quán giữa các chiến lược Prompt
    p_path = RESULTS_DIR / "prompt_sensitivity_summary.csv"
    if p_path.exists():
        df_p = pd.read_csv(p_path)
        labels = [s.replace("_", " ") for s in df_p["strategy"]]
        accs = df_p["accuracy_pct"]
        cons = df_p["consistency_with_baseline_pct"]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        rects1 = ax.bar(x - width/2, accs, width, label="Độ chính xác (Accuracy %)", color="#3498db", edgecolor="black")
        rects2 = ax.bar(x + width/2, cons, width, label="Độ nhất quán so với Ensemble (Consistency %)", color="#9b59b6", edgecolor="black")

        ax.set_ylabel("Tỷ lệ phần trăm (%)", fontsize=11, weight="bold")
        ax.set_title("So sánh hiệu năng và độ ổn định giữa các chiến lược Prompt", fontsize=13, weight="bold", pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
        ax.legend(fontsize=10.5, loc="lower right")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, weight="bold")

        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, weight="bold")

        plt.tight_layout()
        out3 = FIGURES_DIR / "fig3_prompt_stability.png"
        plt.savefig(out3, dpi=300)
        plt.close()
        print(f"Saved: {out3}")

    # 4. Figure 4: Đường cong đánh giá Rejection OOD theo ngưỡng Threshold
    sweep_path = RESULTS_DIR / "ood_threshold_sweep.csv"
    if sweep_path.exists():
        df_sw = pd.read_csv(sweep_path)
        plt.figure(figsize=(8.5, 5))
        plt.plot(df_sw["threshold"], df_sw["true_rejection_rate_pct"], marker="o", color="#27ae60", linewidth=2.2, label="True Rejection Rate (OOD đúng %)")
        plt.plot(df_sw["threshold"], df_sw["false_rejection_rate_pct"], marker="s", color="#e74c3c", linewidth=2.2, label="False Rejection Rate (Bỏ sót rác thật %)")
        plt.plot(df_sw["threshold"], df_sw["balanced_rejection_f1"], marker="^", color="#2980b9", linestyle="--", linewidth=2.0, label="Balanced Rejection F1-Score (%)")

        best_row = df_sw.loc[df_sw["balanced_rejection_f1"].idxmax()]
        plt.axvline(best_row["threshold"], color="#8e44ad", linestyle=":", linewidth=2, label=f"Ngưỡng tối ưu τ* = {best_row['threshold']:.2f}")

        plt.title("Đánh giá đánh đổi (Trade-off) theo ngưỡng Unknown Threshold τ", fontsize=12.5, weight="bold", pad=12)
        plt.xlabel("Ngưỡng từ chối Unknown Threshold (τ)", fontsize=11, weight="bold")
        plt.ylabel("Tỷ lệ phần trăm (%)", fontsize=11, weight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10, loc="center left")
        plt.tight_layout()
        out4 = FIGURES_DIR / "fig4_ood_rejection_curves.png"
        plt.savefig(out4, dpi=300)
        plt.close()
        print(f"Saved: {out4}")

if __name__ == "__main__":
    generate_figures()
