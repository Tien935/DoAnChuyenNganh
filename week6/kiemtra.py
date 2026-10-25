import pandas as pd

df = pd.read_csv(r"D:\DACN\week6\vlm_batch_results.csv")

print(df[["image_name", "top_class", "top_score", "status"]].head(15))