import pandas as pd
from pathlib import Path

TRAIN_DF = Path("/project_antwerp/hbae/data/train_df.csv")
NEW_TRAIN_DF = Path("/project_antwerp/hbae/data/train_df_fixed.csv")

df = pd.read_csv(TRAIN_DF)

print("원본 예시 5개:")
print(df["img_path"].head())

# 여기서 old_root / new_root는 실제 경로에 맞게 수정해야 함
old_root = "/home/students/hbae/processed_loki"
new_root = "/project_antwerp/hbae/data/Processed_Data"

df["img_path"] = df["img_path"].str.replace(old_root, new_root, regex=False)

print("\n변경 후 예시 5개:")
print(df["img_path"].head())

df.to_csv(NEW_TRAIN_DF, index=False)
print(f"\n✅ Fixed train_df saved to: {NEW_TRAIN_DF}")
