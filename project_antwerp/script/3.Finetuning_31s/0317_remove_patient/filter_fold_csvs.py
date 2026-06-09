#!/usr/bin/env python3
"""
0228_New_HVG_10fold의 모든 train/val CSV에서
GSE220978 (Patient1~4) 및 Zenodo 19h1257 샘플 행을 제거하여
0317_New_HVG_10fold 폴더에 새로 저장
"""
import pandas as pd
from pathlib import Path

INPUT_DIR  = Path("/project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold")
OUTPUT_DIR = Path("/project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_PATTERNS = [
    "GSE220978",
    "19h1257",
]

def should_exclude(path: str) -> bool:
    return any(p in path for p in EXCLUDE_PATTERNS)

for csv_file in sorted(INPUT_DIR.glob("fold_*.csv")):
    df = pd.read_csv(csv_file)
    before = len(df)
    df_filtered = df[~df["img_path"].apply(should_exclude)]
    after = len(df_filtered)
    removed = before - after

    out_path = OUTPUT_DIR / csv_file.name
    df_filtered.to_csv(out_path, index=False)
    print(f"{csv_file.name}: {before} → {after} rows (removed {removed})")

# folds.json, fold_stats.csv도 복사
import shutil
for extra in ["folds.json", "fold_stats.csv"]:
    src = INPUT_DIR / extra
    if src.exists():
        shutil.copy(src, OUTPUT_DIR / extra)
        print(f"Copied: {extra}")

print(f"\n✅ Done! Output: {OUTPUT_DIR}")