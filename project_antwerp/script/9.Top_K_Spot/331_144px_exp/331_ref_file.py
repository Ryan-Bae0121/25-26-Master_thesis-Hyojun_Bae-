"""
144px 결과를 331개 patient-level 슬라이드로 필터링
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path

# 331개 patient-level slide IDs (TCGA_patch short name 폴더)
PATCH_DIR = "/project_antwerp/TCGA-HNSC/TCGA_patch"
patient_slides = set(
    p.name for p in Path(PATCH_DIR).iterdir()
    if p.is_dir() and '.' not in p.name  # UUID 없는 short name만
)
print(f"Patient-level slides: {len(patient_slides)}")  # 331 확인

# ref_file에서 331개만 필터링
REF_FILE = "/project_antwerp/hbae/data/ref_file.csv"
ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
ref_331 = ref_df[ref_df["slide_id"].isin(patient_slides)]
print(f"Filtered ref: {len(ref_331)}")  # 331 확인

# 저장
ref_331.to_csv("/project_antwerp/hbae/ref_file_331.csv")
print("Saved ref_file_331.csv")