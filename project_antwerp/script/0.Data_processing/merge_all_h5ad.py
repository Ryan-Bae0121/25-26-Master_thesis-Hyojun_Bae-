#!/usr/bin/env python3
"""
Processed_Data 내 모든 데이터셋의 st_norm.h5ad를 하나로 병합하고
patient_id = {dataset}_{sample} 로 설정합니다. HVG / gene_sentence_builder 입력용.
GSE220978 전체 제외, Zenodo/19h1257 제외.
python merge_all_h5ad.py
"""
import scanpy as sc
from pathlib import Path

PROCESSED_ROOT = Path("/project_antwerp/hbae/data/Processed_Data")
OUTPUT_H5AD = Path("/project_antwerp/hbae/data/0317_training_data_excluding_GSE220978_and_19h1257/merged_all_st_norm.h5ad")

SKIP_DATASETS = {"training_data", "GSE220978"}          # 데이터셋 단위 제외
SKIP_SAMPLES  = {"Zenodo/19h1257"}                      # 특정 샘플만 제외 (dataset/sample 형식)

def main():
    adatas = []
    for dataset_dir in sorted(PROCESSED_ROOT.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in SKIP_DATASETS:
            print(f"  [SKIP dataset] {dataset_dir.name}")
            continue

        dataset_name = dataset_dir.name

        for sample_dir in sorted(dataset_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            sample_key = f"{dataset_name}/{sample_dir.name}"
            if sample_key in SKIP_SAMPLES:
                print(f"  [SKIP sample]  {sample_key}")
                continue

            h5ad = sample_dir / "st_norm.h5ad"
            if not h5ad.exists():
                continue

            ad = sc.read_h5ad(h5ad)
            patient_id = f"{dataset_name}_{sample_dir.name}"
            ad.obs["patient_id"] = patient_id

            sample_label = sample_dir.name
            ad.obs_names = [
                f"{sample_label}_{x}_hires" if not str(x).endswith("_hires") else f"{sample_label}_{x}"
                for x in ad.obs_names
            ]

            adatas.append(ad)
            print(f"  + {dataset_name}/{sample_dir.name}")

    if not adatas:
        raise FileNotFoundError("No st_norm.h5ad found under Processed_Data")

    print(f"\nMerging {len(adatas)} samples (join=inner)...")
    merged = sc.concat(adatas, join="inner")
    merged.obs_names_make_unique()
    merged.write_h5ad(OUTPUT_H5AD)
    print(f"Written: {OUTPUT_H5AD} (n_obs={merged.n_obs}, n_vars={merged.n_vars})")

if __name__ == "__main__":
    main()