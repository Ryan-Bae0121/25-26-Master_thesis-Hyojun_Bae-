#!/usr/bin/env python3
"""
merged_all_st_norm.h5ad 의 n_vars(유전자) 목록을 한 줄에 한 유전자로 저장합니다.

Usage:
    python3 merge_all_h5ad_extract_gene_list.py \
        --h5ad /project_antwerp/hbae/data/0317_training_data_excluding_GSE220978_and_19h1257/merged_all_st_norm.h5ad \
        --out  /project_antwerp/hbae/data/0317_training_data_excluding_GSE220978_and_19h1257/ST_31s_all_shared_genes.txt
"""
import argparse
import scanpy as sc
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Extract gene list (var_names) from merged h5ad")
    p.add_argument("--h5ad", type=str, required=True, help="Path to merged_all_st_norm.h5ad")
    p.add_argument("--out", type=str, required=True, help="Output .txt path (one gene per line)")
    args = p.parse_args()

    h5ad_path = Path(args.h5ad)
    out_path = Path(args.out)

    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad not found: {h5ad_path}")

    ad = sc.read_h5ad(h5ad_path)
    genes = ad.var_names.tolist()
    n = len(genes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(genes))

    print(f"Written: {out_path}")
    print(f"n_vars (genes): {n}")


if __name__ == "__main__":
    main()
