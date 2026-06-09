#!/usr/bin/env python3
"""
HVG_Finetune_meta.csv 기준으로 combined_obs.npy 와 combined_expression_matrix.npy 를 생성합니다.

사용법:
  # 1) merged h5ad에서 expression 추출 (권장)
  python3 build_combined_obs_and_expression.py \
    --meta-csv /data/hbae/Loki_Finetuning/HVG_Finetune_meta.csv \
    --gene-list /data/hbae/Loki_Finetuning/HVG_Finetune_gene_list_full.txt \
    --merged-h5ad /data/hbae/data/Processed_Data/merged_all_st_norm.h5ad \
    --out-dir /data/hbae/Loki_Finetuning

  # 2) 기존 gt_expr / gt_obs에서 서브셋
  python3 build_combined_obs_and_expression.py \
    --meta-csv /data/hbae/Loki_Finetuning/HVG_Finetune_meta.csv \
    --gene-list /data/hbae/Loki_Finetuning/HVG_Finetune_gene_list_full.txt \
    --gt-expr /project_antwerp/hbae/data/combined_expression_matrix.npy \
    --gt-obs  /project_antwerp/hbae/data/combined_obs.npy \
    --all-genes /project_antwerp/hbae/data/all_shared_genes.txt \
    --out-dir /data/hbae/Loki_Finetuning
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build combined_obs.npy and combined_expression_matrix.npy from HVG_Finetune_meta")
    parser.add_argument("--meta-csv", type=str, required=True, help="HVG_Finetune_meta.csv path")
    parser.add_argument("--gene-list", type=str, required=True, help="Gene list (one per line), order = expression columns")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: same as meta-csv dir)")
    # Source 1: merged h5ad
    parser.add_argument("--merged-h5ad", type=str, default=None, help="merged_all_st_norm.h5ad (obs = spot IDs, var = genes)")
    # Source 2: existing gt_expr / gt_obs
    parser.add_argument("--gt-expr", type=str, default=None, help="Existing combined_expression_matrix.npy")
    parser.add_argument("--gt-obs", type=str, default=None, help="Existing combined_obs.npy (same row order as gt-expr)")
    parser.add_argument("--all-genes", type=str, default=None, help="all_shared_genes.txt (column order of gt-expr); required if --gt-expr used")
    args = parser.parse_args()

    meta_path = Path(args.meta_csv)
    gene_list_path = Path(args.gene_list)
    out_dir = Path(args.out_dir) if args.out_dir else meta_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_path}")
    if not gene_list_path.exists():
        raise FileNotFoundError(f"Gene list not found: {gene_list_path}")

    # Meta: obs ID = first column (Unnamed: 0 or 동일)
    df = pd.read_csv(meta_path)
    obs_col = df.columns[0]
    meta_obs_ids = df[obs_col].astype(str).values
    n_obs = len(meta_obs_ids)

    # Gene list (order = column order of expression matrix)
    with open(gene_list_path) as f:
        genes_want = [line.strip() for line in f if line.strip()]
    n_genes = len(genes_want)
    print(f"Meta rows: {n_obs}, Gene list: {n_genes}")

    # --- 1) combined_obs.npy: 메타와 동일 순서
    combined_obs = np.array(meta_obs_ids, dtype=object)
    obs_out = out_dir / "combined_obs.npy"
    np.save(obs_out, combined_obs)
    print(f"Saved: {obs_out} (shape={combined_obs.shape})")

    # --- 2) combined_expression_matrix.npy
    if args.merged_h5ad and Path(args.merged_h5ad).exists():
        # Source: merged h5ad
        import scanpy as sc
        ad = sc.read_h5ad(args.merged_h5ad)
        ad_obs_names = np.array([str(x) for x in ad.obs_names])
        meta_set = set(meta_obs_ids)
        found = [x in meta_set for x in ad_obs_names]
        if sum(found) != n_obs:
            missing_in_h5ad = set(meta_obs_ids) - set(ad_obs_names)
            print(f"Warning: {len(missing_in_h5ad)} meta obs not in h5ad. Filling those rows with 0.")
        # Subset ad to obs that exist in both, then reorder to meta order
        order_in_meta = {oid: i for i, oid in enumerate(meta_obs_ids)}
        # Build row index: for each meta_obs_id, get row in ad (or -1)
        ad_obs_to_row = {str(x): i for i, x in enumerate(ad.obs_names)}
        row_indices = []
        for oid in meta_obs_ids:
            row_indices.append(ad_obs_to_row.get(str(oid), -1))
        # Subset vars to genes_want
        var_in = [g for g in genes_want if g in ad.var_names]
        if len(var_in) != n_genes:
            print(f"Warning: only {len(var_in)}/{n_genes} genes found in h5ad. Missing columns filled with 0.")
        ad_genes = ad[:, var_in]
        X_full = ad_genes.X
        if hasattr(X_full, "toarray"):
            X_full = X_full.toarray()
        X_full = np.asarray(X_full, dtype=np.float32)
        # Build (n_obs, n_genes) in meta order; missing obs stay 0
        expr_matrix = np.zeros((n_obs, n_genes), dtype=np.float32)
        for i, ridx in enumerate(row_indices):
            if ridx >= 0:
                for j, g in enumerate(genes_want):
                    if g in var_in:
                        expr_matrix[i, j] = X_full[ridx, var_in.index(g)]
    elif args.gt_expr and args.gt_obs and Path(args.gt_expr).exists() and Path(args.gt_obs).exists():
        # Source: existing combined_expression_matrix + combined_obs
        gt_obs = np.load(args.gt_obs, allow_pickle=True).ravel()
        gt_obs = np.array([str(x) for x in gt_obs])
        gt_expr = np.load(args.gt_expr)
        if args.all_genes is None or not Path(args.all_genes).exists():
            raise FileNotFoundError("--all-genes required when using --gt-expr (column order of gt-expr)")
        with open(args.all_genes) as f:
            all_genes = [line.strip() for line in f if line.strip()]
        obs_to_row = {str(x): i for i, x in enumerate(gt_obs)}
        # Row indices in meta order
        row_indices = []
        for oid in meta_obs_ids:
            if oid not in obs_to_row:
                raise ValueError(f"Obs ID not in gt_obs: {oid}")
            row_indices.append(obs_to_row[oid])
        # Column indices: genes_want in all_genes order
        gene_to_col_gt = {g: i for i, g in enumerate(all_genes)}
        col_indices = []
        for g in genes_want:
            if g in gene_to_col_gt:
                col_indices.append(gene_to_col_gt[g])
            else:
                raise ValueError(f"Gene not in all_shared_genes: {g}")
        expr_matrix = gt_expr[np.ix_(row_indices, col_indices)].astype(np.float32)
    else:
        raise RuntimeError(
            "Provide either (--merged-h5ad path) or (--gt-expr, --gt-obs, --all-genes). "
            "merged_h5ad or gt_expr/gt_obs not found or not given."
        )

    expr_out = out_dir / "combined_expression_matrix.npy"
    np.save(expr_out, expr_matrix)
    print(f"Saved: {expr_out} (shape={expr_matrix.shape})")
    print("Done.")


if __name__ == "__main__":
    main()
