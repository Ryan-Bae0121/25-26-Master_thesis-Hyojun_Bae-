#!/usr/bin/env python3
"""
Create HVG (Highly Variable Genes) list from training_data combined matrix.
Uses scanpy's highly_variable_genes (flavor='seurat', for log-normalized data).
Output: one gene per line, same format as vocab used by gene_sentence_builder.
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc


def main():
    p = argparse.ArgumentParser(description="Create HVG gene list from combined_expression_matrix + all_shared_genes")
    p.add_argument(
        "--training_dir",
        type=str,
        default="/home/students/hbae/data/Processed_Data/training_data_excluding_GSE220978_and_19h1257",
        help="Directory containing combined_expression_matrix.npy and all_shared_genes.txt",
    )
    p.add_argument(
        "--n_top_genes",
        type=int,
        default=None,
        help="Optional cap for HVG count (default: no fixed cap)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for HVG list (default: <training_dir>/HVG_<n>_genes.txt)",
    )
    p.add_argument(
        "--flavor",
        type=str,
        default="seurat",
        choices=["seurat", "cell_ranger", "seurat_v3"],
        help="Flavor for HVG (default: seurat; data is log-normalized)",
    )
    args = p.parse_args()

    base = Path(args.training_dir)
    matrix_path = base / "combined_expression_matrix.npy"
    genes_path = base / "all_shared_genes.txt"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Not found: {matrix_path}")
    if not genes_path.exists():
        raise FileNotFoundError(f"Not found: {genes_path}")

    print("Loading combined matrix and gene list...")
    X = np.load(matrix_path)
    genes = [g.strip() for g in genes_path.read_text().splitlines() if g.strip()]
    if len(genes) != X.shape[1]:
        raise ValueError(f"Gene count {len(genes)} != matrix columns {X.shape[1]}")

    # Build AnnData (obs can be dummy for HVG)
    adata = sc.AnnData(X=X.astype(np.float32), var=dict(gene=genes))
    adata.var_names = genes
    adata.obs_names = [f"spot_{i}" for i in range(adata.n_obs)]
    print(f"AnnData: {adata.n_obs} spots x {adata.n_vars} genes")

    # HVG: data is already log1p-normalized (from st_norm_noHK)
    if args.flavor in ("seurat_v3",):
        # seurat_v3 expects raw counts; skip if data is log-norm
        print("Using flavor=seurat (log-normalized). For seurat_v3 use raw counts.")
        args.flavor = "seurat"
    n_top_genes = None if args.n_top_genes is None else min(args.n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=args.flavor)
    hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()
    print(f"Selected {len(hvg_genes)} HVG")

    default_name = f"HVG_{args.n_top_genes}_genes.txt" if args.n_top_genes is not None else "HVG_auto_genes.txt"
    out_path = Path(args.output) if args.output else base / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(hvg_genes) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
