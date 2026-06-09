#!/usr/bin/env python3
"""
10-fold Cross Validation for Loki PredEx with Top-K Genes (e.g. top 50).

커스텀 패치 + top 50 유전자 CSV로 10-fold CV를 돌릴 때 사용합니다.

필요 입력:
  - patches_csv: 패치 목록 (filepath, title 또는 label, 선택: obs_key)
  - top_genes_csv: top 50 유전자 CSV (한 열에 유전자명, 열이름 'gene' 또는 첫 번째 열)
  - gt_expr, gt_obs, gene_list: Loki PredEx와 동일 (전체 발현 행렬, spot ID 목록, 전체 유전자 목록)

Usage:
  python run_10fold_loki_predex.py \
    --patches_csv /path/to/patches_with_title.csv \
    --top_genes_csv /path/to/top50_genes.csv \
    --gt_expr /path/to/combined_expression_matrix.npy \
    --gt_obs /path/to/combined_obs.npy \
    --gene_list /path/to/all_shared_genes.txt \
    --output_dir ./loki_predex_10fold \
    --device cuda:0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_top_genes_from_csv(csv_path: str, column: str = None) -> list:
    """Load gene names from CSV. Uses first column or column named 'gene'."""
    df = pd.read_csv(csv_path)
    if column and column in df.columns:
        genes = df[column].dropna().astype(str).str.strip().tolist()
    elif "gene" in df.columns:
        genes = df["gene"].dropna().astype(str).str.strip().tolist()
    else:
        genes = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    return [g for g in genes if g]


def ensure_title_column(patches_df: pd.DataFrame, gt_expr: np.ndarray,
                        gt_obs: np.ndarray, gene_list: list,
                        top_genes: list, topn: int = 50) -> pd.DataFrame:
    """If 'title' or 'label' missing, build from gt_expr (top genes per spot)."""
    if "title" in patches_df.columns:
        return patches_df
    if "label" in patches_df.columns:
        patches_df = patches_df.rename(columns={"label": "title"})
        return patches_df

    patches_df = patches_df.copy()
    obs_to_idx = {b: i for i, b in enumerate(gt_obs)}
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    top_indices = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
    if len(top_indices) == 0:
        raise ValueError("No top genes found in gene_list. Check top_genes_csv and gene_list.")

    if "obs_key" not in patches_df.columns:
        path_col = "filepath" if "filepath" in patches_df.columns else "img_path"
        patches_df["obs_key"] = patches_df[path_col].apply(
            lambda p: Path(p).stem if isinstance(p, str) else ""
        )

    titles = []
    for _, row in patches_df.iterrows():
        idx = obs_to_idx.get(row["obs_key"])
        if idx is None:
            titles.append(" ".join(top_genes[:topn]))
            continue
        expr = gt_expr[idx, :]
        expr_sub = expr[top_indices]
        order = np.argsort(-expr_sub)[:topn]
        genes_at_spot = [gene_list[top_indices[i]] for i in order]
        titles.append(" ".join(genes_at_spot))
    patches_df["title"] = titles
    return patches_df


def run_one_fold(fold_id: int, train_csv: Path, val_csv: Path,
                 hvg_file: Path, output_base: Path, args) -> dict:
    """Run loki_predex_exact.py for one fold and return metrics."""
    out_dir = output_base / f"fold_{fold_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "loki_predex_exact.py"),
        "--train_csv", str(train_csv),
        "--val_csv", str(val_csv),
        "--hvg_file", str(hvg_file),
        "--gt_expr", args.gt_expr,
        "--gt_obs", args.gt_obs,
        "--gene_list", args.gene_list,
        "--pretrained", args.pretrained,
        "--output_dir", str(out_dir),
        "--device", args.device,
        "--pred_style", args.pred_style,
        "--temperature", str(args.temperature),
    ]
    if args.top_k is not None:
        cmd += ["--top_k", str(args.top_k)]
    if args.save_predictions:
        cmd += ["--save_predictions"]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"Fold {fold_id} failed: {r.returncode}")

    results_path = out_dir / "loki_predex_results.json"
    if not results_path.exists():
        return {"fold": fold_id, "error": "no results json"}
    with open(results_path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="10-fold CV for Loki PredEx with top-K genes CSV")
    p.add_argument("--patches_csv", required=True,
                   help="CSV with columns: filepath, title (or label); optional: obs_key")
    p.add_argument("--top_genes_csv", required=True,
                   help="CSV with one column of gene names (e.g. top 50 genes)")
    p.add_argument("--gene_column", default=None,
                   help="Column name in top_genes_csv (default: first column or 'gene')")
    p.add_argument("--gt_expr", required=True,
                   help="Path to combined_expression_matrix.npy (spots x genes)")
    p.add_argument("--gt_obs", required=True,
                   help="Path to combined_obs.npy (spot IDs matching gt_expr rows)")
    p.add_argument("--gene_list", required=True,
                   help="Path to all_shared_genes.txt (one gene per line, same order as gt_expr cols)")
    p.add_argument("--pretrained", default="/project_antwerp/assets/loki_ckpts/checkpoint.pt",
                   help="OmiCLIP checkpoint path")
    p.add_argument("--output_dir", default="./loki_predex_10fold",
                   help="Base output directory (fold_01, fold_02, ... created here)")
    p.add_argument("--pred_style", choices=["exact", "case_study"], default="case_study",
                   help="case_study=합 정규화만 (스케일링/softmax 없음, case study와 동일), exact=temp+softmax")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--top_k", type=int, default=None,
                   help="Top-k similar spots for PredEx (default: use all)")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_folds", type=int, default=10)
    args = p.parse_args()

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Load top genes and write .txt for loki_predex_exact
    top_genes = load_top_genes_from_csv(args.top_genes_csv, args.gene_column)
    hvg_file = out_base / "top_genes.txt"
    with open(hvg_file, "w") as f:
        f.write("\n".join(top_genes))
    print(f"Top genes: {len(top_genes)} -> {hvg_file}")

    # Load patches (filepath or img_path)
    patches_df = pd.read_csv(args.patches_csv)
    if "filepath" not in patches_df.columns and "img_path" not in patches_df.columns:
        raise ValueError("patches_csv must have 'filepath' or 'img_path' column")
    if "filepath" not in patches_df.columns:
        patches_df["filepath"] = patches_df["img_path"]
    if "obs_key" not in patches_df.columns:
        patches_df["obs_key"] = patches_df["filepath"].apply(
            lambda x: Path(x).stem if isinstance(x, str) else ""
        )

    gt_expr = np.load(args.gt_expr)
    gt_obs = np.load(args.gt_obs, allow_pickle=True)
    gene_list = open(args.gene_list).read().strip().split("\n")
    obs_to_idx = {b: i for i, b in enumerate(gt_obs)}

    patches_df = ensure_title_column(
        patches_df, gt_expr, gt_obs, gene_list, top_genes, topn=len(top_genes)
    )
    # Keep only patches that exist in gt_obs
    patches_df = patches_df[patches_df["obs_key"].isin(obs_to_idx)].reset_index(drop=True)
    n = len(patches_df)
    print(f"Patches with GT: {n}")

    if n < args.n_folds:
        raise ValueError(f"Not enough patches ({n}) for {args.n_folds}-fold CV")

    # 10-fold split (shuffle then split)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    fold_size = n // args.n_folds
    all_results = []

    for k in range(args.n_folds):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < args.n_folds - 1 else n
        val_idx = idx[val_start:val_end]
        train_idx = np.concatenate([idx[:val_start], idx[val_end:]])

        train_fold = patches_df.iloc[train_idx][["filepath", "title", "obs_key"]]
        val_fold = patches_df.iloc[val_idx][["filepath", "title", "obs_key"]]
        # loki_predex_exact expects 'filepath' and 'title'; obs_key used for GT lookup
        train_csv = out_base / f"fold_{k+1:02d}_train.csv"
        val_csv = out_base / f"fold_{k+1:02d}_val.csv"
        train_fold.to_csv(train_csv, index=False)
        val_fold.to_csv(val_csv, index=False)

        print(f"\n--- Fold {k+1}/{args.n_folds} ---")
        res = run_one_fold(k + 1, train_csv, val_csv, hvg_file, out_base, args)
        res["fold"] = k + 1
        all_results.append(res)

    # Aggregate
    summary = {
        "n_folds": args.n_folds,
        "n_patches": n,
        "n_genes": len(top_genes),
        "spot_pearson_mean": float(np.mean([r.get("spot_pearson_mean", np.nan) for r in all_results])),
        "spot_pearson_std": float(np.std([r.get("spot_pearson_mean", np.nan) for r in all_results])),
        "gene_pearson_mean": float(np.mean([r.get("gene_pearson_mean", np.nan) for r in all_results])),
        "gene_pearson_std": float(np.std([r.get("gene_pearson_mean", np.nan) for r in all_results])),
        "per_fold": all_results,
    }
    with open(out_base / "10fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("10-fold CV Summary (Loki PredEx, top-%d genes)" % len(top_genes))
    print("=" * 60)
    print("Spot Pearson:  mean = %.4f ± %.4f" % (summary["spot_pearson_mean"], summary["spot_pearson_std"]))
    print("Gene Pearson:  mean = %.4f ± %.4f" % (summary["gene_pearson_mean"], summary["gene_pearson_std"]))
    print("Results: %s" % out_base)
    print("=" * 60)


if __name__ == "__main__":
    main()
