#!/usr/bin/env python3
"""
OpenST 4-method comparison violin plot (from a wide CSV matrix)

Input CSV expected format:
- Rows: spots/samples (any ID column allowed)
- Columns: genes (numeric values)

This script reduces each gene column to a single score (default: mean across rows),
then constructs four distributions:
1) All genes
2) HEG top-300 (first 300 genes from a provided HEG list file)
3) Scanpy HVG top-300 (first 300 genes from a provided HVG list file)
4) Oracle: Top 300 genes by score from the CSV

Saved as a 4-violin plot similar to the original NPY-based script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip()
        if not g or g.startswith("#"):
            continue
        genes.append(g)
    return genes


def gene_wise_pcc(pred: pd.DataFrame, true: pd.DataFrame) -> pd.Series:
    """
    Compute gene-wise Pearson correlation across rows (spots).
    Returns a Series indexed by gene with values in [-1, 1] (NaN when undefined).
    """
    common_genes = pred.columns.intersection(true.columns)
    if len(common_genes) == 0:
        raise ValueError("No overlapping gene columns between pred and true.")

    # Align columns and rows
    pred2 = pred[common_genes].astype(np.float64)
    true2 = true[common_genes].astype(np.float64)

    # Center
    px = pred2.to_numpy()
    tx = true2.to_numpy()

    # Handle NaNs: compute per-gene correlation using pairwise-valid entries
    out = {}
    for j, g in enumerate(common_genes):
        a = px[:, j]
        b = tx[:, j]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            out[g] = np.nan
            continue
        aa = a[mask]
        bb = b[mask]
        da = aa - aa.mean()
        db = bb - bb.mean()
        denom = np.sqrt((da * da).sum()) * np.sqrt((db * db).sum())
        out[g] = float((da * db).sum() / denom) if denom > 0 else np.nan
    return pd.Series(out, name="pcc")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create 4-method comparison violin plot from a wide gene-value CSV."
    )
    p.add_argument(
        "--csv",
        type=str,
        default="/shares/bioit-students/ir_students_2020/OpenST_HNSCC/tido_prediction_results.csv",
        help="Input CSV path (wide format: rows=spots, cols=genes).",
    )
    p.add_argument(
        "--id_col",
        type=str,
        default="Unnamed: 0",
        help="Non-numeric ID column to drop if present (default: 'Unnamed: 0').",
    )
    p.add_argument(
        "--reduce",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help=(
            "DEPRECATED path (only if --true_csv not provided): reduce per-spot values into a "
            "single per-gene score (default: mean)."
        ),
    )
    p.add_argument(
        "--true_csv",
        type=str,
        default=None,
        help=(
            "Ground-truth CSV (same wide format). If provided, compute gene-wise PCC(pred,true) "
            "and plot PCC distributions in [-1,1]."
        ),
    )
    p.add_argument(
        "--true_id_col",
        type=str,
        default="Unnamed: 0",
        help="ID column name in true CSV (default: 'Unnamed: 0').",
    )
    p.add_argument(
        "--heg_list",
        type=str,
        default="/home/students/hbae/data/loki_valgenes/outputs/high_expression_unique_genes.txt",
        help="Path to HEG gene list (one gene per line).",
    )
    p.add_argument(
        "--hvg_list",
        type=str,
        default="/home/students/hbae/0228_HVG_Finetune_gene_list_full.txt",
        help="Path to HVG gene list (one gene per line).",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=300,
        help="Top-N size for HEG/HVG/Oracle distributions (default: 300).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="/home/students/hbae/figures/openst_comparison_violin_from_csv.png",
        help="Output PNG path.",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    pred_raw = pd.read_csv(csv_path, low_memory=False)
    pred_id = None
    if args.id_col in pred_raw.columns:
        pred_id = pred_raw[args.id_col].astype(str)
        pred_raw = pred_raw.drop(columns=[args.id_col])
    pred = pred_raw.select_dtypes(include=[np.number])
    if pred.shape[1] == 0:
        raise ValueError("No numeric gene columns found in pred CSV after dropping id_col/selecting numeric dtypes.")

    if args.true_csv:
        true_path = Path(args.true_csv)
        if not true_path.exists():
            raise FileNotFoundError(f"Not found: {true_path}")
        true_raw = pd.read_csv(true_path, low_memory=False)
        if args.true_id_col in true_raw.columns:
            true_id = true_raw[args.true_id_col].astype(str)
            true_raw = true_raw.drop(columns=[args.true_id_col])
        else:
            true_id = None
        true = true_raw.select_dtypes(include=[np.number])
        if true.shape[1] == 0:
            raise ValueError("No numeric gene columns found in true CSV after dropping true_id_col/selecting numeric dtypes.")

        # Align rows if both have IDs
        if pred_id is not None and true_id is not None:
            pred_idx = pd.Index(pred_id)
            true_idx = pd.Index(true_id)
            common_rows = pred_idx.intersection(true_idx)
            if len(common_rows) == 0:
                raise ValueError("No overlapping row IDs between pred and true CSVs.")
            pred = pred.set_index(pred_idx).loc[common_rows].reset_index(drop=True)
            true = true.set_index(true_idx).loc[common_rows].reset_index(drop=True)
            print(f"Aligned rows by ID: {len(common_rows)} common spots")
        else:
            # Fallback: assume same order/length
            if len(pred) != len(true):
                raise ValueError(
                    f"Row count mismatch and cannot align by ID (pred={len(pred)}, true={len(true)})."
                )

        gene_score = gene_wise_pcc(pred, true).replace([np.inf, -np.inf], np.nan).dropna()
        y_label = "Gene-wise PCC"
        plot_title = "Open-ST HNSCC External Validation\nGene-wise PCC — 4-scheme comparison (from CSV)"
    else:
        # Fallback: NOT PCC (kept for convenience)
        if args.reduce == "mean":
            gene_score = pred.mean(axis=0, skipna=True)
        else:
            gene_score = pred.median(axis=0, skipna=True)
        gene_score = gene_score.replace([np.inf, -np.inf], np.nan).dropna()
        y_label = "Gene-wise score (reduced from CSV)"
        plot_title = "Open-ST HNSCC External Validation\n4-scheme comparison (from CSV; genes reduced)"

    if gene_score.empty:
        raise ValueError("Gene score vector is empty after NaN/Inf filtering.")

    all_genes = gene_score.index.tolist()

    heg_genes = read_gene_list(Path(args.heg_list))[: args.top_n]
    hvg_genes = read_gene_list(Path(args.hvg_list))[: args.top_n]

    def present(genes: list[str]) -> list[str]:
        return [g for g in genes if g in gene_score.index]

    heg_present = present(heg_genes)
    hvg_present = present(hvg_genes)

    oracle_genes = gene_score.sort_values(ascending=False).head(args.top_n).index.tolist()

    methods = [
        {
            "label": f"All genes\n({len(all_genes):,})",
            "data": gene_score.loc[all_genes].values,
            "color": "#AED6F1",
        },
        {
            "label": f"HEG top-{args.top_n}\n(Loki paper)",
            "data": gene_score.loc[heg_present].values,
            "color": "#A9DFBF",
        },
        {
            "label": f"Scanpy HVG\ntop-{args.top_n}",
            "data": gene_score.loc[hvg_present].values,
            "color": "#F5CBA7",
        },
        {
            "label": f"Top score\ntop-{args.top_n} (oracle)",
            "data": gene_score.loc[oracle_genes].values,
            "color": "#D2B4DE",
        },
    ]

    # Print quick summary
    if len(heg_present) < args.top_n:
        print(
            f"WARNING: HEG list overlap is small: {len(heg_present)}/{args.top_n} genes present in CSV."
        )
    if len(hvg_present) < args.top_n:
        print(
            f"WARNING: HVG list overlap is small: {len(hvg_present)}/{args.top_n} genes present in CSV."
        )
    for m in methods:
        data = m["data"]
        print(f"{m['label'].replace(chr(10), ' ')}: n={len(data)}, mean={float(np.mean(data)):.4f}")

    # Plot (mirrors original styling)
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = list(range(1, len(methods) + 1))
    vdata = [m["data"] for m in methods]
    colors = [m["color"] for m in methods]

    vp = ax.violinplot(
        vdata,
        positions=pos,
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.65,
    )

    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.80)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)

    for k in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if k in vp:
            vp[k].set_color("black")
            vp[k].set_linewidth(1.6 if k == "cmeans" else 1.0)
            if k == "cmeans":
                vp[k].set_linestyle("--")

    if args.true_csv:
        ax.set_ylim(-0.4, 1.0)
        ymax_plot = ax.get_ylim()[1]
    else:
        ymin = float(np.nanmin(gene_score.values))
        ymax = float(np.nanmax(gene_score.values))
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.1
        ax.set_ylim(ymin - pad, ymax + pad)
        ymax_plot = ax.get_ylim()[1]

    for p_, m in zip(pos, methods):
        ax.text(
            p_,
            ymax_plot,
            f"mean={float(np.mean(m['data'])):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111",
            fontweight="bold",
        )

    ax.set_xticks(pos)
    ax.set_xticklabels([m["label"] for m in methods], fontsize=10)
    ax.set_xlim(0.3, len(methods) + 0.7)
    ax.set_ylabel(y_label, fontsize=12)
    ax.axhline(0, color="red", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title(
        plot_title,
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    handles = [
        mpatches.Patch(
            facecolor=m["color"],
            label=m["label"].replace("\n", " "),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        for m in methods
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#ccc")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

