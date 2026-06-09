#!/usr/bin/env python3
"""
OpenST 4-method comparison violin plot (compute gene-wise PCC from GT h5 + prediction CSV)

- GT: HDF5 with dataset 'expression' (spots x genes) and attr 'gene_names'
- Pred: CSV wide matrix (rows=spots, cols=genes), with an ID column for spot ids

Violin distributions (gene-wise PCC, values in [-1, 1]):
1) All shared genes
2) HEG top-300 (by mean GT expression across spots)
3) HVG top-300 (by variance on GT)   # scanpy 없이 대체
4) Oracle top-300 (by PCC among "All genes")

Example:
python Open-st_TIDO_Loki_compare.py \
  --pred_csv /project_antwerp/hbae/data/TIDO/Open-ST/tido_prediction_results.csv \
  --h5_path  /project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5 \
  --out_png  /project_antwerp/hbae/figures/Violin_plot/openst_comparison_violin_from_gt_pred.png
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def gene_wise_pcc(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """pred/true: (n_spots, n_genes). Return PCC per gene in [-1,1], NaN if undefined."""
    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    for g in range(pred.shape[1]):
        a = pred[:, g]
        b = true[:, g]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            continue
        aa = a[mask]
        bb = b[mask]
        da = aa - aa.mean()
        db = bb - bb.mean()
        denom = np.sqrt((da * da).sum()) * np.sqrt((db * db).sum())
        if denom > 1e-12:
            out[g] = (da * db).sum() / denom
    return out

def spot_wise_pcc(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """pred/true: (n_spots, n_genes). Return PCC per spot in [-1,1], NaN if undefined."""
    out = np.full(pred.shape[0], np.nan, dtype=np.float64)
    for i in range(pred.shape[0]):
        a = pred[i, :]
        b = true[i, :]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            continue
        aa = a[mask]
        bb = b[mask]
        da = aa - aa.mean()
        db = bb - bb.mean()
        denom = np.sqrt((da * da).sum()) * np.sqrt((db * db).sum())
        if denom > 1e-12:
            out[i] = (da * db).sum() / denom
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_csv", required=True, help="Prediction CSV (rows=spots, cols=genes).")
    p.add_argument("--pred_id_col", default="Unnamed: 0", help="Spot id column in pred csv.")
    p.add_argument("--h5_path", default="/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5")
    p.add_argument("--top_n", type=int, default=300)
    p.add_argument("--out_png", required=True)
    args = p.parse_args()

    pred_csv = Path(args.pred_csv)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # --- Load GT
    print("[1] Loading GT from h5...")
    with h5py.File(args.h5_path, "r") as f:
        gt_expr = f["expression"][:]  # (spots, genes)
        gt_col = f["coords_col"][:]   # (spots,)
        gt_row = f["coords_row"][:]   # (spots,)
        gene_names_bytes = f.attrs["gene_names"]
        gt_genes = [
            g.decode() if isinstance(g, (bytes, np.bytes_)) else str(g)
            for g in gene_names_bytes
        ]
    gt_expr = np.asarray(gt_expr, dtype=np.float32)
    gt_genes = list(gt_genes)
    print(f"  GT expr shape: {gt_expr.shape} (spots x genes)")
    print(f"  GT genes: {len(gt_genes)}")

    # --- Load Pred CSV
    print("[2] Loading prediction CSV...")
    df = pd.read_csv(pred_csv, low_memory=False)
    if args.pred_id_col not in df.columns:
        raise ValueError(f"pred_id_col '{args.pred_id_col}' not found in pred csv columns.")
    pred_ids = df[args.pred_id_col].astype(str).values
    pred_df = df.drop(columns=[args.pred_id_col]).select_dtypes(include=[np.number])
    print(f"  Pred shape (raw): {pred_df.shape} (spots x genes)")

    # --- Gene overlap
    print("[3] Aligning genes...")
    pred_genes = pred_df.columns.astype(str)
    common_genes = pred_genes.intersection(pd.Index(gt_genes))
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between pred CSV and GT h5 gene_names.")
    print(f"  Common genes: {len(common_genes)}")

    gt_gene_to_idx = {g: i for i, g in enumerate(gt_genes)}
    gt_col_idx = np.array([gt_gene_to_idx[g] for g in common_genes], dtype=int)

    pred_mat = pred_df[common_genes].to_numpy(dtype=np.float32)  # (spots, common_genes)
    gt_mat = gt_expr[:, gt_col_idx]                               # (spots, common_genes)

         # --- Reorder GT rows to match pred rows using (col,row) parsed from pred_ids
    def parse_col_row(s: str):
        # example: GSM7990099_primary_HNSCC_1000;1014_
        try:
            tail = s.split("_")[-2] if s.endswith("_") else s.split("_")[-1]
            # safer: find the token that contains ';'
            tok = next(t for t in s.split("_") if ";" in t)
            c, r = tok.split(";")[:2]
            return int(c), int(r)
        except Exception:
            return None
    
    gt_coord_to_i = {(int(c), int(r)): i for i, (c, r) in enumerate(zip(gt_col, gt_row))}
    
    gt_reidx = []
    missing = 0
    for sid in pred_ids:
        cr = parse_col_row(sid)
        if cr is None or cr not in gt_coord_to_i:
            missing += 1
            continue
        gt_reidx.append(gt_coord_to_i[cr])
    
    if missing > 0:
        raise ValueError(f"Failed to match {missing} pred rows to GT by (col,row). Example pred_id: {pred_ids[0]}")
    
    gt_mat = gt_mat[np.array(gt_reidx, dtype=int), :]
    print(f"[Reorder] GT reordered to pred by coords: {gt_mat.shape[0]} rows")

    # --- Spot alignment (handle off-by-one)
    print("[4] Aligning spots...")
    if pred_mat.shape[0] != gt_mat.shape[0]:
        if gt_mat.shape[0] == pred_mat.shape[0] + 1:
            row_var = gt_mat.var(axis=1)
            drop_i = int(np.argmin(row_var))
            print(f"[Fix] GT has 1 extra row. Dropping row index={drop_i} (min var={row_var[drop_i]:.6g})")
            gt_mat = np.delete(gt_mat, drop_i, axis=0)

        elif pred_mat.shape[0] == gt_mat.shape[0] + 1:
            row_var = pred_mat.var(axis=1)
            drop_i = int(np.argmin(row_var))
            print(f"[Fix] Pred has 1 extra row. Dropping row index={drop_i} (min var={row_var[drop_i]:.6g})")
            pred_mat = np.delete(pred_mat, drop_i, axis=0)
            pred_ids = np.delete(pred_ids, drop_i, axis=0)

        else:
            raise ValueError(f"Spot count mismatch: pred={pred_mat.shape[0]} vs gt={gt_mat.shape[0]}")

    print(f"  Final aligned shapes: pred={pred_mat.shape}, gt={gt_mat.shape}")

    # --- Compute PCC for ALL shared genes (must come first)
    print("[5] Computing gene-wise PCC...")
        # --- Quick sanity checks: scale & spot-wise PCC
    print("\n[Sanity] pred/gt value scale (common genes only):")
    print(f"  pred: mean={pred_mat.mean():.4f}, std={pred_mat.std():.4f}, min={pred_mat.min():.4f}, max={pred_mat.max():.4f}")
    print(f"  gt  : mean={gt_mat.mean():.4f}, std={gt_mat.std():.4f}, min={gt_mat.min():.4f}, max={gt_mat.max():.4f}")

    spot_corrs = spot_wise_pcc(pred_mat, gt_mat)
    spot_corrs = spot_corrs[np.isfinite(spot_corrs)]
    print("\n[Sanity] Spot-wise PCC (all common genes):")
    print(f"  n={len(spot_corrs)}, mean={spot_corrs.mean():.4f}, median={np.median(spot_corrs):.4f}, p10={np.quantile(spot_corrs,0.1):.4f}, p90={np.quantile(spot_corrs,0.9):.4f}")
    pcc_all_raw = gene_wise_pcc(pred_mat, gt_mat)
    ok = np.isfinite(pcc_all_raw)
    pcc_all = pcc_all_raw[ok]  # finite only

    # --- HEG top-N (by GT mean expression)
    gt_mean = gt_mat.mean(axis=0)
    heg_idx = np.argsort(gt_mean)[::-1][: args.top_n]
    pcc_heg = gene_wise_pcc(pred_mat[:, heg_idx], gt_mat[:, heg_idx])
    pcc_heg = pcc_heg[np.isfinite(pcc_heg)]

    # --- HVG top-N (variance; no scanpy)
    gt_var = gt_mat.var(axis=0)
    hvg_idx = np.argsort(gt_var)[::-1][: args.top_n]
    pcc_hvg = gene_wise_pcc(pred_mat[:, hvg_idx], gt_mat[:, hvg_idx])
    pcc_hvg = pcc_hvg[np.isfinite(pcc_hvg)]

    # --- Oracle top-N by PCC (from ALL genes PCC)
    oracle_idx = np.argsort(pcc_all)[::-1][: args.top_n]
    pcc_oracle = pcc_all[oracle_idx]

    methods = [
        {"label": f"All genes\n({len(pcc_all):,})", "data": pcc_all, "color": "#AED6F1"},
        {"label": f"HEG top-{args.top_n}\n(GT mean)", "data": pcc_heg, "color": "#A9DFBF"},
        {"label": f"HVG (variance)\ntop-{args.top_n}", "data": pcc_hvg, "color": "#F5CBA7"},
        {"label": f"Top PCC\ntop-{args.top_n} (oracle)", "data": pcc_oracle, "color": "#D2B4DE"},
    ]

    print("\n=== Summary ===")
    np.savez(
        "/project_antwerp/hbae/figures/Violin_plot/tido_pcc_4methods.npz",
        all_genes=pcc_all,
        heg=pcc_heg,
        hvg=pcc_hvg,
        oracle=pcc_oracle,
    )
    print("Saved TIDO npz: /project_antwerp/hbae/figures/Violin_plot/tido_pcc_4methods.npz")
    for m in methods:
        arr = np.asarray(m["data"])
        print(f"{m['label'].replace(chr(10),' ')}: n={len(arr)}, mean={arr.mean():.4f}, median={np.median(arr):.4f}")

    # --- Plot (same style)
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = list(range(1, len(methods) + 1))
    vdata = [m["data"] for m in methods]
    colors = [m["color"] for m in methods]

    vp = ax.violinplot(vdata, positions=pos, showmeans=True, showmedians=True, showextrema=True, widths=0.65)

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

    ax.set_ylim(-0.4, 1.0)
    ymax = ax.get_ylim()[1]

    for p_, m in zip(pos, methods):
        ax.text(
            p_, ymax, f"mean={np.mean(m['data']):.3f}",
            ha="center", va="bottom", fontsize=8,
            color="#111", fontweight="bold"
        )

    ax.set_xticks(pos)
    ax.set_xticklabels([m["label"] for m in methods], fontsize=10)
    ax.set_xlim(0.3, len(methods) + 0.7)
    ax.set_ylabel("Gene-wise PCC", fontsize=12)
    ax.axhline(0, color="red", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title(
        "Open-ST HNSCC External Validation\nGene-wise PCC — All Evaluation Schemes",
        fontsize=12, fontweight="bold", pad=20
    )

    handles = [
        mpatches.Patch(
            facecolor=m["color"],
            label=m["label"].replace("\n", " "),
            alpha=0.85, edgecolor="black", linewidth=0.5
        )
        for m in methods
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#ccc")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()