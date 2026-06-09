#!/usr/bin/env python3
"""
Export Appendix A gene lists for Open-ST and TCGA external validation.

Outputs (under --out_dir):
  README.md
  tables/Table_A1_OpenST_summary.tsv
  tables/Table_A2_TCGA_summary.tsv
  openst/OpenST_shared_*_genes.txt
  openst/OpenST_HEG300_genes.txt
  openst/OpenST_HVG300_genes.txt
  openst/OpenST_oracle300_genes.txt          (if --pred_csv provided)
  openst/OpenST_gene_set_membership.tsv
  tcga/TCGA_shared_*_genes.txt

Open-ST shared universe:
  intersection(Open-ST catalogue from h5 gene_names, Visium HVG training vocabulary)

Open-ST HEG/HVG top-300:
  computed on GT expression in h5 (all spots), restricted to shared universe

Open-ST oracle top-300:
  top gene-wise PCC vs GT (requires --pred_csv, optional coord alignment)

TCGA shared universe:
  intersection(training HVG vocabulary, genes in ref_file bulk RNA columns)

Example (project_antwerp):
  python export_appendix_gene_lists.py \\
    --out_dir /project_antwerp/hbae/appendix_gene_lists \\
    --h5_path /project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5 \\
    --train_hvg /project_antwerp/hbae/data/0317_hvg_2000_list.txt \\
    --ref_file /project_antwerp/hbae/ref_file.csv \\
    --pred_csv /project_antwerp/hbae/data/TIDO/Open-ST/tido_prediction_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def read_gene_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return [str(x).strip() for x in arr if str(x).strip()]
    genes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip()
        if g and not g.startswith("#"):
            genes.append(g)
    return genes


def write_gene_list(path: Path, genes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(genes) + "\n", encoding="utf-8")


def load_openst_genes(h5_path: Path) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        raw = f.attrs["gene_names"]
    return [g.decode() if isinstance(g, (bytes, np.bytes_)) else str(g) for g in raw]


def load_openst_gt(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        expr = np.asarray(f["expression"][:], dtype=np.float32)
        col = np.asarray(f["coords_col"][:])
        row = np.asarray(f["coords_row"][:])
    return expr, col, row


def parse_col_row(s: str) -> tuple[int, int] | None:
    try:
        tok = next(t for t in str(s).split("_") if ";" in t)
        c, r = tok.split(";")[:2]
        return int(c), int(r)
    except Exception:
        return None


def align_gt_to_pred(
    gt_expr: np.ndarray,
    gt_col: np.ndarray,
    gt_row: np.ndarray,
    pred_ids: np.ndarray,
    gene_col_idx: np.ndarray,
) -> np.ndarray:
    """Return GT matrix (n_pred_spots, n_genes) aligned to pred row order."""
    gt_coord_to_i = {(int(c), int(r)): i for i, (c, r) in enumerate(zip(gt_col, gt_row))}
    gt_reidx: list[int] = []
    missing = 0
    for sid in pred_ids:
        cr = parse_col_row(sid)
        if cr is None or cr not in gt_coord_to_i:
            missing += 1
            continue
        gt_reidx.append(gt_coord_to_i[cr])
    if missing:
        raise ValueError(f"{missing} pred spots could not be matched to GT (col,row).")
    gt_sub = gt_expr[np.array(gt_reidx, dtype=int)][:, gene_col_idx]
    return gt_sub


def select_heg_idx(gt_mat: np.ndarray, top_n: int) -> np.ndarray:
    gt_mean = gt_mat.mean(axis=0)
    return np.argsort(gt_mean)[::-1][:top_n]


def select_hvg_idx(gt_mat: np.ndarray, genes: list[str], top_n: int, use_scanpy: bool) -> np.ndarray:
    if use_scanpy:
        try:
            import scanpy as sc

            ad = sc.AnnData(X=gt_mat.copy())
            ad.var_names = genes
            sc.pp.highly_variable_genes(ad, n_top_genes=top_n, flavor="seurat")
            mask = ad.var["highly_variable"].values
            idx = np.where(mask)[0]
            if len(idx) >= top_n:
                return idx[:top_n]
        except ImportError:
            pass
    gt_var = gt_mat.var(axis=0)
    return np.argsort(gt_var)[::-1][:top_n]


def gene_wise_pcc(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    for g in range(pred.shape[1]):
        a, b = pred[:, g], true[:, g]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            continue
        aa, bb = a[mask], b[mask]
        da, db = aa - aa.mean(), bb - bb.mean()
        denom = np.sqrt((da * da).sum()) * np.sqrt((db * db).sum())
        if denom > 1e-12:
            out[g] = (da * db).sum() / denom
    return out


def ensure_out_dirs(out_dir: Path) -> None:
    """Create output subdirectories (out_dir itself must exist or be created by caller)."""
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "openst").mkdir(parents=True, exist_ok=True)
    (out_dir / "tcga").mkdir(parents=True, exist_ok=True)


def export_openst(
    out_dir: Path,
    h5_path: Path,
    train_hvg: list[str],
    top_n: int,
    pred_csv: Path | None,
    pred_id_col: str,
    use_scanpy: bool,
) -> dict:
    ensure_out_dirs(out_dir)
    openst_dir = out_dir / "openst"

    openst_genes = load_openst_genes(h5_path)
    train_set = set(train_hvg)
    shared = sorted(g for g in openst_genes if g in train_set)

    write_gene_list(openst_dir / f"OpenST_shared_{len(shared)}_genes.txt", shared)

    gt_expr, gt_col, gt_row = load_openst_gt(h5_path)
    openst_to_i = {g: i for i, g in enumerate(openst_genes)}
    shared_idx = np.array([openst_to_i[g] for g in shared], dtype=int)
    gt_shared = gt_expr[:, shared_idx]

    heg_idx = select_heg_idx(gt_shared, top_n)
    hvg_idx = select_hvg_idx(gt_shared, shared, top_n, use_scanpy)

    heg_genes = [shared[i] for i in heg_idx]
    hvg_genes = [shared[i] for i in hvg_idx]
    write_gene_list(openst_dir / "OpenST_HEG300_genes.txt", heg_genes)
    write_gene_list(openst_dir / "OpenST_HVG300_genes.txt", hvg_genes)

    oracle_genes: list[str] | None = None
    hvg_method = "scanpy_seurat" if use_scanpy else "variance_fallback"

    if pred_csv is not None:
        df = pd.read_csv(pred_csv, low_memory=False)
        pred_ids = df[pred_id_col].astype(str).values
        pred_df = df.drop(columns=[pred_id_col]).select_dtypes(include=[np.number])
        pred_shared = pred_df[[g for g in shared if g in pred_df.columns]].to_numpy(dtype=np.float32)
        gt_aligned = align_gt_to_pred(gt_expr, gt_col, gt_row, pred_ids, shared_idx)
        if pred_shared.shape[0] != gt_aligned.shape[0]:
            raise ValueError("pred/GT row count mismatch after coord alignment.")
        pcc = gene_wise_pcc(pred_shared, gt_aligned)
        ok = np.isfinite(pcc)
        pcc_ok = pcc[ok]
        genes_ok = np.array(shared)[ok]
        oracle_idx = np.argsort(pcc_ok)[::-1][:top_n]
        oracle_genes = genes_ok[oracle_idx].tolist()
        write_gene_list(openst_dir / "OpenST_oracle300_genes.txt", oracle_genes)

    # membership table
    heg_set, hvg_set = set(heg_genes), set(hvg_genes)
    oracle_set = set(oracle_genes) if oracle_genes else set()
    rows = []
    for g in shared:
        rows.append(
            {
                "gene": g,
                "in_shared_universe": 1,
                "in_HEG300": int(g in heg_set),
                "in_HVG300": int(g in hvg_set),
                "in_oracle300": int(g in oracle_set),
            }
        )
    pd.DataFrame(rows).to_csv(openst_dir / "OpenST_gene_set_membership.tsv", sep="\t", index=False)

    # overlap stats for optional figure
    overlap = {
        "n_openst_catalogue": len(openst_genes),
        "n_train_hvg": len(train_hvg),
        "n_shared": len(shared),
        "n_HEG300": len(heg_genes),
        "n_HVG300": len(hvg_genes),
        "n_oracle300": len(oracle_genes) if oracle_genes else 0,
        "HEG_and_HVG": len(heg_set & hvg_set),
        "HEG_and_oracle": len(heg_set & oracle_set) if oracle_set else None,
        "HVG_and_oracle": len(hvg_set & oracle_set) if oracle_set else None,
    }
    (openst_dir / "OpenST_overlap_stats.json").write_text(
        json.dumps(overlap, indent=2), encoding="utf-8"
    )

    summary = pd.DataFrame(
        [
            {
                "strategy": "All shared genes",
                "n_genes": len(shared),
                "selection_rule": "Open-ST catalogue ∩ Visium HVG training vocabulary",
            },
            {
                "strategy": "HEG top-300",
                "n_genes": top_n,
                "selection_rule": f"Top {top_n} by mean GT expression on Open-ST validation spots (shared universe)",
            },
            {
                "strategy": "HVG top-300",
                "n_genes": top_n,
                "selection_rule": f"Top {top_n} HVG on GT ({hvg_method}) within shared universe",
            },
            {
                "strategy": "Oracle top-300",
                "n_genes": top_n if oracle_genes else 0,
                "selection_rule": "Top 300 by post-hoc gene-wise PCC vs GT (requires predictions)",
            },
        ]
    )
    summary.to_csv(out_dir / "tables" / "Table_A1_OpenST_summary.tsv", sep="\t", index=False)

    return overlap


def export_tcga(out_dir: Path, train_hvg: list[str], ref_file: Path) -> int:
    ensure_out_dirs(out_dir)
    tcga_dir = out_dir / "tcga"

    ref_df = pd.read_csv(ref_file, index_col=0, nrows=1)
    rna_genes = [c.replace("rna_", "") for c in ref_df.columns if str(c).startswith("rna_")]
    train_set = set(train_hvg)
    shared = sorted(g for g in train_hvg if g in rna_genes)

    write_gene_list(tcga_dir / f"TCGA_shared_{len(shared)}_genes.txt", shared)

    summary = pd.DataFrame(
        [
            {
                "strategy": "All shared genes (no subsetting)",
                "n_genes": len(shared),
                "selection_rule": "HVG training vocabulary ∩ TCGA bulk RNA-seq panel (ref_file rna_* columns)",
            },
        ]
    )
    summary.to_csv(out_dir / "tables" / "Table_A2_TCGA_summary.tsv", sep="\t", index=False)
    return len(shared)


def write_readme(out_dir: Path, openst_ok: bool, tcga_n: int | None) -> None:
    lines = [
        "# Appendix A — Gene lists",
        "",
        "Plain-text gene lists (one gene symbol per line) for external validation.",
        "",
        "## Open-ST",
    ]
    if openst_ok:
        lines += [
            "- `openst/OpenST_shared_*_genes.txt` — evaluation universe",
            "- `openst/OpenST_HEG300_genes.txt`",
            "- `openst/OpenST_HVG300_genes.txt`",
            "- `openst/OpenST_oracle300_genes.txt` (if predictions were provided)",
            "- `openst/OpenST_gene_set_membership.tsv` — 0/1 membership per gene",
            "- `tables/Table_A1_OpenST_summary.tsv`",
        ]
    else:
        lines.append("- *(not generated — provide `--h5_path` on project_antwerp)*")

    lines += ["", "## TCGA bulk RNA-seq"]
    if tcga_n is not None:
        lines += [
            f"- `tcga/TCGA_shared_{tcga_n}_genes.txt`",
            "- `tables/Table_A2_TCGA_summary.tsv`",
        ]
    else:
        lines.append("- *(not generated)*")

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Export Appendix A gene lists.")
    p.add_argument(
        "--out_dir",
        type=str,
        default="/project_antwerp/hbae/appendix_gene_lists",
    )
    p.add_argument(
        "--h5_path",
        type=str,
        default="/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5",
        help="Open-ST GT HDF5 (required for Open-ST lists unless --skip_openst).",
    )
    p.add_argument(
        "--train_hvg",
        type=str,
        default="/project_antwerp/hbae/data/0317_hvg_2000_list.txt",
    )
    p.add_argument(
        "--ref_file",
        type=str,
        default="/project_antwerp/hbae/ref_file.csv",
        help="TCGA bulk RNA reference (rna_* columns).",
    )
    p.add_argument("--pred_csv", type=str, default=None, help="Optional: for oracle gene list.")
    p.add_argument("--pred_id_col", type=str, default="Unnamed: 0")
    p.add_argument("--top_n", type=int, default=300)
    p.add_argument("--skip_openst", action="store_true")
    p.add_argument("--skip_tcga", action="store_true")
    p.add_argument(
        "--use_scanpy_hvg",
        action="store_true",
        help="Use Scanpy seurat HVG; otherwise variance top-N on GT.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_out_dirs(out_dir)
    train_hvg = read_gene_list(Path(args.train_hvg))

    openst_ok = False
    if not args.skip_openst:
        h5_path = Path(args.h5_path)
        if not h5_path.exists():
            print(f"[WARN] Open-ST h5 not found: {h5_path}")
            print("       Skipping Open-ST export. Run on project_antwerp or set --h5_path.")
        else:
            pred_csv = Path(args.pred_csv) if args.pred_csv else None
            overlap = export_openst(
                out_dir,
                h5_path,
                train_hvg,
                args.top_n,
                pred_csv,
                args.pred_id_col,
                args.use_scanpy_hvg,
            )
            openst_ok = True
            print("[Open-ST]", json.dumps(overlap, indent=2))

    tcga_n = None
    if not args.skip_tcga:
        ref_file = Path(args.ref_file)
        if not ref_file.exists():
            print(f"[WARN] ref_file not found: {ref_file}")
        else:
            tcga_n = export_tcga(out_dir, train_hvg, ref_file)
            print(f"[TCGA] shared genes: {tcga_n}")

    write_readme(out_dir, openst_ok, tcga_n)
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
