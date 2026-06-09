#!/usr/bin/env python3
"""
Extract H&E patches aligned to a fixed physical size on tissue (hires), then resize.

Uses Visium convention: spot diameter 55 µm at tissue →
  MPP_fullres = 55 / spot_diameter_fullres (from scalefactors_json.json).

Crops are taken on the **hires** PNG (Data select/.../spatial/tissue_hires_image.png), so the
relevant scale is hires space:
  MPP_hires = MPP_fullres / tissue_hires_scalef
  crop_px (hires) = round(target_um / MPP_hires)
then resize to (output_px, output_px) with LANCZOS.

This is not the same as target_um / MPP_fullres: that fullres pixel count is only valid if you
crop the full-resolution source image, not the downsampled hires export.

GSE252265 without obsm['spatial'] uses tissue_positions_list.csv next to the hires image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SPOT_DIAMETER_UM = 55.0  # standard Visium v1

DATA_ROOT = Path("/home/students/hbae/data")
DATA_SELECT = DATA_ROOT / "Data select"
PROCESSED_DEFAULT = DATA_ROOT / "Processed_Data"
OUT_DEFAULT = DATA_ROOT / "Processed_Data_65um_224"


def resolve_data_select_paths(dataset: str, sample: str) -> tuple[Path, Path, Path | None]:
    """
    Returns (hires_png, scalefactors_json, tissue_positions_csv_or_None).
    Positions file may be tissue_positions_list.csv or tissue_positions.csv.
    """
    if dataset == "GSE220978":
        base = DATA_SELECT / "GSE220978" / sample / "spatial"
    elif dataset == "Queensland":
        base = DATA_SELECT / "Queensland" / sample / "spatial"
    elif dataset == "Queensland P5 Data":
        base = DATA_SELECT / "Queensland P5 Data" / sample / "spatial"
    elif dataset == "Zenodo":
        base = DATA_SELECT / "Zenodo" / sample / "spatial"
    else:
        base = DATA_SELECT / dataset / sample / "spatial"

    img = base / "tissue_hires_image.png"
    sf = base / "scalefactors_json.json"
    pos = None
    for name in ("tissue_positions_list.csv", "tissue_positions.csv"):
        p = base / name
        if p.exists():
            pos = p
            break
    return img, sf, pos


def load_scalefactors(sf_path: Path) -> dict[str, Any]:
    with open(sf_path) as f:
        return json.load(f)


def spatial_coords_for_sample(
    adata: ad.AnnData,
    positions_csv: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution pixel coords: pxl_col, pxl_row (matches 10x CSV columns)."""
    if "spatial" in adata.obsm and adata.obsm["spatial"].size:
        s = np.asarray(adata.obsm["spatial"], dtype=float)
        if s.ndim != 2 or s.shape[1] != 2:
            raise ValueError("obsm['spatial'] must be (n_spots, 2)")
        return s[:, 0], s[:, 1]

    if positions_csv is None or not positions_csv.exists():
        raise FileNotFoundError(
            "No obsm['spatial'] and no positions CSV; cannot place spots."
        )

    pos = pd.read_csv(positions_csv)
    row_col = "pxl_row_in_fullres"
    col_col = "pxl_col_in_fullres"
    if row_col not in pos.columns:
        raise ValueError(f"Unexpected positions columns: {pos.columns.tolist()}")
    pos = pos.set_index("barcode")
    rows: list[pd.Series] = []
    for b in adata.obs_names:
        bs = str(b)
        if bs in pos.index:
            rows.append(pos.loc[bs])
        elif bs.endswith("-1") and bs[:-2] in pos.index:
            rows.append(pos.loc[bs[:-2]])
        else:
            raise KeyError(f"barcode {bs!r} not in {positions_csv}")

    sub = pd.DataFrame(rows, index=adata.obs_names)
    col = sub[col_col].values.astype(float)
    row = sub[row_col].values.astype(float)
    return col, row


def crop_resize_patch(
    img: np.ndarray,
    xc: float,
    yc: float,
    crop_px: int,
    out_px: int,
) -> np.ndarray | None:
    """img: HxWxC uint8. xc=x col, yc=y row. Returns HxWxC or None if fully OOB."""
    h, w = img.shape[:2]
    half = crop_px / 2.0
    x1 = int(np.round(xc - half))
    y1 = int(np.round(yc - half))
    x2 = x1 + crop_px
    y2 = y1 + crop_px
    if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
        patch = img[y1:y2, x1:x2]
    else:
        if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
            return None
        px1, px2 = max(0, x1), min(w, x2)
        py1, py2 = max(0, y1), min(h, y2)
        ph, pw = py2 - py1, px2 - px1
        if ph < 1 or pw < 1:
            return None
        patch = np.zeros((crop_px, crop_px, img.shape[2]), dtype=img.dtype)
        sx = px1 - x1
        sy = py1 - y1
        patch[sy : sy + ph, sx : sx + pw] = img[py1:py2, px1:px2]
    pil = Image.fromarray(patch)
    pil = pil.resize((out_px, out_px), Image.Resampling.LANCZOS)
    return np.asarray(pil)


def iter_processed_samples(processed_root: Path) -> list[tuple[str, str, Path]]:
    """(dataset_name, sample_name, sample_dir) — only dirs containing st_norm.h5ad."""
    out: list[tuple[str, str, Path]] = []
    for dataset_dir in sorted(processed_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name in ("training_data", "training_data_excluding_GSE220978_and_19h1257"):
            continue
        if dataset_dir.suffix == ".csv" or dataset_dir.name.endswith(".csv"):
            continue
        for sample_dir in sorted(dataset_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if (sample_dir / "st_norm.h5ad").exists():
                out.append((dataset_dir.name, sample_dir.name, sample_dir))
    return out


def process_one_sample(
    dataset: str,
    sample: str,
    sample_dir: Path,
    out_root: Path,
    target_um: float,
    out_px: int,
    overwrite: bool,
) -> dict[str, Any]:
    st_path = sample_dir / "st_norm.h5ad"
    adata = ad.read_h5ad(st_path)

    img_path, sf_path, pos_csv = resolve_data_select_paths(dataset, sample)
    if not img_path.exists():
        raise FileNotFoundError(f"Hires image missing: {img_path}")
    if not sf_path.exists():
        raise FileNotFoundError(f"scalefactors missing: {sf_path}")

    sf = load_scalefactors(sf_path)
    spot_px = float(sf["spot_diameter_fullres"])
    th = float(sf["tissue_hires_scalef"])
    mpp_fr = SPOT_DIAMETER_UM / spot_px
    mpp_hires = mpp_fr / th
    crop_px = max(1, int(round(target_um / mpp_hires)))

    col_fr, row_fr = spatial_coords_for_sample(adata, pos_csv)
    sf_scale = th
    xc = col_fr * sf_scale
    yc = row_fr * sf_scale

    img = np.array(Image.open(img_path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    out_dir = out_root / dataset / sample / "patches"
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for p in out_dir.glob("*.png"):
            p.unlink()

    n_ok = 0
    n_skip = 0
    for i, bc in enumerate(tqdm(adata.obs_names, desc=f"{dataset}/{sample}", leave=False)):
        arr = crop_resize_patch(img, float(xc[i]), float(yc[i]), crop_px, out_px)
        if arr is None:
            n_skip += 1
            continue
        fn = str(bc).replace("/", "_") + ".png"
        Image.fromarray(arr).save(out_dir / fn)
        n_ok += 1

    meta = {
        "dataset": dataset,
        "sample": sample,
        "spot_diameter_fullres_px": spot_px,
        "tissue_hires_scalef": th,
        "mpp_fullres_um_per_px": mpp_fr,
        "mpp_hires_um_per_px": mpp_hires,
        "crop_px_hires": crop_px,
        "output_px": out_px,
        "target_um": target_um,
        "hires_image_path": str(img_path),
        "n_spots": adata.n_obs,
        "n_patches_written": n_ok,
        "n_skipped_oob": n_skip,
        "patches_dir": str(out_dir),
    }
    pd.DataFrame([meta]).to_csv(out_dir.parent / "patch_extraction_65um_meta.csv", index=False)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="MPP-unified patch extraction for Visium cohorts.")
    ap.add_argument("--processed-root", type=Path, default=PROCESSED_DEFAULT)
    ap.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--target-um", type=float, default=65.0)
    ap.add_argument("--output-px", type=int, default=224)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional 'Dataset/sample' (e.g. GSE208253/GSM6339631_s1) to process one sample.",
    )
    args = ap.parse_args()

    samples = iter_processed_samples(args.processed_root)
    if args.only:
        ds, _, rest = args.only.partition("/")
        if not rest:
            print("--only must be Dataset/sample", file=sys.stderr)
            sys.exit(1)
        samples = [(d, s, p) for d, s, p in samples if d == ds and s == rest]
        if not samples:
            print(f"No sample matched --only {args.only!r}", file=sys.stderr)
            sys.exit(1)

    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str]] = []

    for dataset, sample, sample_dir in tqdm(samples, desc="samples"):
        try:
            meta = process_one_sample(
                dataset,
                sample,
                sample_dir,
                args.out_root,
                args.target_um,
                args.output_px,
                args.overwrite,
            )
            rows.append(meta)
        except Exception as e:
            errors.append((dataset, sample, repr(e)))

    manifest = args.out_root / "manifest_mpp_unified.csv"
    if rows:
        pd.DataFrame(rows).to_csv(manifest, index=False)
        print(f"Wrote manifest: {manifest}")
    if errors:
        err_path = args.out_root / "manifest_errors.csv"
        pd.DataFrame(errors, columns=["dataset", "sample", "error"]).to_csv(err_path, index=False)
        print(f"{len(errors)} errors logged to {err_path}", file=sys.stderr)
        for d, s, e in errors[:15]:
            print(f"  {d}/{s}: {e}", file=sys.stderr)
        if not rows:
            sys.exit(1)


if __name__ == "__main__":
    main()
