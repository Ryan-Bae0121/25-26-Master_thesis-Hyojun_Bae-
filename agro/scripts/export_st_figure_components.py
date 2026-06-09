#!/usr/bin/env python3
"""Export Visium ST figure components parallel to TCGA slide export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image, ImageDraw

from make_figure2_1_combined import (
    load_spots,
    otsu_tissue_mask,
    overlay_mask_contour,
)

# GSE252265 aggregated Space Ranger: gem suffix on barcode (1..8).
GEM_TO_SAMPLE = {
    1: "GSM7998252",
    2: "GSM7998253",
    3: "GSM7998254",
    4: "GSM7998255",
    5: "GSM7998256",
    6: "GSM7998257",
    7: "GSM7998258",
    8: "GSM7998259",
}


def filter_spots_for_sample(
    df: pd.DataFrame, sample_id: str | None, barcode_suffix: int | None
) -> pd.DataFrame:
    """Keep one sample when positions file is GSE252265 aggregated (8 gem groups)."""
    if barcode_suffix is not None:
        suf = int(barcode_suffix)
        return df[df["barcode"].str.endswith(f"-{suf}")].copy()
    if sample_id and sample_id in GEM_TO_SAMPLE.values():
        gem = next(k for k, v in GEM_TO_SAMPLE.items() if v == sample_id)
        return df[df["barcode"].str.endswith(f"-{gem}")].copy()
    return df


def load_spots_from_h5ad(h5ad_path: Path, scale_path: Path) -> pd.DataFrame:
    """Build spot table from st.h5ad (GSE208253 etc. when positions CSV is absent)."""
    with open(scale_path) as f:
        scalef = json.load(f)["tissue_hires_scalef"]
    ad = sc.read_h5ad(h5ad_path)
    if "spatial" not in ad.obsm:
        raise ValueError(f"No obsm['spatial'] in {h5ad_path}")
    sp = np.asarray(ad.obsm["spatial"])
    df = pd.DataFrame(
        {
            "barcode": ad.obs_names.astype(str),
            "in_tissue": ad.obs["in_tissue"].astype(int),
            "pxl_col_fullres": sp[:, 0],
            "pxl_row_fullres": sp[:, 1],
        }
    )
    if "array_row" in ad.obs.columns:
        df["array_row"] = ad.obs["array_row"].values
        df["array_col"] = ad.obs["array_col"].values
    df["x_hires"] = df["pxl_col_fullres"] * scalef
    df["y_hires"] = df["pxl_row_fullres"] * scalef
    return df


def rasterize_in_tissue_mask(
    df: pd.DataFrame,
    height: int,
    width: int,
    radius_px: float,
) -> np.ndarray:
    """Boolean mask: True where Visium in-tissue spots cover."""
    mask = np.zeros((height, width), dtype=bool)
    in_t = df[df["in_tissue"].astype(int) == 1]
    r = max(2, int(round(radius_px)))
    for x, y in zip(in_t["x_hires"], in_t["y_hires"]):
        xi, yi = int(round(x)), int(round(y))
        y0, y1 = max(0, yi - r), min(height, yi + r + 1)
        x0, x1 = max(0, xi - r), min(width, xi + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - xi) ** 2 + (yy - yi) ** 2 <= r**2
        mask[y0:y1, x0:x1] |= disk
    return mask


def background_crop_from_he(
    he: np.ndarray,
    size: int = 144,
    bright_min: float = 230.0,
) -> np.ndarray | None:
    """Bright glass crop on hires H&E (same logic as WSI background, no OpenSlide)."""
    tissue = otsu_tissue_mask(he)
    gray = (
        0.299 * he[..., 0] + 0.587 * he[..., 1] + 0.114 * he[..., 2]
    ).astype(np.float32)
    glass = (~tissue) & (gray >= bright_min)
    if not glass.any():
        non_tissue = ~tissue
        if not non_tissue.any():
            return None
        thr = float(np.percentile(gray[non_tissue], 95))
        glass = non_tissue & (gray >= thr)
    if not glass.any():
        glass = ~tissue
    cy, cx = np.unravel_index(int(np.argmax(np.where(glass, gray, -1.0))), gray.shape)
    half = size // 2
    y0 = max(0, min(cy - half, he.shape[0] - size))
    x0 = max(0, min(cx - half, he.shape[1] - size))
    return he[y0 : y0 + size, x0 : x0 + size].copy()


def resize_max(he: np.ndarray, max_px: int) -> tuple[np.ndarray, float]:
    h, w = he.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    if scale >= 1.0:
        return he, 1.0
    dw = max(1, int(round(w * scale)))
    dh = max(1, int(round(h * scale)))
    out = np.array(Image.fromarray(he).resize((dw, dh), Image.Resampling.LANCZOS))
    return out, scale


def export_st(
    out_dir: Path,
    hires: Path,
    spots: Path | None,
    scale: Path,
    patches_dir: Path,
    st_h5ad: Path | None,
    spots_h5ad: Path | None = None,
    max_px: int = 2000,
    patch_size: int = 224,
    sample_id: str | None = None,
    barcode_suffix: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    he_full = np.array(Image.open(hires).convert("RGB"))
    H, W = he_full.shape[:2]
    if spots_h5ad and Path(spots_h5ad).exists():
        df_all = load_spots_from_h5ad(Path(spots_h5ad), Path(scale))
        spots_source = str(spots_h5ad)
    elif spots and Path(spots).exists():
        df_all = load_spots(str(spots), str(scale))
        spots_source = str(spots)
    else:
        raise FileNotFoundError(f"No spots CSV or --spots_h5ad for {out_dir.name}")
    n_all_in = int((df_all["in_tissue"].astype(int) == 1).sum())
    df = filter_spots_for_sample(df_all, sample_id or out_dir.name, barcode_suffix)

    meta_path = patches_dir.parent / "patch_extraction_65um_meta.csv"
    spot_diam_hires = 16.0
    if meta_path.exists():
        meta = pd.read_csv(meta_path).iloc[0]
        spot_diam_hires = float(meta["spot_diameter_fullres_px"]) * float(
            meta["tissue_hires_scalef"]
        )

    mask_full = rasterize_in_tissue_mask(df, H, W, radius_px=spot_diam_hires / 2.0)
    he, disp_scale = resize_max(he_full, max_px)
    mask_disp = (
        np.array(
            Image.fromarray(mask_full.astype(np.uint8) * 255).resize(
                (he.shape[1], he.shape[0]), Image.Resampling.NEAREST
            )
        )
        > 127
    )
    overlay = overlay_mask_contour(he, mask_disp, alpha=0.25)

    Image.fromarray(he).save(out_dir / "01_original_he_hires.png")
    Image.fromarray((mask_disp.astype(np.uint8) * 255)).save(out_dir / "02_mask_in_tissue.png")
    Image.fromarray(overlay).save(out_dir / "03_overlay.png")

    bg = background_crop_from_he(he_full, size=patch_size)
    if bg is None:
        out_t = df[df["in_tissue"].astype(int) == 0]
        if len(out_t):
            row = out_t.iloc[len(out_t) // 2]
            cx, cy = int(row["x_hires"]), int(row["y_hires"])
            half = patch_size // 2
            bg = he_full[
                max(0, cy - half) : cy - half + patch_size,
                max(0, cx - half) : cx - half + patch_size,
            ]
    if bg is not None:
        if bg.shape[0] != patch_size or bg.shape[1] != patch_size:
            bg = np.array(
                Image.fromarray(bg).resize((patch_size, patch_size), Image.Resampling.LANCZOS)
            )
        Image.fromarray(bg).save(out_dir / "04_background_tile.png")

    # low / high: total expression per QC spot (parallel to tissue_frac ranking)
    low_bc = high_bc = None
    low_val = high_val = None
    if st_h5ad and Path(st_h5ad).exists():
        ad = sc.read_h5ad(st_h5ad)
        X = ad.X
        totals = np.array(X.sum(axis=1)).flatten() if hasattr(X, "toarray") else np.asarray(X.sum(axis=1)).flatten()
        i_lo = int(np.argmin(totals))
        i_hi = int(np.argmax(totals))
        low_bc, high_bc = ad.obs_names[i_lo], ad.obs_names[i_hi]
        low_val, high_val = float(totals[i_lo]), float(totals[i_hi])

    def load_patch(barcode: str) -> np.ndarray | None:
        matches = sorted(patches_dir.glob(f"{barcode}*.png"))
        if not matches:
            matches = sorted(patches_dir.glob(f"*{barcode.split('-')[0]}*"))
        if matches:
            return np.array(Image.open(matches[0]).convert("RGB"))
        return None

    if low_bc:
        low_img = load_patch(low_bc)
        if low_img is not None:
            Image.fromarray(low_img).save(
                out_dir / f"05_low_total_counts{low_val:.0f}.png"
            )
    if high_bc:
        high_img = load_patch(high_bc)
        if high_img is not None:
            Image.fromarray(high_img).save(
                out_dir / f"06_high_total_counts{high_val:.0f}.png"
            )

    gem = barcode_suffix
    if gem is None and sample_id:
        for k, v in GEM_TO_SAMPLE.items():
            if v == (sample_id or out_dir.name):
                gem = k
                break

    readme = {
        "sample": sample_id or out_dir.name,
        "hires_path": str(hires),
        "spots_source": spots_source,
        "he_display_shape_hw": list(he.shape[:2]),
        "full_hires_shape_hw": [H, W],
        "display_scale": disp_scale,
        "spots_filter": {
            "barcode_suffix": gem,
            "note": "GSE252265 aggr positions filtered to this sample only",
        },
        "n_spots_in_file_total": int(len(df_all)),
        "n_in_tissue_all_8_samples_in_file": n_all_in,
        "n_spots_this_sample": int(len(df)),
        "n_in_tissue": int((df["in_tissue"].astype(int) == 1).sum()),
        "n_out_tissue": int((df["in_tissue"].astype(int) == 0).sum()),
        "mask_note": "02 = rasterized in_tissue Visium spots for this sample only (not tumor annotation)",
        "spot_radius_hires_px": spot_diam_hires / 2.0,
        "low_patch": {"barcode": low_bc, "total_counts_norm": low_val},
        "high_patch": {"barcode": high_bc, "total_counts_norm": high_val},
        "patches_dir": str(patches_dir),
    }
    (out_dir / "README.json").write_text(json.dumps(readme, indent=2))
    print(
        f"Exported to {out_dir}  "
        f"(in_tissue {readme['n_in_tissue']} for sample; {n_all_in} in file across 8 samples)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/students/hbae/figures/GSM7998257")
    ap.add_argument(
        "--hires",
        default="/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/tissue_hires_image.png",
    )
    ap.add_argument(
        "--spots",
        default=None,
        help="tissue_positions_list.csv (optional if --spots_h5ad set)",
    )
    ap.add_argument(
        "--spots_h5ad",
        default=None,
        help="st.h5ad with obsm['spatial'] when positions CSV missing",
    )
    ap.add_argument(
        "--scale",
        default="/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/scalefactors_json.json",
    )
    ap.add_argument(
        "--patches",
        default="/home/students/hbae/data/Processed_Data_65um_224/GSE252265/GSM7998257/patches",
    )
    ap.add_argument(
        "--st_h5ad",
        default="/home/students/hbae/data/Processed_Data/GSE252265/GSM7998257/st_norm.h5ad",
    )
    ap.add_argument("--max_px", type=int, default=2000)
    ap.add_argument(
        "--sample_id",
        default=None,
        help="Sample ID (e.g. GSM7998257); auto-filters GSE252265 aggr suffix",
    )
    ap.add_argument(
        "--barcode_suffix",
        type=int,
        default=None,
        help="Override gem suffix (GSM7998257 -> 6)",
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    sample_id = args.sample_id or out.name
    export_st(
        out,
        Path(args.hires),
        Path(args.spots) if args.spots else None,
        Path(args.scale),
        Path(args.patches),
        Path(args.st_h5ad) if args.st_h5ad else None,
        max_px=args.max_px,
        sample_id=sample_id,
        barcode_suffix=args.barcode_suffix,
        spots_h5ad=Path(args.spots_h5ad) if args.spots_h5ad else None,
    )


if __name__ == "__main__":
    main()
