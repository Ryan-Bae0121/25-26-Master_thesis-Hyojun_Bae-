#!/usr/bin/env python3
"""
Export Figure 2.1 panel images as separate PNGs (no axes/labels).

Usage:
    python export_figure2_1_components.py --out_dir /home/students/hbae/figures/figure2_1_components
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
from PIL import Image

# Reuse helpers from combined figure script
from make_figure2_1_combined import (
    DEFAULT_H5_DIR,
    DEFAULT_HIRES,
    DEFAULT_MANIFEST,
    DEFAULT_MASK_DIR,
    DEFAULT_META,
    DEFAULT_PATCHES,
    DEFAULT_SCALE,
    DEFAULT_SPOTS,
    DEFAULT_TCGA_SLIDE,
    DEFAULT_WSI_DIR,
    MASK_LEVEL,
    background_crop_from_wsi,
    load_tcga_he_and_mask,
    overlay_mask_contour,
    pick_tcga_example_tiles,
    pick_tissue_patches,
    resolve_mask_path,
)
from make_figure2_1_v2 import load_spots


def save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_gray(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def export_tcga(
    out: Path,
    wsi_dir: Path,
    mask_dir: Path,
    slide_id: str,
    manifest: Path,
    h5_dir: Path,
    tile_root: Path,
    mask_source: str,
    max_px: int,
) -> None:
    sub = out / "tcga"
    sub.mkdir(parents=True, exist_ok=True)
    wsi_path = wsi_dir / slide_id
    mask_path = resolve_mask_path(mask_dir, slide_id)

    he, mask, mask_l3, _, _, down, _ = load_tcga_he_and_mask(
        wsi_path, mask_path, mask_source=mask_source, max_px=max_px
    )
    overlay = overlay_mask_contour(he, mask, alpha=0.22)

    save_rgb(he, sub / "a_tcga_he_l3.png")
    save_gray(mask, sub / "a_tcga_mask.png")
    save_rgb(overlay, sub / "a_tcga_he_mask_overlay.png")

    tiles = pick_tcga_example_tiles(slide_id, manifest, h5_dir, tile_root)
    bg = background_crop_from_wsi(wsi_path, down=down)
    if bg is not None:
        save_rgb(bg, sub / "a_tcga_tile_background.png")
    if tiles:
        low, high, low_f, high_f = tiles
        if low is not None:
            save_rgb(low, sub / f"a_tcga_tile_low_tissue_frac{low_f:.2f}.png")
        if high is not None:
            save_rgb(high, sub / f"a_tcga_tile_high_tissue_frac{high_f:.2f}.png")

    meta = {
        "slide_id": slide_id,
        "mask_source": mask_source,
        "mask_level": MASK_LEVEL,
        "he_shape_hw": list(he.shape[:2]),
        "wsi_path": str(wsi_path),
        "mask_path": str(mask_path),
    }
    (sub / "README_tcga.json").write_text(json.dumps(meta, indent=2))


def export_visium(
    out: Path,
    hires: Path,
    spots: Path,
    scale: Path,
    patches_dir: Path,
    meta: Path,
    seed: int,
) -> None:
    sub = out / "visium"
    sub.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(hires).convert("RGB"))
    H, W = img.shape[:2]
    df, _scalef = load_spots(str(spots), str(scale))
    in_t = df[df["in_tissue"] == 1]
    out_t = df[df["in_tissue"] == 0]

    save_rgb(img, sub / "b_visium_he_section.png")

    # Spot overlay layers (RGBA-style via matplotlib-free rasterization)
    from PIL import ImageDraw

    base = Image.fromarray(img)
    spot_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(spot_layer)
    r_out = 2
    r_in = 4
    for x, y in zip(out_t["x_hires"], out_t["y_hires"]):
        draw.ellipse((x - r_out, y - r_out, x + r_out, y + r_out), fill=(187, 187, 187, 40))
    for x, y in zip(in_t["x_hires"], in_t["y_hires"]):
        draw.ellipse((x - r_in, y - r_in, x + r_in, y + r_in), fill=(46, 134, 171, 75))
    combined = Image.alpha_composite(base.convert("RGBA"), spot_layer).convert("RGB")
    combined.save(sub / "b_visium_he_spots_overlay.png")

    cx = float(in_t["x_hires"].quantile(0.5))
    cy = float(in_t["y_hires"].quantile(0.4))
    box = min(W, H) * 0.2
    bx0, by0 = max(0, cx - box / 2), max(0, cy - box / 2)
    bx1, by1 = min(W, bx0 + box), min(H, by0 + box)
    xi0, yi0, xi1, yi1 = int(bx0), int(by0), int(bx1), int(by1)

    zoom_he = img[yi0:yi1, xi0:xi1]
    save_rgb(zoom_he, sub / "b_visium_zoom_he.png")

    zoom_base = Image.fromarray(zoom_he)
    zoom_layer = Image.new("RGBA", (xi1 - xi0, yi1 - yi0), (0, 0, 0, 0))
    zdraw = ImageDraw.Draw(zoom_layer)
    zm = (
        (in_t["x_hires"] >= bx0)
        & (in_t["x_hires"] <= bx1)
        & (in_t["y_hires"] >= by0)
        & (in_t["y_hires"] <= by1)
    )
    zs = in_t[zm]
    for x, y in zip(zs["x_hires"] - bx0, zs["y_hires"] - by0):
        zdraw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(46, 134, 171, 90), outline=(255, 255, 255, 180))
    zoom_ov = Image.alpha_composite(zoom_base.convert("RGBA"), zoom_layer).convert("RGB")
    zoom_ov.save(sub / "b_visium_zoom_spots_overlay.png")

    zoom_box = {"x0": bx0, "y0": by0, "x1": bx1, "y1": by1, "image_wh": [W, H]}
    (sub / "b_visium_zoom_box.json").write_text(json.dumps(zoom_box, indent=2))

    random.seed(seed)
    patch_files = pick_tissue_patches(patches_dir, n=6, seed=seed)
    for i, pf in enumerate(patch_files, start=1):
        arr = np.array(Image.open(pf).convert("RGB"))
        dst = sub / f"b_visium_patch_{i:02d}_{pf.stem}.png"
        save_rgb(arr, dst)

    meta_d = {
        "sample": "GSM7998257",
        "n_in_tissue_spots": int(len(in_t)),
        "n_out_tissue_spots": int(len(out_t)),
        "hires_wh": [W, H],
        "patches": [p.name for p in patch_files],
    }
    (sub / "README_visium.json").write_text(json.dumps(meta_d, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/students/hbae/figures/figure2_1_components")
    ap.add_argument("--wsi_dir", default=DEFAULT_WSI_DIR)
    ap.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    ap.add_argument("--tcga_slide", default=DEFAULT_TCGA_SLIDE)
    ap.add_argument("--mask_source", choices=("precomputed", "otsu"), default="precomputed")
    ap.add_argument("--h5_dir", default=DEFAULT_H5_DIR)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--tile_root", default="/home/students/hbae/data/TCGA_HNSC_tiles_144px")
    ap.add_argument("--max_px", type=int, default=2000, help="Max long side for TCGA L3 export")
    ap.add_argument("--hires", default=DEFAULT_HIRES)
    ap.add_argument("--spots", default=DEFAULT_SPOTS)
    ap.add_argument("--scale", default=DEFAULT_SCALE)
    ap.add_argument("--patches", default=DEFAULT_PATCHES)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Exporting TCGA components...")
    export_tcga(
        out,
        Path(args.wsi_dir),
        Path(args.mask_dir),
        args.tcga_slide,
        Path(args.manifest),
        Path(args.h5_dir),
        Path(args.tile_root),
        args.mask_source,
        args.max_px,
    )

    print("Exporting Visium components...")
    export_visium(
        out,
        Path(args.hires),
        Path(args.spots),
        Path(args.scale),
        Path(args.patches),
        Path(args.meta),
        args.seed,
    )

    print(f"\nDone. Components in: {out}")
    for p in sorted(out.rglob("*.png")):
        print(f"  {p.relative_to(out)}")


if __name__ == "__main__":
    main()
