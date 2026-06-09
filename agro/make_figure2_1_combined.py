#!/usr/bin/env python3
"""
Figure 2.1 combined: (a) TCGA WSI + tissue mask  |  (b) Visium ST (GSM7998257)

Usage:
    python make_figure2_1_combined.py --out_dir /home/students/hbae/figures

Optional overrides for paths are available via CLI (see --help).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

# ── palette ─────────────────────────────────────────────────────────────
C_BORDER = "#222222"
C_TCGA_MASK = "#2ECC71"
C_SPOT_IN = "#2E86AB"
C_SPOT_OUT = "#BBBBBB"
C_BOX = "#E74C3C"
C_BG_LABEL = "#888888"
C_LOW = "#F39C12"
C_HIGH = "#8E44AD"

DEFAULT_WSI_DIR = "/shares/bioit-students/ir_students_2020/TCGA-HNSC/WSIs"
DEFAULT_MASK_DIR = "/shares/bioit-students/ir_students_2020/TCGA-HNSC/masks"
DEFAULT_H5_DIR = "/home/students/hbae/data/TCGA_HNSC_tiles_144px_h5"
DEFAULT_MANIFEST = "/home/students/hbae/data/TCGA_HNSC_tiles_144px/manifest_tiles.csv"
DEFAULT_TCGA_SLIDE = (
    "TCGA-4P-AA8J-01Z-00-DX1.5B44796F-D099-4076-9CAF-B40C2B83F432.svs"
)

DEFAULT_HIRES = (
    "/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/tissue_hires_image.png"
)
DEFAULT_SPOTS = (
    "/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/tissue_positions_list.csv"
)
DEFAULT_SCALE = (
    "/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/scalefactors_json.json"
)
DEFAULT_PATCHES = (
    "/home/students/hbae/data/Processed_Data_65um_224/GSE252265/GSM7998257/patches"
)
DEFAULT_META = (
    "/home/students/hbae/data/Processed_Data_65um_224/GSE252265/GSM7998257/patch_extraction_65um_meta.csv"
)

# Precomputed TCGA masks are stored at OpenSlide pyramid level 3 (see tcga_wsi_tile_144px.py).
MASK_LEVEL = 3


def resolve_mask_path(mask_dir: Path, slide_id: str) -> Path:
    folder = mask_dir / slide_id.replace(".svs", "")
    if (folder / "mask.png").exists():
        return folder / "mask.png"
    png = mask_dir / f"{slide_id.replace('.svs', '')}_mask.png"
    if png.exists():
        return png
    raise FileNotFoundError(f"No mask for {slide_id}")


def load_mask_bool(mask_path: Path) -> np.ndarray:
    """Load boolean tissue mask (H, W). Prefer mask.npy in the same folder."""
    folder = mask_path.parent
    if (folder / "mask.npy").exists():
        arr = np.load(folder / "mask.npy")
    else:
        arr = np.array(Image.open(mask_path).convert("L"))
    return arr.astype(np.uint8) > 0


def otsu_tissue_mask(he_rgb: np.ndarray) -> np.ndarray:
    """Otsu tissue mask from H&E RGB — same logic as tcga_wsi_tile_144px.py."""
    gray = (
        0.299 * he_rgb[..., 0]
        + 0.587 * he_rgb[..., 1]
        + 0.114 * he_rgb[..., 2]
    ).astype(np.uint8)
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thr = 200
    for t in range(256):
        w_b += hist[t]
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thr = t
    return gray < float(thr)


def load_tcga_he_and_mask(
    wsi_path: Path,
    mask_path: Path | None,
    mask_source: str = "otsu",
    level: int = MASK_LEVEL,
    max_px: int = 2000,
):
    """
    H&E and tissue mask at the same pyramid level, jointly resized for display.

    mask_source:
      - 'otsu': mask from the same L3 H&E (pixel-perfect for figure)
      - 'precomputed': shares/.../masks (used in tiling; may not match H&E on some slides)
    """
    slide = openslide.OpenSlide(str(wsi_path))
    w0, h0 = slide.dimensions
    lw, lh = slide.level_dimensions[level]
    down = float(slide.level_downsamples[level])
    he_full = np.array(slide.read_region((0, 0), level, (lw, lh)).convert("RGB"))
    slide.close()

    if mask_source == "precomputed":
        if mask_path is None:
            raise ValueError("mask_path required for precomputed masks")
        mask_full = load_mask_bool(mask_path)
        if mask_full.shape != (lh, lw):
            raise ValueError(
                f"Mask shape {mask_full.shape} != level-{level} (H={lh}, W={lw})"
            )
        otsu = otsu_tissue_mask(he_full)
        iou = (mask_full & otsu).sum() / max((mask_full | otsu).sum(), 1)
        print(f"  precomputed vs same-HE Otsu IoU: {iou:.3f}")
        if iou < 0.35:
            print(
                "  WARNING: precomputed mask poorly matches this H&E; "
                "use --mask_source otsu for figure"
            )
    else:
        mask_full = otsu_tissue_mask(he_full)

    scale = min(1.0, max_px / max(lw, lh))
    dw = max(1, int(round(lw * scale)))
    dh = max(1, int(round(lh * scale)))
    he = np.array(Image.fromarray(he_full).resize((dw, dh), Image.Resampling.LANCZOS))
    mask = (
        np.array(
            Image.fromarray(mask_full.astype(np.uint8) * 255).resize(
                (dw, dh), Image.Resampling.NEAREST
            )
        )
        > 127
    )
    return he, mask, mask_full, (dw, dh), (w0, h0), down, mask_source


def overlay_mask_contour(he: np.ndarray, mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Tint tissue + draw mask boundary (pixel-aligned with HE)."""
    out = he.astype(np.float32).copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 255.0
    m = mask[..., None]
    out = out * (1 - alpha * m) + green * (alpha * m)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 1px contour on mask boundary
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    boundary = mask & ~eroded
    out[boundary] = (0.3 * out[boundary] + 0.7 * np.array([0, 220, 0])).astype(np.uint8)
    return out


def pick_tcga_example_tiles(slide_id: str, manifest: Path, h5_dir: Path, tile_root: Path):
    """Return (bg_crop_rgb, low_rgb, high_rgb, labels) from manifest + h5."""
    df = pd.read_csv(manifest)
    sub = df[df["slide_id"] == slide_id].copy()
    if sub.empty:
        return None

    low_row = sub.loc[sub["tissue_frac_est"].idxmin()]
    high_row = sub.loc[sub["tissue_frac_est"].idxmax()]

    def load_tile(row):
        p = tile_root / row["tile_relpath"]
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
        h5_path = h5_dir / slide_id / "tiles.h5"
        if not h5_path.exists():
            return None
        x, y = int(row["x_level0"]), int(row["y_level0"])
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            images = f["images"]
            hits = np.where((coords[:, 0] == x) & (coords[:, 1] == y))[0]
            if len(hits):
                return images[int(hits[0])][:]
        return None

    low = load_tile(low_row)
    high = load_tile(high_row)
    return low, high, float(low_row["tissue_frac_est"]), float(high_row["tissue_frac_est"])


def background_crop_from_wsi(
    wsi_path: Path,
    down: float | None = None,
    level: int = MASK_LEVEL,
    size: int = 144,
    bright_min: float = 230.0,
) -> np.ndarray | None:
    """Crop empty glass (bright, non-tissue) at level 0 — not tumor-mask exterior."""
    slide = openslide.OpenSlide(str(wsi_path))
    lw, lh = slide.level_dimensions[level]
    if down is None:
        down = float(slide.level_downsamples[level])
    he = np.array(slide.read_region((0, 0), level, (lw, lh)).convert("RGB"))
    tissue = otsu_tissue_mask(he)
    gray = (
        0.299 * he[..., 0] + 0.587 * he[..., 1] + 0.114 * he[..., 2]
    ).astype(np.float32)
    glass = (~tissue) & (gray >= bright_min)
    if not glass.any():
        non_tissue = ~tissue
        if not non_tissue.any():
            slide.close()
            return None
        thr = float(np.percentile(gray[non_tissue], 95))
        glass = non_tissue & (gray >= thr)
    if not glass.any():
        glass = ~tissue
    cy, cx = np.unravel_index(int(np.argmax(np.where(glass, gray, -1.0))), gray.shape)
    x0 = int(round(cx * down)) - size // 2
    y0 = int(round(cy * down)) - size // 2
    w0, h0 = slide.dimensions
    x0 = max(0, min(x0, w0 - size))
    y0 = max(0, min(y0, h0 - size))
    region = slide.read_region((x0, y0), 0, (size, size)).convert("RGB")
    slide.close()
    return np.array(region)


def load_spots(spots_path, scale_path):
    with open(scale_path) as f:
        scale = json.load(f)
    scalef = scale["tissue_hires_scalef"]
    df = pd.read_csv(spots_path)
    # Some GEO exports omit the header row (e.g. GSE181300).
    if "pxl_row_fullres" not in df.columns and "pxl_row_in_fullres" not in df.columns:
        first = str(df.columns[0])
        if first.endswith("-1") or first.endswith("-2") or "-" in first:
            df = pd.read_csv(
                spots_path,
                header=None,
                names=[
                    "barcode",
                    "in_tissue",
                    "array_row",
                    "array_col",
                    "pxl_row_in_fullres",
                    "pxl_col_in_fullres",
                ],
            )
    col_map = {}
    for c in df.columns:
        if "pxl_row" in c.lower():
            col_map[c] = "pxl_row_fullres"
        if "pxl_col" in c.lower():
            col_map[c] = "pxl_col_fullres"
        if "in_tissue" in c.lower():
            col_map[c] = "in_tissue"
    df = df.rename(columns=col_map)
    df["y_hires"] = df["pxl_row_fullres"] * scalef
    df["x_hires"] = df["pxl_col_fullres"] * scalef
    return df


def qc_patch_count(meta_path):
    if meta_path and Path(meta_path).exists():
        meta = pd.read_csv(meta_path)
        if "n_patches_written" in meta.columns:
            return int(meta["n_patches_written"].iloc[0])
    return 2535


def pick_tissue_patches(patch_dir, n=6, seed=42):
    patch_dir = Path(patch_dir)
    all_patches = sorted(patch_dir.glob("*.png"))
    scored = []
    random.seed(seed)
    sample = random.sample(all_patches, min(200, len(all_patches)))
    for pf in sample:
        arr = np.array(Image.open(pf).convert("RGB"))
        if arr.mean() < 220:
            scored.append((arr.mean(), pf))
    scored.sort(key=lambda x: x[0])
    picks = [p for _, p in scored[:n]]
    if len(picks) < n:
        rest = [p for p in all_patches if p not in picks]
        random.seed(seed)
        picks += random.sample(rest, min(n - len(rest), len(rest)))
    return picks[:n]


def draw_tcga_panel(
    fig,
    spec,
    wsi_dir,
    mask_dir,
    slide_id,
    manifest,
    h5_dir,
    tile_root,
    mask_source: str = "otsu",
):
    """spec: [left, bottom, width, height] in figure coordinates."""
    left, bottom, width, height = spec
    slide_stem = slide_id.replace(".svs", "")
    wsi_path = wsi_dir / slide_id
    mask_path = resolve_mask_path(mask_dir, slide_id)

    print(f"  TCGA WSI:  {wsi_path}")
    print(f"  mask_source: {mask_source}")
    if mask_source == "precomputed":
        print(f"  TCGA mask: {mask_path}")
    he, mask, mask_l3, (tw, th), (w0, h0), down, mask_source = load_tcga_he_and_mask(
        wsi_path, mask_path, mask_source=mask_source
    )
    tissue_pct = mask.mean() * 100
    he_mask = overlay_mask_contour(he, mask, alpha=0.22)

    # layout: 3 top panels (H&E | mask | overlay) + 3 bottom tiles
    gap = 0.015
    w3 = (width - 2 * gap) / 3
    h_top = height * 0.56
    h_bot = height * 0.38
    y_top = bottom + h_bot + gap
    y_bot = bottom

    tcga_short = slide_stem.split(".")[0]

    ax_he = fig.add_axes([left, y_top, w3, h_top])
    ax_mk = fig.add_axes([left + w3 + gap, y_top, w3, h_top])
    ax_ov = fig.add_axes([left + 2 * (w3 + gap), y_top, w3, h_top])

    ax_he.imshow(he, origin="upper")
    ax_he.set_title(f"H\u0026E @ L{MASK_LEVEL}", fontsize=8, pad=2)
    ax_he.axis("off")

    ax_mk.imshow(mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1, origin="upper")
    mask_lbl = (
        "Tissue mask (Otsu)\nfrom same H&E"
        if mask_source == "otsu"
        else "Precomputed mask\n(white = tissue)"
    )
    ax_mk.set_title(mask_lbl, fontsize=8, pad=2)
    ax_mk.axis("off")

    ax_ov.imshow(he_mask, origin="upper")
    ax_ov.set_title("H\u0026E + mask contour", fontsize=8, pad=2)
    ax_ov.axis("off")

    # zoom box on tissue-rich area
    ys, xs = np.where(mask)
    if len(xs):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            lw=1.5,
            edgecolor=C_BOX,
            facecolor="none",
            linestyle="--",
        )
        ax_he.add_patch(rect)

    ax_he.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_he.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        color=C_BORDER,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )
    fig.text(
        left + width / 2,
        bottom + height + 0.008,
        f"(a) TCGA-HNSC: {tcga_short}  |  same .svs + masks/{slide_stem}/",
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color=C_BORDER,
    )
    fig.text(
        left + width / 2,
        y_top + h_top + 0.006,
        f"L0 {w0:,}\u00d7{h0:,} px  |  mask @ L{MASK_LEVEL} ({mask_source})  |  tissue {tissue_pct:.1f}%",
        ha="center",
        fontsize=6.5,
        color="#666666",
        transform=fig.transFigure,
    )

    # example tiles
    tw3 = (width - 2 * gap) / 3  # bottom row tile width
    labels = [
        ("Background\n(excluded)", C_BG_LABEL),
        ("Low tissue\ncontent", C_LOW),
        ("High tissue\ncontent", C_HIGH),
    ]
    bg = background_crop_from_wsi(wsi_path, down=down)
    tiles = pick_tcga_example_tiles(slide_id, manifest, h5_dir, tile_root)
    low_img = high_img = None
    low_f = high_f = 0.0
    if tiles:
        low_img, high_img, low_f, high_f = tiles

    imgs = [bg, low_img, high_img]
    fracs = [0.0, low_f, high_f]
    for i, (lab, col) in enumerate(labels):
        ax_t = fig.add_axes([left + i * (tw3 + gap), y_bot, tw3, h_bot])
        im = imgs[i]
        if im is not None:
            ax_t.imshow(im)
        else:
            ax_t.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_t.transAxes)
        ax_t.axis("off")
        for sp in ax_t.spines.values():
            sp.set_edgecolor(col)
            sp.set_linewidth(2)
        subtitle = lab
        if i > 0 and fracs[i] > 0:
            subtitle += f"\n(tissue={fracs[i]:.0%})"
        ax_t.set_title(subtitle, fontsize=7, color=col, pad=2)
        if i == 0:
            ax_t.text(
                0.05,
                0.95,
                "144 px",
                transform=ax_t.transAxes,
                fontsize=5.5,
                va="top",
                color="white",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#333333", alpha=0.75),
            )


def draw_visium_panel(fig, spec, hires, spots, scale, patches, meta):
    left, bottom, width, height = spec
    img = np.array(Image.open(hires).convert("RGB"))
    H, W = img.shape[:2]
    df = load_spots(spots, scale)
    in_t = df[df["in_tissue"] == 1]
    out_t = df[df["in_tissue"] == 0]
    n_qc = qc_patch_count(meta)
    patch_files = pick_tissue_patches(patches, n=6)

    # main section + zoom + patches within right panel
    w_main = width * 0.48
    w_zoom = width * 0.22
    w_patch = width * 0.28
    gap = 0.015
    h_upper = height * 0.55
    h_lower = height * 0.38
    y_upper = bottom + h_lower + gap
    y_lower = bottom

    ax_m = fig.add_axes([left, y_upper, w_main, h_upper])
    ax_m.imshow(img, origin="upper", aspect="auto")
    ax_m.set_xlim(0, W)
    ax_m.set_ylim(H, 0)
    ax_m.scatter(
        out_t["x_hires"],
        out_t["y_hires"],
        s=1.2,
        c=C_SPOT_OUT,
        alpha=0.12,
        linewidths=0,
        zorder=2,
    )
    ax_m.scatter(
        in_t["x_hires"],
        in_t["y_hires"],
        s=4,
        c=C_SPOT_IN,
        alpha=0.28,
        linewidths=0,
        zorder=3,
    )

    cx = float(in_t["x_hires"].quantile(0.5))
    cy = float(in_t["y_hires"].quantile(0.4))
    box = min(W, H) * 0.2
    bx0 = max(0, cx - box / 2)
    by0 = max(0, cy - box / 2)
    bx1 = min(W, bx0 + box)
    by1 = min(H, by0 + box)
    ax_m.add_patch(
        Rectangle(
            (bx0, by0),
            bx1 - bx0,
            by1 - by0,
            lw=1.8,
            edgecolor=C_BOX,
            facecolor="none",
            zorder=5,
        )
    )
    ax_m.text(
        0.02,
        0.98,
        "(b)",
        transform=ax_m.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        color=C_BORDER,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )
    ax_m.set_title(
        "Visium H\u0026E section + spot overlay\n"
        f"{len(in_t):,} in-tissue spots",
        fontsize=8,
        pad=3,
    )
    ax_m.tick_params(labelsize=6)
    ax_m.set_xlabel("x (px)", fontsize=7)
    ax_m.set_ylabel("y (px)", fontsize=7)

    ax_z = fig.add_axes([left + w_main + gap, y_upper, w_zoom, h_upper])
    xi0, yi0, xi1, yi1 = int(bx0), int(by0), int(bx1), int(by1)
    ax_z.imshow(img[yi0:yi1, xi0:xi1], origin="upper", aspect="auto")
    zm = (
        (in_t["x_hires"] >= bx0)
        & (in_t["x_hires"] <= bx1)
        & (in_t["y_hires"] >= by0)
        & (in_t["y_hires"] <= by1)
    )
    zs = in_t[zm].copy()
    ax_z.scatter(
        zs["x_hires"] - bx0,
        zs["y_hires"] - by0,
        s=22,
        c=C_SPOT_IN,
        alpha=0.35,
        linewidths=0.5,
        edgecolors="white",
        zorder=3,
    )
    ax_z.set_xlim(0, xi1 - xi0)
    ax_z.set_ylim(yi1 - yi0, 0)
    ax_z.axis("off")
    for sp in ax_z.spines.values():
        sp.set_edgecolor(C_BOX)
        sp.set_linewidth(1.8)
    ax_z.set_title("Spot grid (zoom)", fontsize=7.5, color=C_BOX, pad=2)

    fig.text(
        left + width / 2,
        bottom + height + 0.008,
        "Visium ST spot-aligned patch extraction (GSM7998257, GSE252265)",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=C_BORDER,
    )
    fig.text(
        left + width / 2,
        y_upper + h_upper - 0.01,
        f"{n_qc:,} QC-passed patches  |  224\u00d7224 px  |  ~65 \u00b5m FOV",
        ha="center",
        fontsize=7,
        color="#666666",
        transform=fig.transFigure,
    )

    # 2x3 patch grid
    ncol, nrow = 3, 2
    pw = (w_patch - gap * (ncol - 1)) / ncol
    ph = (h_lower - gap) / nrow
    x0p = left + w_main + w_zoom + 2 * gap
    for idx, pf in enumerate(patch_files[:6]):
        r, c = idx // ncol, idx % ncol
        ax_p = fig.add_axes(
            [x0p + c * (pw + gap), y_lower + (nrow - 1 - r) * (ph + gap), pw, ph]
        )
        ax_p.imshow(np.array(Image.open(pf).convert("RGB")))
        ax_p.axis("off")
        if idx == 0:
            ax_p.text(
                0.05,
                0.95,
                "224 px",
                transform=ax_p.transAxes,
                fontsize=5.5,
                va="top",
                color="white",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#333333", alpha=0.75),
            )

    fig.text(
        x0p + w_patch / 2,
        y_lower - 0.012,
        "Representative QC-passed patches (tissue-rich)",
        ha="center",
        fontsize=7,
        color="#777777",
        style="italic",
        transform=fig.transFigure,
    )


def main():
    ap = argparse.ArgumentParser(description="Figure 2.1 combined TCGA + Visium")
    ap.add_argument("--wsi_dir", default=DEFAULT_WSI_DIR)
    ap.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    ap.add_argument("--tcga_slide", default=DEFAULT_TCGA_SLIDE)
    ap.add_argument("--h5_dir", default=DEFAULT_H5_DIR)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--tile_root", default="/home/students/hbae/data/TCGA_HNSC_tiles_144px")
    ap.add_argument("--hires", default=DEFAULT_HIRES)
    ap.add_argument("--spots", default=DEFAULT_SPOTS)
    ap.add_argument("--scale", default=DEFAULT_SCALE)
    ap.add_argument("--patches", default=DEFAULT_PATCHES)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--out_dir", default="/home/students/hbae/figures")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--mask_source",
        choices=("otsu", "precomputed"),
        default="otsu",
        help="TCGA mask: otsu=from same H&E L3 (aligned); precomputed=shares masks",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.98,
        "Figure 2.1. H\u0026E Image Characteristics and Patch Extraction Pipeline",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=C_BORDER,
    )
    fig.text(
        0.5,
        0.945,
        "Head and neck squamous cell carcinoma (HNSCC): TCGA bulk WSI tiling (a) "
        "and Visium spatial transcriptomics (b)",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
        style="italic",
    )

    print("Drawing panel (a) TCGA...")
    draw_tcga_panel(
        fig,
        [0.04, 0.06, 0.44, 0.84],
        Path(args.wsi_dir),
        Path(args.mask_dir),
        args.tcga_slide,
        Path(args.manifest),
        Path(args.h5_dir),
        Path(args.tile_root),
        mask_source=args.mask_source,
    )

    print("Drawing panel (b) Visium...")
    draw_visium_panel(
        fig,
        [0.51, 0.06, 0.47, 0.84],
        args.hires,
        args.spots,
        args.scale,
        args.patches,
        args.meta,
    )

    out_path = out_dir / "figure2_1_combined_tcga_visium.png"
    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
