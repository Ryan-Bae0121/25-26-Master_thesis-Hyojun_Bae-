#!/usr/bin/env python3
"""
make_figure2_1_v2.py  —  Figure 2.1 revised
Fixes: text overlap, spot alpha, tissue patch selection, caption accuracy

Usage:
    python make_figure2_1_v2.py \
        --hires   "/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/tissue_hires_image.png" \
        --spots   "/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/tissue_positions_list.csv" \
        --scale   "/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/scalefactors_json.json" \
        --patches "/home/students/hbae/data/Processed_Data_65um_224/GSE252265/GSM7998257/patches" \
        --meta    "/home/students/hbae/data/Processed_Data_65um_224/GSE252265/GSM7998257/patch_extraction_65um_meta.csv" \
        --out_dir "/home/students/hbae/figures"
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

C_SPOT_IN = "#2E86AB"
C_SPOT_OUT = "#BBBBBB"
C_BOX = "#E74C3C"
C_BORDER = "#222222"

N_QC_PATCHES = 2535


def load_spots(spots_path, scale_path):
    with open(scale_path) as f:
        scale = json.load(f)
    scalef = scale["tissue_hires_scalef"]
    df = pd.read_csv(spots_path)
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
    return df, scalef


def qc_patch_count(meta_path):
    if meta_path and Path(meta_path).exists():
        meta = pd.read_csv(meta_path)
        if "n_patches_written" in meta.columns:
            return int(meta["n_patches_written"].iloc[0])
    return N_QC_PATCHES


def pick_tissue_patches(patch_dir, meta_path, n=6, seed=42):
    """Pick patches with high tissue content (brightness fallback if no per-patch meta)."""
    patch_dir = Path(patch_dir)

    if meta_path and Path(meta_path).exists():
        meta = pd.read_csv(meta_path)
        frac_col = None
        for c in meta.columns:
            if "tissue" in c.lower() and ("frac" in c.lower() or "ratio" in c.lower()):
                frac_col = c
                break
        if frac_col:
            meta = meta.sort_values(frac_col, ascending=False)
            top = meta.head(50)
            candidates = []
            for _, row in top.iterrows():
                barcode = str(row.get("barcode", row.get("patch_name", ""))).split(".")[0]
                matches = list(patch_dir.glob(f"{barcode}*.png"))
                if matches:
                    candidates.extend(matches)
                if len(candidates) >= 20:
                    break
            if len(candidates) >= n:
                random.seed(seed)
                return random.sample(candidates, n)

    all_patches = sorted(patch_dir.glob("*.png"))
    scored = []
    random.seed(seed)
    sample = random.sample(all_patches, min(200, len(all_patches)))
    for pf in sample:
        arr = np.array(Image.open(pf).convert("RGB"))
        brightness = arr.mean()
        # lower brightness = more H&E stain / tissue
        if brightness < 220:
            scored.append((brightness, pf))
    scored.sort(key=lambda x: x[0])
    picks = [p for _, p in scored[:n]]
    if len(picks) < n:
        remaining = [p for p in all_patches if p not in picks]
        random.seed(seed)
        picks += random.sample(remaining, min(n - len(picks), len(remaining)))
    return picks[:n]


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading H&E hires image...")
    img = np.array(Image.open(args.hires).convert("RGB"))
    H, W = img.shape[:2]
    print(f"  Image: {W}x{H} px")

    print("Loading spots...")
    df, scalef = load_spots(args.spots, args.scale)
    in_t = df[df["in_tissue"] == 1]
    out_t = df[df["in_tissue"] == 0]
    n_qc = qc_patch_count(args.meta)
    print(f"  In-tissue: {len(in_t)}, Out: {len(out_t)}, QC patches: {n_qc}")

    print("Picking tissue patches...")
    patches = pick_tissue_patches(args.patches, args.meta, n=6)
    print(f"  Selected {len(patches)} patches")
    for p in patches:
        print(f"    {p.name}")

    fig = plt.figure(figsize=(16, 8.5))
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.98,
        "H\u0026E Whole-Slide Image Characteristics and Patch Extraction Pipeline",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=C_BORDER,
    )
    fig.text(
        0.5,
        0.945,
        "Representative HNSCC Visium sample: GSM7998257 (GSE252265, tongue SCC)",
        ha="center",
        va="top",
        fontsize=9,
        color="#666666",
        style="italic",
    )
    fig.text(
        0.5,
        0.905,
        f"{len(in_t):,} in-tissue Visium array spots  |  {n_qc:,} QC-passed patches used for analysis",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )

    ax_a = fig.add_axes([0.03, 0.06, 0.43, 0.82])
    ax_a.imshow(img, origin="upper", aspect="auto")
    ax_a.set_xlim(0, W)
    ax_a.set_ylim(H, 0)

    ax_a.scatter(
        out_t["x_hires"],
        out_t["y_hires"],
        s=1.5,
        c=C_SPOT_OUT,
        alpha=0.15,
        linewidths=0,
        zorder=2,
    )
    ax_a.scatter(
        in_t["x_hires"],
        in_t["y_hires"],
        s=5,
        c=C_SPOT_IN,
        alpha=0.30,
        linewidths=0,
        zorder=3,
    )

    cx = float(in_t["x_hires"].quantile(0.5))
    cy = float(in_t["y_hires"].quantile(0.4))
    box = min(W, H) * 0.18
    bx0 = max(0, cx - box / 2)
    by0 = max(0, cy - box / 2)
    bx1 = min(W, bx0 + box)
    by1 = min(H, by0 + box)

    rect = Rectangle(
        (bx0, by0),
        bx1 - bx0,
        by1 - by0,
        lw=2,
        edgecolor=C_BOX,
        facecolor="none",
        zorder=5,
    )
    ax_a.add_patch(rect)

    ax_a.text(
        0.015,
        0.985,
        "(a)",
        transform=ax_a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        color=C_BORDER,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
    )

    leg = [
        mpatches.Patch(
            color=C_SPOT_IN,
            alpha=0.6,
            label=f"In-tissue Visium spots (n={len(in_t):,})",
        ),
        mpatches.Patch(
            color=C_SPOT_OUT,
            alpha=0.5,
            label=f"Out-of-tissue spots (n={len(out_t):,})",
        ),
    ]
    ax_a.legend(handles=leg, loc="lower right", fontsize=7.5, framealpha=0.85, edgecolor="#CCCCCC")
    ax_a.set_xlabel("Pixel position (x)", fontsize=8.5)
    ax_a.set_ylabel("Pixel position (y)", fontsize=8.5)
    ax_a.tick_params(labelsize=7.5)
    ax_a.set_title(
        "H\u0026E tissue section with Visium spot overlay\n"
        "(55 \u03bcm spot diameter, 100 \u03bcm center-to-center)",
        fontsize=9.5,
        pad=5,
    )

    ax_b = fig.add_axes([0.49, 0.48, 0.235, 0.38])
    xi0, yi0 = int(bx0), int(by0)
    xi1, yi1 = int(bx1), int(by1)
    zoom_img = img[yi0:yi1, xi0:xi1]
    ax_b.imshow(zoom_img, origin="upper", aspect="auto")

    mask = (
        (in_t["x_hires"] >= bx0)
        & (in_t["x_hires"] <= bx1)
        & (in_t["y_hires"] >= by0)
        & (in_t["y_hires"] <= by1)
    )
    zs = in_t[mask].copy()
    zs["zx"] = zs["x_hires"] - bx0
    zs["zy"] = zs["y_hires"] - by0
    ax_b.scatter(
        zs["zx"],
        zs["zy"],
        s=30,
        c=C_SPOT_IN,
        alpha=0.35,
        linewidths=0.6,
        edgecolors="white",
        zorder=3,
    )

    ax_b.set_xlim(0, xi1 - xi0)
    ax_b.set_ylim(yi1 - yi0, 0)
    ax_b.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
    for sp in ax_b.spines.values():
        sp.set_edgecolor(C_BOX)
        sp.set_linewidth(2)
    ax_b.set_title("(b) Zoomed: Visium spot grid on tissue", fontsize=9, color=C_BOX, pad=4)

    fig.add_artist(
        mpatches.FancyArrowPatch(
        (0.462, 0.67),
        (0.488, 0.67),
            transform=fig.transFigure,
            arrowstyle="->",
            color=C_BOX,
            lw=1.8,
            mutation_scale=16,
        )
    )

    fig.text(
        0.735,
        0.90,
        "(c) Representative 224\u00d7224 pixel patches (65 \u00b5m FOV each)",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color=C_BORDER,
    )

    pw, ph = 0.07, 0.115
    xs0, ys0 = 0.495, 0.07
    gap_x, gap_y = 0.012, 0.015

    for idx, pf in enumerate(patches[:6]):
        row = idx // 3
        col = idx % 3
        px = xs0 + col * (pw + gap_x)
        py = ys0 + (1 - row) * (ph + gap_y)
        ax_p = fig.add_axes([px, py, pw, ph])
        p_img = np.array(Image.open(pf).convert("RGB"))
        ax_p.imshow(p_img)
        ax_p.axis("off")
        for sp in ax_p.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#999999")
            sp.set_linewidth(0.7)
        if idx == 0:
            ax_p.text(
                0.03,
                0.95,
                "224 px",
                transform=ax_p.transAxes,
                fontsize=6,
                va="top",
                color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#333333", alpha=0.75),
            )

    fig.text(
        0.735,
        0.045,
        f"{n_qc:,} QC-passed patches | 224\u00d7224 px | ~65 \u00b5m FOV | OpenCLIP normalization",
        ha="center",
        fontsize=7.5,
        color="#777777",
        style="italic",
    )

    out_path = out_dir / "figure2_1_hnscc_v2.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hires", required=True)
    ap.add_argument("--spots", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument(
        "--meta",
        default=None,
        help="Optional: patch_extraction_65um_meta.csv for tissue_frac ranking",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args)
