#!/usr/bin/env python3
"""
make_figure2_1_hnscc.py
========================
Figure 2.1: H&E WSI characteristics and patch extraction pipeline
Using GSM7998257 (GSE252265, tongue SCC, HNSCC Visium)

Usage:
    python make_figure2_1_hnscc.py \\
        --hires   "/home/students/hbae/data/Data select/GSE252265/GSM7998257/spatial/tissue_hires_image.png" \\
        --spots   /home/students/hbae/data/Data\ select/GSE252265/GSM7998257/spatial/tissue_positions_list.csv \
        --scale   /home/students/hbae/data/Data\ select/GSE252265/GSM7998257/spatial/scalefactors_json.json \
        --patches /home/students/hbae/data/Processed_Data_65um_224/GSE252265/GSM7998257/patches \
        --out_dir /home/students/hbae/figures
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

# ── Color palette ──────────────────────────────────────────────────────
C_SPOT_IN = "#2E86AB"  # blue — in-tissue spots
C_SPOT_OUT = "#CCCCCC"  # grey — out-of-tissue
C_BOX = "#E74C3C"  # red — zoom box
C_BORDER = "#333333"


def load_spots(spots_path, scale_path):
    """Load spot coordinates scaled to hires image space."""
    with open(scale_path) as f:
        scale = json.load(f)
    scalef = scale["tissue_hires_scalef"]

    df = pd.read_csv(spots_path)
    df = df.rename(
        columns={
            "pxl_row_in_fullres": "pxl_row_fullres",
            "pxl_col_in_fullres": "pxl_col_fullres",
        }
    )
    df["y_hires"] = df["pxl_row_fullres"] * scalef
    df["x_hires"] = df["pxl_col_fullres"] * scalef
    return df, scalef, scale


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading H&E hires image...")
    img = np.array(Image.open(args.hires).convert("RGB"))
    H, W = img.shape[:2]
    print(f"  Image size: {W} x {H} px")

    print("Loading spot coordinates...")
    df, scalef, scale_json = load_spots(args.spots, args.scale)
    in_tissue = df[df["in_tissue"] == 1]
    out_tissue = df[df["in_tissue"] == 0]
    print(f"  In-tissue spots: {len(in_tissue)}, Out: {len(out_tissue)}")

    # ── Load sample patches ────────────────────────────────────────────
    patch_dir = Path(args.patches)
    patch_files = sorted(patch_dir.glob("*.png"))
    print(f"  Patches found: {len(patch_files)}")
    random.seed(42)
    sample_patches = random.sample(patch_files, min(6, len(patch_files)))

    # ── Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    # Title
    fig.text(
        0.5,
        0.97,
        "Figure 2.1. H\u0026E Whole-Slide Image Characteristics and Patch Extraction Pipeline",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=C_BORDER,
    )
    fig.text(
        0.5,
        0.935,
        "Representative HNSCC Visium sample: GSM7998257 (GSE252265, tongue SCC)",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#666666",
        style="italic",
    )

    # ── Panel (a): Tissue section + spot overlay ───────────────────────
    ax_a = fig.add_axes([0.03, 0.08, 0.44, 0.82])
    ax_a.imshow(img, origin="upper")
    ax_a.set_xlim(0, W)
    ax_a.set_ylim(H, 0)

    # out-of-tissue spots (small, grey)
    ax_a.scatter(
        out_tissue["x_hires"],
        out_tissue["y_hires"],
        s=3,
        c=C_SPOT_OUT,
        alpha=0.3,
        linewidths=0,
        zorder=2,
    )
    # in-tissue spots (colored, semi-transparent)
    ax_a.scatter(
        in_tissue["x_hires"],
        in_tissue["y_hires"],
        s=8,
        c=C_SPOT_IN,
        alpha=0.55,
        linewidths=0,
        zorder=3,
    )

    # zoom region box — pick a high-tissue-content area
    cx = in_tissue["x_hires"].median()
    cy = in_tissue["y_hires"].median()
    box_size = min(W, H) * 0.2
    bx0 = max(0, cx - box_size / 2)
    by0 = max(0, cy - box_size / 2)
    rect = Rectangle(
        (bx0, by0),
        box_size,
        box_size,
        linewidth=2,
        edgecolor=C_BOX,
        facecolor="none",
        zorder=5,
    )
    ax_a.add_patch(rect)
    ax_a.text(
        bx0 + box_size / 2,
        by0 - H * 0.015,
        "Zoom region",
        ha="center",
        fontsize=8,
        color=C_BOX,
        fontweight="bold",
    )

    ax_a.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_a.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        color=C_BORDER,
    )

    leg = [
        mpatches.Patch(color=C_SPOT_IN, label=f"In-tissue spots (n={len(in_tissue)})"),
        mpatches.Patch(
            color=C_SPOT_OUT,
            alpha=0.5,
            label=f"Out-of-tissue spots (n={len(out_tissue)})",
        ),
    ]
    ax_a.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.85, edgecolor="#AAAAAA")

    ax_a.set_xlabel("Pixel position (x)", fontsize=9)
    ax_a.set_ylabel("Pixel position (y)", fontsize=9)
    ax_a.tick_params(labelsize=8)
    ax_a.set_title(
        "H\u0026E tissue section with Visium spot overlay\n"
        f"({len(in_tissue)} in-tissue spots, ~55 \u03bcm diameter)",
        fontsize=10,
        pad=6,
    )

    # ── Panel (b): Zoom + patch grid ──────────────────────────────────
    ax_zoom = fig.add_axes([0.50, 0.52, 0.23, 0.38])
    x0i, y0i = int(bx0), int(by0)
    x1i = min(W, int(bx0 + box_size))
    y1i = min(H, int(by0 + box_size))
    zoom_img = img[y0i:y1i, x0i:x1i]
    ax_zoom.imshow(zoom_img, origin="upper")

    mask = (
        (in_tissue["x_hires"] >= bx0)
        & (in_tissue["x_hires"] <= bx0 + box_size)
        & (in_tissue["y_hires"] >= by0)
        & (in_tissue["y_hires"] <= by0 + box_size)
    )
    zoom_spots = in_tissue[mask].copy()
    zoom_spots["zx"] = zoom_spots["x_hires"] - bx0
    zoom_spots["zy"] = zoom_spots["y_hires"] - by0
    ax_zoom.scatter(
        zoom_spots["zx"],
        zoom_spots["zy"],
        s=40,
        c=C_SPOT_IN,
        alpha=0.5,
        linewidths=0.8,
        edgecolors="white",
        zorder=3,
    )

    ax_zoom.set_xlim(0, x1i - x0i)
    ax_zoom.set_ylim(y1i - y0i, 0)
    ax_zoom.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor(C_BOX)
        spine.set_linewidth(2)
    ax_zoom.set_title(
        "Zoomed tissue region\nwith Visium spot grid", fontsize=9, color=C_BOX
    )
    ax_zoom.text(
        0.02,
        0.97,
        "(b)",
        transform=ax_zoom.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        color=C_BORDER,
    )

    fig.add_artist(
        mpatches.FancyArrowPatch(
            (0.47, 0.72),
            (0.50, 0.72),
            transform=fig.transFigure,
            arrowstyle="->",
            color=C_BOX,
            lw=2,
            mutation_scale=18,
        )
    )

    # ── Panel (c): Sample patches grid ────────────────────────────────
    n_cols = 3
    n_rows = 2
    patch_w = 0.07
    patch_h = 0.12
    x_start = 0.50
    y_start = 0.10

    fig.text(
        x_start + (n_cols * (patch_w + 0.01)) / 2,
        y_start + n_rows * (patch_h + 0.015) + 0.03,
        "(c) Extracted 224\u00d7224 pixel patches (65 \u00b5m FOV)",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color=C_BORDER,
    )

    for idx, pf in enumerate(sample_patches[: n_cols * n_rows]):
        row = idx // n_cols
        col = idx % n_cols
        px = x_start + col * (patch_w + 0.012)
        py = y_start + (n_rows - 1 - row) * (patch_h + 0.015)
        ax_p = fig.add_axes([px, py, patch_w, patch_h])
        p_img = np.array(Image.open(pf).convert("RGB"))
        ax_p.imshow(p_img)
        ax_p.axis("off")
        for spine in ax_p.spines.values():
            spine.set_edgecolor("#999999")
            spine.set_linewidth(0.8)
        if idx == 0:
            ax_p.text(
                0.02,
                0.97,
                "224 px",
                transform=ax_p.transAxes,
                fontsize=6.5,
                va="top",
                color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#333333", alpha=0.7),
            )

    fig.text(
        x_start + 0.01,
        y_start - 0.03,
        "Each patch: 224\u00d7224 px | ~65 \u00b5m physical FOV | Preprocessed with OpenCLIP normalization",
        fontsize=8,
        color="#666666",
        style="italic",
    )

    out_path = out_dir / "figure2_1_hnscc_wsi_characteristics.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hires", required=True, help="Path to tissue_hires_image.png")
    ap.add_argument("--spots", required=True, help="Path to tissue_positions_list.csv")
    ap.add_argument("--scale", required=True, help="Path to scalefactors_json.json")
    ap.add_argument("--patches", required=True, help="Path to 224px patches directory")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()
    main(args)
