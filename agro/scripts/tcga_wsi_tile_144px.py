#!/usr/bin/env python3
"""
Tile TCGA WSI (.svs/.tif) into 144x144 patches and save under /home/students/hbae/data.

This script is designed to be safe by default:
- uses precomputed tissue masks (level-3) when provided
- falls back to a thumbnail-based mask if no mask exists
- only saves tiles that overlap tissue above a threshold
- caps tiles per slide (configurable)

Example:
  python3 scripts/tcga_wsi_tile_144px.py \
    --wsi_dir /shares/bioit-students/ir_students_2020/TCGA-HNSC/WSIs \
    --out_root /home/students/hbae/data/TCGA_HNSC_tiles_144px_h5 \
    --tile_size 144 --stride 144 --max_tiles_per_slide 50000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

import h5py
import openslide


@dataclass(frozen=True)
class SlideInfo:
    path: Path
    slide_id: str
    width: int
    height: int
    mpp_x: float | None
    mpp_y: float | None
    level3_width: int
    level3_height: int
    level3_downsample: float


def iter_wsi_files(wsi_dir: Path) -> list[Path]:
    exts = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
    files = [p for p in sorted(wsi_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return files


def read_slide_info(slide_path: Path) -> SlideInfo:
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.dimensions
    level3_w, level3_h = slide.level_dimensions[-1]
    level3_down = float(slide.level_downsamples[-1])
    props = slide.properties
    mpp_x = props.get("openslide.mpp-x")
    mpp_y = props.get("openslide.mpp-y")
    try:
        mpp_x_f = float(mpp_x) if mpp_x is not None else None
    except Exception:
        mpp_x_f = None
    try:
        mpp_y_f = float(mpp_y) if mpp_y is not None else None
    except Exception:
        mpp_y_f = None
    slide.close()
    return SlideInfo(
        path=slide_path,
        slide_id=slide_path.name,
        width=int(w),
        height=int(h),
        mpp_x=mpp_x_f,
        mpp_y=mpp_y_f,
        level3_width=int(level3_w),
        level3_height=int(level3_h),
        level3_downsample=level3_down,
    )


def _thumbnail_to_mask(thumb_rgb: np.ndarray) -> np.ndarray:
    """
    Build a simple tissue mask from a thumbnail.
    Returns a boolean array (H, W), True = tissue.
    """
    # grayscale
    r = thumb_rgb[..., 0].astype(np.float32)
    g = thumb_rgb[..., 1].astype(np.float32)
    b = thumb_rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Otsu threshold (implemented here to avoid extra deps)
    hist, _ = np.histogram(gray.astype(np.uint8), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thr = 200  # fallback (bright background)
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

    # tissue tends to be darker than background
    mask = gray < float(thr)

    # remove obvious artifacts: very dark (pen marks) also kept as tissue (fine)
    return mask


def build_tissue_mask(
    slide: openslide.OpenSlide,
    thumb_max_side: int = 2048,
) -> tuple[np.ndarray, float]:
    """
    Returns (mask, downsample_factor).
    downsample_factor maps level-0 coordinates -> thumbnail coordinates:
      x_thumb = x_level0 / downsample_factor
    """
    w0, h0 = slide.dimensions
    scale = max(w0, h0) / float(thumb_max_side)
    if scale < 1.0:
        scale = 1.0
    w_t = max(1, int(round(w0 / scale)))
    h_t = max(1, int(round(h0 / scale)))

    thumb = slide.get_thumbnail((w_t, h_t)).convert("RGB")
    arr = np.asarray(thumb)
    mask = _thumbnail_to_mask(arr)
    return mask, float(scale)


def load_precomputed_mask(mask_root: Path, slide_id: str, expected_wh: tuple[int, int]) -> np.ndarray | None:
    """
    Loads precomputed mask as boolean array (H, W) if available.
    expected_wh is (W, H) to validate.
    """
    slide_key = slide_id
    if slide_key.lower().endswith(".svs"):
        slide_key = slide_key[:-4]

    # Preferred: directory with mask.npy / mask.png
    d = mask_root / slide_key
    npy = d / "mask.npy"
    png = d / "mask.png"

    arr: np.ndarray | None = None
    if npy.exists():
        a = np.load(npy)
        if a.ndim == 2:
            arr = a
        elif a.ndim == 3:
            arr = a[..., 0]
    elif png.exists():
        im = Image.open(png).convert("L")
        arr = np.asarray(im)

    # Fallback: <slide>_mask.png in root
    if arr is None:
        direct_png = mask_root / f"{slide_key}_mask.png"
        if direct_png.exists():
            im = Image.open(direct_png).convert("L")
            arr = np.asarray(im)

    if arr is None:
        return None

    # binarize (supports 0/1 or 0/255)
    mask = arr.astype(np.uint8) > 0
    w_exp, h_exp = expected_wh
    if mask.shape[0] != h_exp or mask.shape[1] != w_exp:
        return None
    return mask


def iter_tile_coords(width: int, height: int, tile_size: int, stride: int) -> Iterator[tuple[int, int]]:
    # upper-left coords in level-0 space
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            yield x, y


def mask_coverage(
    mask: np.ndarray,
    downsample: float,
    x0: int,
    y0: int,
    tile_size: int,
) -> float:
    """
    Estimate tissue coverage in a tile using a mask array in a downsampled space.
    """
    x1 = x0 + tile_size
    y1 = y0 + tile_size
    mx0 = int(math.floor(x0 / downsample))
    my0 = int(math.floor(y0 / downsample))
    mx1 = int(math.ceil(x1 / downsample))
    my1 = int(math.ceil(y1 / downsample))
    mx0 = max(0, min(mask.shape[1], mx0))
    mx1 = max(0, min(mask.shape[1], mx1))
    my0 = max(0, min(mask.shape[0], my0))
    my1 = max(0, min(mask.shape[0], my1))
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    sub = mask[my0:my1, mx0:mx1]
    return float(sub.mean()) if sub.size else 0.0


def read_tile_rgb(slide: openslide.OpenSlide, x: int, y: int, tile_size: int) -> np.ndarray:
    region = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
    return np.asarray(region)


def _ensure_h5_datasets(
    h5: h5py.File,
    tile_size: int,
) -> tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset]:
    if "images" not in h5:
        h5.create_dataset(
            "images",
            shape=(0, tile_size, tile_size, 3),
            maxshape=(None, tile_size, tile_size, 3),
            chunks=(256, tile_size, tile_size, 3),
            dtype=np.uint8,
            compression="lzf",
            shuffle=True,
        )
    if "coords" not in h5:
        h5.create_dataset(
            "coords",
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(8192, 2),
            dtype=np.int32,
            compression="lzf",
            shuffle=True,
        )
    if "tissue_frac" not in h5:
        h5.create_dataset(
            "tissue_frac",
            shape=(0,),
            maxshape=(None,),
            chunks=(8192,),
            dtype=np.float32,
            compression="lzf",
            shuffle=True,
        )
    return h5["images"], h5["coords"], h5["tissue_frac"]


def _append_rows(
    ds_images: h5py.Dataset,
    ds_coords: h5py.Dataset,
    ds_frac: h5py.Dataset,
    imgs: list[np.ndarray],
    coords: list[tuple[int, int]],
    fracs: list[float],
) -> None:
    if not imgs:
        return
    n0 = ds_images.shape[0]
    n = len(imgs)
    ds_images.resize((n0 + n, *ds_images.shape[1:]))
    ds_coords.resize((n0 + n, 2))
    ds_frac.resize((n0 + n,))

    ds_images[n0 : n0 + n] = np.stack(imgs, axis=0)
    ds_coords[n0 : n0 + n] = np.asarray(coords, dtype=np.int32)
    ds_frac[n0 : n0 + n] = np.asarray(fracs, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tile TCGA WSI into 144px patches (tissue-filtered).")
    ap.add_argument("--wsi_dir", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, default=Path("/home/students/hbae/data/TCGA_HNSC_tiles_144px_h5"))
    ap.add_argument(
        "--mask_root",
        type=Path,
        default=Path("/shares/bioit-students/ir_students_2020/TCGA-HNSC/masks"),
        help="Root folder containing precomputed masks (level-3).",
    )
    ap.add_argument("--tile_size", type=int, default=144)
    ap.add_argument("--stride", type=int, default=144)
    ap.add_argument("--thumb_max_side", type=int, default=2048)
    ap.add_argument("--tissue_thresh", type=float, default=0.30, help="Min tissue fraction in tile (0-1).")
    ap.add_argument("--max_tiles_per_slide", type=int, default=50000, help="Safety cap per slide.")
    ap.add_argument("--limit_slides", type=int, default=0, help="If >0, only process first N slides.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip slide if output dir exists and non-empty.")
    ap.add_argument("--flush_every", type=int, default=512, help="Write buffer size (tiles) before flushing to HDF5.")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "manifest_tiles.csv"
    os.makedirs(args.out_root, exist_ok=True)

    wsi_files = iter_wsi_files(args.wsi_dir)
    if args.limit_slides and args.limit_slides > 0:
        wsi_files = wsi_files[: args.limit_slides]
    if not wsi_files:
        raise SystemExit(f"No WSI files found in {args.wsi_dir}")

    # Create/append manifest
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "slide_id",
                    "tile_relpath",
                    "x_level0",
                    "y_level0",
                    "tile_size_px",
                    "stride_px",
                    "tissue_frac_est",
                    "mpp_x",
                    "mpp_y",
                    "w0",
                    "h0",
                ]
            )

        for idx, slide_path in enumerate(wsi_files, start=1):
            info = read_slide_info(slide_path)
            slide_out = args.out_root / info.slide_id
            slide_out.mkdir(parents=True, exist_ok=True)
            if args.skip_existing:
                try:
                    # consider done if tiles.h5 exists and has non-zero images
                    h5p = slide_out / "tiles.h5"
                    if h5p.exists() and h5p.stat().st_size > 0:
                        with h5py.File(h5p, "r") as h5:
                            if "images" in h5 and h5["images"].shape[0] > 0:
                                print(f"[{idx}/{len(wsi_files)}] skip existing: {info.slide_id}")
                                continue
                    # otherwise if directory has anything at all, also skip (legacy)
                    if any(slide_out.iterdir()):
                        print(f"[{idx}/{len(wsi_files)}] skip existing: {info.slide_id}")
                        continue
                except Exception:
                    pass

            print(f"[{idx}/{len(wsi_files)}] {info.slide_id}  dim={info.width}x{info.height}  mpp=({info.mpp_x},{info.mpp_y})")

            slide = openslide.OpenSlide(str(slide_path))
            mask = load_precomputed_mask(
                args.mask_root,
                slide_id=info.slide_id,
                expected_wh=(info.level3_width, info.level3_height),
            )
            if mask is not None:
                down = info.level3_downsample
            else:
                mask, down = build_tissue_mask(slide, thumb_max_side=args.thumb_max_side)

            h5_path = slide_out / "tiles.h5"
            h5 = h5py.File(h5_path, "a")
            ds_images, ds_coords, ds_frac = _ensure_h5_datasets(h5, args.tile_size)
            h5.attrs["slide_id"] = info.slide_id
            h5.attrs["tile_size"] = int(args.tile_size)
            h5.attrs["stride"] = int(args.stride)
            h5.attrs["tissue_thresh"] = float(args.tissue_thresh)
            h5.attrs["mpp_x"] = float(info.mpp_x) if info.mpp_x is not None else np.nan
            h5.attrs["mpp_y"] = float(info.mpp_y) if info.mpp_y is not None else np.nan
            h5.attrs["w0"] = int(info.width)
            h5.attrs["h0"] = int(info.height)
            h5.attrs["mask_space"] = "level3" if mask is not None else "thumbnail"
            h5.attrs["mask_downsample"] = float(down)

            saved = 0
            scanned = 0
            buf_imgs: list[np.ndarray] = []
            buf_coords: list[tuple[int, int]] = []
            buf_fracs: list[float] = []
            for x0, y0 in iter_tile_coords(info.width, info.height, args.tile_size, args.stride):
                scanned += 1
                frac = mask_coverage(mask, down, x0, y0, args.tile_size)
                if frac < args.tissue_thresh:
                    continue
                tile = read_tile_rgb(slide, x0, y0, args.tile_size)
                # additional cheap filter: drop near-white tiles
                if float(tile.mean()) > 235.0:
                    continue

                buf_imgs.append(tile)
                buf_coords.append((x0, y0))
                buf_fracs.append(float(frac))
                if len(buf_imgs) >= args.flush_every:
                    _append_rows(ds_images, ds_coords, ds_frac, buf_imgs, buf_coords, buf_fracs)
                    h5.flush()
                    buf_imgs.clear()
                    buf_coords.clear()
                    buf_fracs.clear()

                w.writerow(
                    [
                        info.slide_id,
                        str(Path(info.slide_id) / "tiles.h5"),
                        x0,
                        y0,
                        args.tile_size,
                        args.stride,
                        f"{frac:.4f}",
                        info.mpp_x if info.mpp_x is not None else "",
                        info.mpp_y if info.mpp_y is not None else "",
                        info.width,
                        info.height,
                    ]
                )
                saved += 1
                if saved >= args.max_tiles_per_slide:
                    break

            slide.close()
            if buf_imgs:
                _append_rows(ds_images, ds_coords, ds_frac, buf_imgs, buf_coords, buf_fracs)
                h5.flush()
            h5.close()
            print(f"  saved={saved} (tissue_thresh={args.tissue_thresh}, stride={args.stride}, tile={args.tile_size})")

    print(f"Done. Output root: {args.out_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

