#!/usr/bin/env python3
"""Export HE vs mask-overlay side-by-side for top-N TCGA slides by IoU."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont


def otsu_mask(rgb: np.ndarray) -> np.ndarray:
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
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


def green_overlay(he: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = he.astype(np.float32)
    green = np.zeros_like(out)
    green[:, :, 1] = 255.0
    m = mask[..., None]
    return np.clip(out * (1 - alpha * m) + green * (alpha * m), 0, 255).astype(np.uint8)


def rank_slides(wsi_root: Path, mask_root: Path, h5_root: Path, require_h5: bool) -> list:
    rows = []
    for d in sorted(mask_root.iterdir()):
        if not d.is_dir() or not (d / "mask.npy").exists():
            continue
        sid = d.name + ".svs"
        if not (wsi_root / sid).exists():
            continue
        if require_h5 and not (h5_root / sid / "tiles.h5").exists():
            continue
        try:
            m = np.load(d / "mask.npy") > 0
            slide = openslide.OpenSlide(str(wsi_root / sid))
            lw, lh = slide.level_dimensions[3]
            if m.shape != (lh, lw):
                slide.close()
                continue
            he = np.array(slide.read_region((0, 0), 3, (lw, lh)).convert("RGB"))
            slide.close()
            o = otsu_mask(he)
            iou = (m & o).sum() / max((m | o).sum(), 1)
            rows.append((float(iou), float(m.mean()), float(o.mean()), sid))
        except Exception:
            pass
    rows.sort(reverse=True)
    return rows


def export_comparison(
    sid: str,
    rank: int,
    iou: float,
    mp: float,
    op: float,
    wsi_root: Path,
    mask_root: Path,
    out_dir: Path,
    max_px: int,
) -> Path:
    stem = sid.replace(".svs", "")
    m = np.load(mask_root / stem / "mask.npy") > 0
    slide = openslide.OpenSlide(str(wsi_root / sid))
    lw, lh = slide.level_dimensions[3]
    he = np.array(slide.read_region((0, 0), 3, (lw, lh)).convert("RGB"))
    slide.close()

    scale = max_px / max(lw, lh)
    dw, dh = max(1, int(lw * scale)), max(1, int(lh * scale))
    he_s = np.array(Image.fromarray(he).resize((dw, dh), Image.Resampling.LANCZOS))
    m_s = (
        np.array(Image.fromarray((m.astype(np.uint8) * 255)).resize((dw, dh), Image.Resampling.NEAREST))
        > 127
    )
    ov = green_overlay(he_s, m_s)

    combo = np.hstack([he_s, ov])
    img = Image.fromarray(combo)
    draw = ImageDraw.Draw(img)
    label = f"#{rank} {stem.split('.')[0]}  IoU={iou:.2f}  mask={mp*100:.0f}%  otsu={op*100:.0f}%"
    draw.rectangle((0, 0, combo.shape[1], 22), fill=(255, 255, 255))
    draw.text((6, 4), label, fill=(0, 0, 0))

    tag = stem.split(".")[0]
    out_path = out_dir / f"{rank:02d}_IoU{iou:.2f}_{tag}_he_vs_overlay.png"
    img.save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_n", type=int, default=15)
    ap.add_argument("--max_px", type=int, default=1800)
    ap.add_argument("--include_ref", default="TCGA-4P-AA8J-01Z-00-DX1")
    ap.add_argument(
        "--out_dir",
        default="/home/students/hbae/figures/figure2_1_components/slide_comparison",
    )
    args = ap.parse_args()

    wsi_root = Path("/shares/bioit-students/ir_students_2020/TCGA-HNSC/WSIs")
    mask_root = Path("/shares/bioit-students/ir_students_2020/TCGA-HNSC/masks")
    h5_root = Path("/home/students/hbae/data/TCGA_HNSC_tiles_144px_h5")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = rank_slides(wsi_root, mask_root, h5_root, require_h5=True)
    selected = rows[: args.top_n]

    if args.include_ref:
        if not any(args.include_ref in s[3] for s in selected):
            for r in rows:
                if args.include_ref in r[3]:
                    selected.append(r)
                    break

    # bottom 3 for contrast
    if len(rows) >= 3:
        for r in rows[-3:]:
            if r not in selected:
                selected.append(r)

    print(f"Writing {len(selected)} images to {out_dir}")
    for i, (iou, mp, op, sid) in enumerate(selected, start=1):
        p = export_comparison(sid, i, iou, mp, op, wsi_root, mask_root, out_dir, args.max_px)
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
