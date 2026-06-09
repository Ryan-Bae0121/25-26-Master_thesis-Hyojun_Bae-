#!/usr/bin/env python3
"""
Compute µm per pixel in tissue_hires_image from scalefactors_json.json.

Standard Visium: spot diameter in real space = 55 µm.
  fullres:  spot_diameter_fullres [px] = 55 µm  =>  1 fullres px = 55/spot_diameter_fullres µm
  hires:    x_hires = x_fullres * tissue_hires_scalef  =>  1 fullres px = 1/tissue_hires_scalef hire px
  =>  1 hire px = (55/spot_diameter_fullres) / tissue_hires_scalef  µm
      i.e.  µm_per_hires_px = 55 / (tissue_hires_scalef * spot_diameter_fullres)
"""

import json
from pathlib import Path

SPOT_DIAMETER_UM = 55.0  # standard Visium


def main():
    import glob
    root = Path("/home/students/hbae/data")
    files = sorted(root.rglob("scalefactors_json.json"))[:25]
    # dedupe by content path (some are in spatial/, some at sample root)
    seen = set()
    rows = []
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
        except Exception:
            continue
        sf = d.get("tissue_hires_scalef")
        sd = d.get("spot_diameter_fullres")
        if sf is None or sd is None:
            continue
        key = (round(sf, 6), round(sd, 2))
        if key in seen:
            continue
        seen.add(key)
        um_per_px = SPOT_DIAMETER_UM / (sf * sd)
        patch_20_um = 20 * um_per_px
        sample = f.relative_to(root).parts[0] if root in f.parents else str(f.parent)
        if "Data select" in str(f):
            sample = str(f.relative_to(root))
        rows.append((sample, sf, sd, um_per_px, patch_20_um))

    print("SPOT_DIAMETER_UM =", SPOT_DIAMETER_UM)
    print("µm per hires px = 55 / (tissue_hires_scalef * spot_diameter_fullres)")
    print()
    print(f"{'sample':<55} {'tissue_hires':>12} {'spot_d_fullres':>14} {'µm/px':>8} {'20px patch µm':>14}")
    print("-" * 110)
    for sample, sf, sd, um_per_px, patch_20 in rows:
        print(f"{sample[:54]:<55} {sf:>12.6f} {sd:>14.2f} {um_per_px:>8.3f} {patch_20:>14.2f}")
    print()
    if rows:
        avg_um = sum(r[3] for r in rows) / len(rows)
        avg_20 = sum(r[4] for r in rows) / len(rows)
        print(f"Average µm/hires px: {avg_um:.3f}  =>  20×20 px patch ≈ {avg_20:.1f} µm")


if __name__ == "__main__":
    main()
