#!/usr/bin/env python3
"""Batch-export ST figure components for many samples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATA_SELECT = Path("/home/students/hbae/data/Data select")
PROC65 = Path("/home/students/hbae/data/Processed_Data_65um_224")
PROC = Path("/home/students/hbae/data/Processed_Data")
FIGURES = Path("/home/students/hbae/figures")
SCRIPT = Path(__file__).resolve().parent / "export_st_figure_components.py"

SAMPLES = [
    "GSM6339637_s7",
    "GSM6339638_s8",
    "GSM6339640_s10",
    "GSM6339641_s11",
    "GSM6339642_s12",
    "GSM7998252",
    "GSM7998253",
    "GSM7998254",
    "GSM7998255",
    "GSM7998256",
    "GSM7998257",
    "GSM7998259",
    "GSM8633891_21_00757_LI_SING",
    "GSM8633892_21_00758_LI_SING",
    "GSM8633893_21_01569_LI_SING",
    "GSM8633895_21_01586_LI_SING",
    "GSM8633896_21_01587_LI_SING",
    "P5",
    "Patient1",
    "Patient2",
    "Patient3",
    "Patient4",
    "Visium_S01",
]


def find_sample_root(sample_id: str) -> tuple[str, Path] | None:
    for ds_dir in sorted(DATA_SELECT.iterdir()):
        if not ds_dir.is_dir():
            continue
        cand = ds_dir / sample_id
        if (cand / "spatial/tissue_hires_image.png").exists():
            return ds_dir.name, cand
        for hires in cand.rglob("tissue_hires_image.png"):
            return ds_dir.name, hires.parent.parent
    return None


def export_one(sample_id: str) -> tuple[bool, str]:
    found = find_sample_root(sample_id)
    if not found:
        return False, "sample root not found"
    dataset, root = found
    hires = root / "spatial/tissue_hires_image.png"
    if not hires.exists():
        hires = next(root.rglob("tissue_hires_image.png"), None)
    scale = root / "spatial/scalefactors_json.json"
    if not scale.exists():
        scale = next(root.rglob("scalefactors_json.json"), None)
    spots = root / "spatial/tissue_positions_list.csv"
    st_h5 = root / "st.h5ad"
    patches = PROC65 / dataset / sample_id / "patches"
    st_norm = PROC / dataset / sample_id / "st_norm.h5ad"

    if not hires or not hires.exists():
        return False, "no hires image"
    if not scale or not scale.exists():
        return False, "no scalefactors"
    if not patches.exists():
        return False, f"no patches dir: {patches}"
    if not st_norm.exists():
        return False, f"no st_norm: {st_norm}"

    out = FIGURES / sample_id
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--out_dir",
        str(out),
        "--sample_id",
        sample_id,
        "--hires",
        str(hires),
        "--scale",
        str(scale),
        "--patches",
        str(patches),
        "--st_h5ad",
        str(st_norm),
    ]
    if spots.exists():
        cmd += ["--spots", str(spots)]
    elif st_h5.exists():
        cmd += ["--spots_h5ad", str(st_h5)]
    else:
        return False, "no spots CSV or st.h5ad"

    r = subprocess.run(
        cmd,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-500:]
    return True, r.stdout.strip().split("\n")[-1]


def main() -> None:
    ok, fail = [], []
    for sid in SAMPLES:
        print(f"Exporting {sid}...", flush=True)
        success, msg = export_one(sid)
        if success:
            ok.append((sid, msg))
            print(f"  OK: {msg}")
        else:
            fail.append((sid, msg))
            print(f"  FAIL: {msg}")

    print(f"\nDone: {len(ok)} ok, {len(fail)} failed")
    if fail:
        print("Failed:")
        for sid, msg in fail:
            print(f"  {sid}: {msg}")


if __name__ == "__main__":
    main()
