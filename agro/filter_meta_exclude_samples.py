#!/usr/bin/env python3
"""
Filter Loki training meta.csv by excluding specific samples/datasets.

Default exclusions:
- dataset == GSE220978 and sample_name in {Patient1, Patient2, Patient3, Patient4}
- dataset == Zenodo and sample_name == 19h1257

Writes a new meta CSV (same columns) that can be fed into build_loki_training_data.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in_meta",
        default="/home/students/hbae/data/Processed_Data/training_data/meta.csv",
        help="Input meta.csv path",
    )
    p.add_argument(
        "--out_meta",
        default="/home/students/hbae/data/Processed_Data/training_data/meta_excluding_GSE220978_and_19h1257.csv",
        help="Output filtered meta.csv path",
    )
    args = p.parse_args()

    in_path = Path(args.in_meta)
    out_path = Path(args.out_meta)
    if not in_path.exists():
        raise FileNotFoundError(f"meta.csv not found: {in_path}")

    df = pd.read_csv(in_path)
    required = {"dataset", "sample_name", "img_idx", "img_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"meta is missing columns: {sorted(missing)}")

    n0 = len(df)
    ex_patients = {"Patient1", "Patient2", "Patient3", "Patient4"}

    mask_exclude = (
        ((df["dataset"] == "GSE220978") & (df["sample_name"].isin(ex_patients)))
        | ((df["dataset"] == "Zenodo") & (df["sample_name"] == "19h1257"))
    )

    removed = df[mask_exclude]
    kept = df[~mask_exclude].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    print(f"Input rows: {n0:,}")
    print(f"Removed rows: {len(removed):,}")
    if len(removed) > 0:
        print("Removed breakdown:")
        print(removed.groupby(["dataset", "sample_name"]).size().sort_values(ascending=False).to_string())
    print(f"Kept rows: {len(kept):,}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

