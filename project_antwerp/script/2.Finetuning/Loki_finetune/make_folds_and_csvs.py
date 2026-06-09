#!/usr/bin/env python3
"""
Create 10-fold CV splits from meta CSV (slide-level grouping).

Usage:
    python make_folds_and_csvs.py \
        --meta_csv /project_antwerp/hbae/data/HVG_finetune_meta_gpulab.csv \
        --out_dir ./folds_10fold \
        --n_folds 10 \
        --seed 42 \
        --img_col img_path \
        --label_col label \
        --slide_col slide_id

Output:
    - out_dir/folds.json: fold별 slide_id 리스트
    - out_dir/fold_01_train.csv ... fold_10_train.csv
    - out_dir/fold_01_val.csv ... fold_10_val.csv
    - out_dir/fold_stats.csv: fold별 통계

    python make_folds_and_csvs.py \
        --meta_csv /project_antwerp/hbae/data/0317_training_data_excluding_GSE220978_and_19h1257/0228_HVG_Finetune_meta_excluding_GSE220978_and_19h1257__ordered_like_combined_obs.csv \
        --out_dir /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_10fold \
        --n_folds 10 \
        --seed 42 \
        --img_col img_path \
        --label_col label \
        --slide_col img_idx

    
"""

import argparse
import json
import numpy as np
import pandas as pd
import re
from collections import Counter
from pathlib import Path
from sklearn.model_selection import GroupKFold


def extract_slide_id_from_path(img_path: str) -> str:
    """
    Extract TCGA slide ID from image path.
    
    Tries multiple patterns:
    1. TCGA-XX-XXXX-XX (standard TCGA format)
    2. Any TCGA-* pattern
    3. Extract from filename/directory structure
    
    Returns:
        slide_id or None if not found
    """
    if pd.isna(img_path) or not isinstance(img_path, str):
        return None
    
    # Pattern 1: TCGA-XX-XXXX-XX (first 3 parts, standard)
    pattern1 = r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z])'
    match = re.search(pattern1, img_path)
    if match:
        return match.group(1)
    
    # Pattern 2: Any TCGA-* pattern (more flexible)
    pattern2 = r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]?)'
    match = re.search(pattern2, img_path)
    if match:
        return match.group(1)
    
    # Pattern 3: Extract from directory structure (e.g., /path/to/slide_id/file.jpg)
    # Try to extract from parent directory if it looks like a slide ID
    path_parts = Path(img_path).parts
    for part in reversed(path_parts):  # Check from end
        if re.match(r'TCGA-', part):
            # Extract TCGA ID part
            match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]?)', part)
            if match:
                return match.group(1)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Create 10-fold CV splits from meta CSV (slide-level grouping)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--meta_csv", type=str, required=True,
                       help="Path to input meta CSV file")
    parser.add_argument("--out_dir", type=str, required=True,
                       help="Output directory for fold CSVs")
    parser.add_argument("--n_folds", type=int, default=10,
                       help="Number of folds (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--img_col", type=str, default="img_path",
                       help="Column name for image paths (default: img_path)")
    parser.add_argument("--label_col", type=str, default="label",
                       help="Column name for labels (default: label)")
    parser.add_argument("--slide_col", type=str, default="slide_id",
                       help="Column name for slide IDs (default: slide_id)")
    parser.add_argument("--output_img_col", type=str, default=None,
                       help="Output CSV image column name (default: same as --img_col)")
    parser.add_argument("--output_label_col", type=str, default=None,
                       help="Output CSV label column name (default: same as --label_col)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output directory")
    
    args = parser.parse_args()
    
    meta_csv = Path(args.meta_csv)
    out_dir = Path(args.out_dir)
    
    if not meta_csv.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_csv}")
    
    if out_dir.exists() and not args.overwrite:
        raise RuntimeError(f"Output directory exists: {out_dir}. Use --overwrite to overwrite.")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Creating Fold Splits (Slide-level Grouping)")
    print("=" * 80)
    print(f"Input CSV: {meta_csv}")
    print(f"Output directory: {out_dir}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 1. Load meta CSV
    # ========================================================================
    print("[1] Loading meta CSV...")
    df = pd.read_csv(meta_csv)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check required columns
    if args.img_col not in df.columns:
        raise ValueError(f"Image column '{args.img_col}' not found in CSV")
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in CSV")
    
    # ========================================================================
    # 2. Extract or use slide_id
    # ========================================================================
    print(f"\n[2] Processing slide IDs...")
    
    if args.slide_col in df.columns:
        print(f"  Using existing column '{args.slide_col}'")
        df['slide_id'] = df[args.slide_col].astype(str)
    else:
        # Try common column names first
        potential_slide_cols = ['sample_name', 'slide_id', 'slide', 'sample_id', 'case_id']
        found_col = None
        
        for col in potential_slide_cols:
            if col in df.columns:
                print(f"  Found potential slide ID column: '{col}'")
                found_col = col
                break
        
        if found_col:
            print(f"  Using column '{found_col}' as slide_id")
            df['slide_id'] = df[found_col].astype(str)
        else:
            # Try extracting from img_path
            print(f"  No slide ID column found. Extracting from '{args.img_col}'...")
            
            # Show sample paths for debugging
            sample_paths = df[args.img_col].dropna().head(5).tolist()
            print(f"  Sample paths:")
            for i, path in enumerate(sample_paths, 1):
                print(f"    {i}. {path}")
            
            df['slide_id'] = df[args.img_col].apply(extract_slide_id_from_path)
            
            extracted_count = df['slide_id'].notna().sum()
            print(f"  Extracted slide IDs: {extracted_count} / {len(df)} rows")
            
            if extracted_count < len(df) * 0.9:  # Warn if < 90% success
                print(f"  WARNING: Only {extracted_count}/{len(df)} rows have valid slide IDs")
                print(f"  Please check img_path format or provide --slide_col argument")
    
    # Remove rows without slide_id
    df_clean = df[df['slide_id'].notna()].copy()
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"  Removed {removed} rows without slide_id")
    
    print(f"  Final rows: {len(df_clean)}")
    
    # ========================================================================
    # 3. Create slide-level groups
    # ========================================================================
    print(f"\n[3] Creating slide-level groups...")
    
    unique_slides = df_clean['slide_id'].unique()
    n_slides = len(unique_slides)
    print(f"  Unique slides: {n_slides}")
    
    # Count rows per slide
    slide_counts = df_clean['slide_id'].value_counts()
    print(f"  Rows per slide: min={slide_counts.min()}, max={slide_counts.max()}, mean={slide_counts.mean():.1f}")
    
    if n_slides < args.n_folds:
        raise ValueError(f"Not enough unique slides ({n_slides}) for {args.n_folds} folds")
    
    # ========================================================================
    # 4. Create fold splits (GroupKFold)
    # ========================================================================
    print(f"\n[4] Creating {args.n_folds}-fold splits (GroupKFold)...")
    
    # GroupKFold requires groups array (same length as data)
    groups = df_clean['slide_id'].values
    
    # Create GroupKFold
    gkf = GroupKFold(n_splits=args.n_folds)
    
    # Get splits
    splits = list(gkf.split(df_clean, groups=groups))
    
    fold_slides = {}  # fold_idx -> list of slide_ids
    fold_stats = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_slides = df_clean.iloc[train_idx]['slide_id'].unique()
        val_slides = df_clean.iloc[val_idx]['slide_id'].unique()
        
        fold_slides[fold_idx + 1] = {
            "train_slides": sorted(train_slides.tolist()),
            "val_slides": sorted(val_slides.tolist())
        }
        
        train_rows = len(train_idx)
        val_rows = len(val_idx)
        
        fold_stats.append({
            "fold": fold_idx + 1,
            "n_train_slides": len(train_slides),
            "n_val_slides": len(val_slides),
            "n_train_rows": train_rows,
            "n_val_rows": val_rows,
            "train_ratio": train_rows / len(df_clean),
            "val_ratio": val_rows / len(df_clean),
        })
        
        print(f"  Fold {fold_idx + 1}: train={len(train_slides)} slides ({train_rows} rows), "
              f"val={len(val_slides)} slides ({val_rows} rows)")
    
    # ========================================================================
    # 5. Save fold CSVs
    # ========================================================================
    print(f"\n[5] Saving fold CSVs...")
    
    # Determine output column names
    output_img_col = args.output_img_col if args.output_img_col else args.img_col
    output_label_col = args.output_label_col if args.output_label_col else args.label_col
    
    if output_img_col != args.img_col or output_label_col != args.label_col:
        print(f"  Renaming columns: '{args.img_col}' -> '{output_img_col}', '{args.label_col}' -> '{output_label_col}'")
    
    for fold_idx in range(1, args.n_folds + 1):
        train_idx, val_idx = splits[fold_idx - 1]
        
        # Train CSV
        train_df = df_clean.iloc[train_idx][[args.img_col, args.label_col]].copy()
        train_df = train_df.rename(columns={args.img_col: output_img_col, args.label_col: output_label_col})
        train_csv = out_dir / f"fold_{fold_idx:02d}_train.csv"
        train_df.to_csv(train_csv, index=False)
        print(f"  Saved: {train_csv} ({len(train_df)} rows)")
        
        # Val CSV
        val_df = df_clean.iloc[val_idx][[args.img_col, args.label_col]].copy()
        val_df = val_df.rename(columns={args.img_col: output_img_col, args.label_col: output_label_col})
        val_csv = out_dir / f"fold_{fold_idx:02d}_val.csv"
        val_df.to_csv(val_csv, index=False)
        print(f"  Saved: {val_csv} ({len(val_df)} rows)")
    
    # ========================================================================
    # 6. Save metadata
    # ========================================================================
    print(f"\n[6] Saving metadata...")
    
    # Save folds.json
    folds_json = out_dir / "folds.json"
    with open(folds_json, 'w') as f:
        json.dump(fold_slides, f, indent=2)
    print(f"  Saved: {folds_json}")
    
    # Save fold_stats.csv
    stats_df = pd.DataFrame(fold_stats)
    stats_csv = out_dir / "fold_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  Saved: {stats_csv}")
    
    print("\n" + "=" * 80)
    print("✅ Fold splits created successfully!")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"  - folds.json: Fold metadata")
    print(f"  - fold_stats.csv: Fold statistics")
    print(f"  - fold_*_train.csv: Training CSVs ({args.n_folds} files)")
    print(f"  - fold_*_val.csv: Validation CSVs ({args.n_folds} files)")
    print("=" * 80)


if __name__ == "__main__":
    main()

