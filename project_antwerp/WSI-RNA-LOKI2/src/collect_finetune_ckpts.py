#!/usr/bin/env python3
"""
Collect checkpoint paths from 10-fold CV fine-tuning runs.

Usage:
    python collect_finetune_ckpts.py \
        --out_root ./finetune_10fold_runs \
        --pattern "*most_recent*.pt"

This script:
1. Scans out_root/fold_*/ for checkpoint files
2. Creates an index CSV/JSON with checkpoint paths and metadata
3. Helps identify which checkpoints to use for evaluation
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def find_checkpoints(root_dir: Path, pattern: str = "*most_recent*.pt") -> list:
    """
    Find checkpoint files matching pattern in fold directories.
    
    Returns:
        List of dicts with checkpoint info
    """
    checkpoints = []
    
    # Look for fold directories
    fold_dirs = sorted(root_dir.glob("fold_*"))
    
    for fold_dir in fold_dirs:
        if not fold_dir.is_dir():
            continue
        
        fold_name = fold_dir.name
        
        # Search for checkpoints
        ckpt_files = sorted(fold_dir.glob(pattern))
        
        if len(ckpt_files) == 0:
            # Try alternative patterns
            alt_patterns = [
                "checkpoints/*.pt",
                "checkpoints/*most_recent*.pt",
                "*.pt",
            ]
            for alt_pattern in alt_patterns:
                ckpt_files = sorted(fold_dir.glob(alt_pattern))
                if len(ckpt_files) > 0:
                    break
        
        # Also check subdirectories
        if len(ckpt_files) == 0:
            for subdir in fold_dir.iterdir():
                if subdir.is_dir():
                    ckpt_files = sorted(subdir.glob(pattern))
                    if len(ckpt_files) > 0:
                        break
        
        # Get most recent checkpoint if multiple found
        if len(ckpt_files) > 0:
            # Sort by modification time
            ckpt_files = sorted(ckpt_files, key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt_path = ckpt_files[0]  # Most recent
            
            mtime = datetime.fromtimestamp(ckpt_path.stat().st_mtime)
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            
            checkpoints.append({
                "fold": fold_name,
                "ckpt_path": str(ckpt_path),
                "ckpt_name": ckpt_path.name,
                "exists": True,
                "mtime": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                "size_mb": round(size_mb, 2),
                "n_ckpts_found": len(ckpt_files),
            })
        else:
            # No checkpoint found
            checkpoints.append({
                "fold": fold_name,
                "ckpt_path": None,
                "ckpt_name": None,
                "exists": False,
                "mtime": None,
                "size_mb": None,
                "n_ckpts_found": 0,
            })
    
    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Collect checkpoint paths from 10-fold CV fine-tuning runs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out_root", type=str, required=True,
                       help="Root output directory from run_finetune_10fold.py")
    parser.add_argument("--pattern", type=str, default="*most_recent*.pt",
                       help="Glob pattern for checkpoint files (default: *most_recent*.pt)")
    parser.add_argument("--output_name", type=str, default="ckpt_index",
                       help="Output filename (without extension, default: ckpt_index)")
    
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    
    if not out_root.exists():
        raise FileNotFoundError(f"Output root not found: {out_root}")
    
    print("=" * 80)
    print("Collecting Checkpoint Paths")
    print("=" * 80)
    print(f"Output root: {out_root}")
    print(f"Pattern: {args.pattern}")
    print("=" * 80)
    print()
    
    # Find checkpoints
    print("[1] Scanning for checkpoints...")
    checkpoints = find_checkpoints(out_root, args.pattern)
    
    if len(checkpoints) == 0:
        print("  WARNING: No checkpoints found!")
        return
    
    # Print summary
    found = sum(1 for c in checkpoints if c["exists"])
    missing = len(checkpoints) - found
    
    print(f"  Total folds: {len(checkpoints)}")
    print(f"  Checkpoints found: {found}")
    print(f"  Missing: {missing}")
    print()
    
    if missing > 0:
        print("  Missing checkpoints:")
        for c in checkpoints:
            if not c["exists"]:
                print(f"    - {c['fold']}")
        print()
    
    # Create DataFrame
    df = pd.DataFrame(checkpoints)
    
    # Sort by fold
    df = df.sort_values("fold")
    
    # Save CSV
    csv_path = out_root / f"{args.output_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[2] Saved checkpoint index: {csv_path}")
    
    # Save JSON
    json_path = out_root / f"{args.output_name}.json"
    with open(json_path, 'w') as f:
        json.dump(checkpoints, f, indent=2)
    print(f"[3] Saved checkpoint index: {json_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Checkpoint Index Summary")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # List valid checkpoints
    valid_ckpts = df[df["exists"] == True]
    if len(valid_ckpts) > 0:
        print("\nValid checkpoints:")
        for _, row in valid_ckpts.iterrows():
            print(f"  {row['fold']}: {row['ckpt_path']}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

