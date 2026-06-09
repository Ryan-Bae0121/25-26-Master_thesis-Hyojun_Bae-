#!/usr/bin/env python3
"""
Analyze performance metrics from 10-fold CV fine-tuning runs and identify the best fold.

Usage:
    python analyze_fold_performance.py \
        --out_root ./finetune_10fold_runs \
        --metric cmc_r10

This script:
1. Parses training logs from each fold to extract validation metrics
2. Compares performance across all folds
3. Identifies and reports the best performing fold
"""

import argparse
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_log_for_metrics(log_file: Path, metric_name: str = "cmc_r10") -> Optional[Dict]:
    """
    Parse training log file to extract validation metrics.
    
    Looks for patterns like:
    - "val/cmc_r10: 0.1234"
    - "Validation cmc_r10: 0.1234"
    - "Best val/cmc_r10: 0.1234"
    
    Returns:
        Dict with best metric value and epoch, or None if not found
    """
    if not log_file.exists():
        return None
    
    best_value = None
    best_epoch = None
    all_values = []
    
    # Patterns to match validation metrics
    patterns = [
        rf"val/{re.escape(metric_name)}:\s*([\d.]+)",
        rf"Validation\s+{re.escape(metric_name)}:\s*([\d.]+)",
        rf"Best\s+val/{re.escape(metric_name)}:\s*([\d.]+)",
        rf"val_{re.escape(metric_name)}:\s*([\d.]+)",
        rf"Epoch\s+(\d+).*val/{re.escape(metric_name)}:\s*([\d.]+)",
    ]
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Try each pattern
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        if len(match.groups()) == 2:
                            # Pattern with epoch and metric
                            epoch = int(match.group(1))
                            value = float(match.group(2))
                        else:
                            # Pattern with just metric
                            value = float(match.group(1))
                            # Try to extract epoch from line
                            epoch_match = re.search(r"epoch[:\s]+(\d+)", line, re.IGNORECASE)
                            epoch = int(epoch_match.group(1)) if epoch_match else None
                        
                        all_values.append((epoch, value))
                        
                        if best_value is None or value > best_value:
                            best_value = value
                            best_epoch = epoch
                
                # Also look for "Best" summary lines
                if "best" in line.lower() and metric_name.lower() in line.lower():
                    best_match = re.search(rf"best.*?{re.escape(metric_name)}.*?([\d.]+)", line, re.IGNORECASE)
                    if best_match:
                        value = float(best_match.group(1))
                        if best_value is None or value > best_value:
                            best_value = value
                            # Try to find epoch
                            epoch_match = re.search(r"epoch[:\s]+(\d+)", line, re.IGNORECASE)
                            best_epoch = int(epoch_match.group(1)) if epoch_match else None
    
    except Exception as e:
        print(f"  ⚠ Warning: Error parsing {log_file}: {e}")
        return None
    
    if best_value is None:
        return None
    
    return {
        "best_value": best_value,
        "best_epoch": best_epoch,
        "all_values": all_values,
        "n_measurements": len(all_values),
    }


def find_fold_metrics(out_root: Path, metric_name: str = "cmc_r10") -> List[Dict]:
    """
    Find and parse metrics for all folds.
    
    Returns:
        List of dicts with fold performance info
    """
    results = []
    
    # Look for fold directories
    fold_dirs = sorted(out_root.glob("fold_*"))
    
    for fold_dir in fold_dirs:
        if not fold_dir.is_dir():
            continue
        
        fold_name = fold_dir.name
        
        # Look for log file
        log_file = fold_dir / "train.log"
        
        # Also check subdirectories
        if not log_file.exists():
            for subdir in fold_dir.iterdir():
                if subdir.is_dir():
                    potential_log = subdir / "train.log"
                    if potential_log.exists():
                        log_file = potential_log
                        break
        
        print(f"  Processing {fold_name}...")
        
        if not log_file.exists():
            print(f"    ⚠ Log file not found: {log_file}")
            results.append({
                "fold": fold_name,
                "log_file": None,
                "metric_name": metric_name,
                "best_value": None,
                "best_epoch": None,
                "n_measurements": 0,
                "found": False,
            })
            continue
        
        # Parse log
        metrics = parse_log_for_metrics(log_file, metric_name)
        
        if metrics:
            results.append({
                "fold": fold_name,
                "log_file": str(log_file),
                "metric_name": metric_name,
                "best_value": metrics["best_value"],
                "best_epoch": metrics["best_epoch"],
                "n_measurements": metrics["n_measurements"],
                "found": True,
            })
            print(f"    ✓ Found best {metric_name}: {metrics['best_value']:.6f} (epoch {metrics['best_epoch']})")
        else:
            print(f"    ⚠ No {metric_name} metrics found in log")
            results.append({
                "fold": fold_name,
                "log_file": str(log_file),
                "metric_name": metric_name,
                "best_value": None,
                "best_epoch": None,
                "n_measurements": 0,
                "found": False,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze performance metrics from 10-fold CV fine-tuning runs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out_root", type=str, required=True,
                       help="Root output directory from run_finetune_10fold.py")
    parser.add_argument("--metric", type=str, default="cmc_r10",
                       help="Metric name to analyze (default: cmc_r10)")
    parser.add_argument("--output_name", type=str, default="fold_performance",
                       help="Output filename (without extension, default: fold_performance)")
    
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    
    if not out_root.exists():
        raise FileNotFoundError(f"Output root not found: {out_root}")
    
    print("=" * 80)
    print("Analyzing Fold Performance")
    print("=" * 80)
    print(f"Output root: {out_root}")
    print(f"Metric: {args.metric}")
    print("=" * 80)
    print()
    
    # Find and parse metrics
    print(f"[1] Parsing logs for {args.metric}...")
    results = find_fold_metrics(out_root, args.metric)
    print()
    
    if len(results) == 0:
        print("  WARNING: No folds found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Filter to only folds with metrics found
    df_found = df[df["found"] == True].copy()
    
    if len(df_found) == 0:
        print("  WARNING: No metrics found in any fold!")
        print("\nAll folds:")
        print(df.to_string(index=False))
        return
    
    # Sort by best_value (descending)
    df_found = df_found.sort_values("best_value", ascending=False)
    
    # Save CSV
    csv_path = out_root / f"{args.output_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[2] Saved performance analysis: {csv_path}")
    
    # Save JSON
    json_path = out_root / f"{args.output_name}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[3] Saved performance analysis: {json_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Fold Performance Summary")
    print("=" * 80)
    print(f"Metric: {args.metric}")
    print(f"Total folds: {len(results)}")
    print(f"Folds with metrics: {len(df_found)}")
    print(f"Folds without metrics: {len(results) - len(df_found)}")
    print()
    
    if len(df_found) > 0:
        print("Performance Ranking (best to worst):")
        print("-" * 80)
        for idx, (_, row) in enumerate(df_found.iterrows(), start=1):
            marker = "🏆" if idx == 1 else f"{idx}."
            print(f"{marker} {row['fold']:12s} | {args.metric}: {row['best_value']:.6f} | "
                  f"Epoch: {row['best_epoch']} | Measurements: {row['n_measurements']}")
        print("-" * 80)
        print()
        
        # Best fold
        best_fold = df_found.iloc[0]
        print("=" * 80)
        print("🏆 BEST FOLD")
        print("=" * 80)
        print(f"Fold: {best_fold['fold']}")
        print(f"Best {args.metric}: {best_fold['best_value']:.6f}")
        print(f"Best Epoch: {best_fold['best_epoch']}")
        print(f"Log file: {best_fold['log_file']}")
        print()
        
        # Statistics
        print("Statistics:")
        print(f"  Mean {args.metric}: {df_found['best_value'].mean():.6f}")
        print(f"  Std {args.metric}:  {df_found['best_value'].std():.6f}")
        print(f"  Min {args.metric}:  {df_found['best_value'].min():.6f}")
        print(f"  Max {args.metric}:  {df_found['best_value'].max():.6f}")
        print("=" * 80)
    
    # Show folds without metrics
    df_missing = df[df["found"] == False]
    if len(df_missing) > 0:
        print("\n⚠ Folds without metrics:")
        for _, row in df_missing.iterrows():
            print(f"  - {row['fold']}: {row['log_file'] or 'No log file found'}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

