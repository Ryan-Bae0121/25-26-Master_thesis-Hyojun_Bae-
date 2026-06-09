#!/usr/bin/env python3
"""
aggregate_loki_predex_results.py
=================================
Aggregate Loki PredEx 10-fold CV results

Usage:
    python aggregate_loki_predex_results.py \
        --results_dir /path/to/loki_predex_10fold \
        --output_file summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def main(args):
    results_dir = Path(args.results_dir)
    
    # Collect all fold results
    all_results = []
    
    for fold in range(1, 11):
        fold_dir = results_dir / f"fold_{fold:02d}"
        result_file = fold_dir / "loki_predex_results.json"
        
        if not result_file.exists():
            print(f"⚠️  Fold {fold:02d}: Results not found")
            continue
        
        with open(result_file) as f:
            data = json.load(f)
        
        all_results.append({
            'fold': fold,
            'train_spots': data['train_spots'],
            'val_spots': data['val_spots'],
            'spot_pearson_mean': data['spot_pearson_mean'],
            'spot_pearson_median': data['spot_pearson_median'],
            'spot_pearson_std': data['spot_pearson_std'],
            'gene_pearson_mean': data['gene_pearson_mean'],
            'gene_pearson_median': data['gene_pearson_median'],
            'gene_pearson_std': data['gene_pearson_std'],
        })
        
        print(f"✓ Fold {fold:02d}: Spot Pearson = {data['spot_pearson_mean']:.4f}")
    
    if not all_results:
        print("❌ No results found!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Summary statistics
    summary = {
        'method': 'Loki PredEx (10-Fold CV)',
        'n_folds': len(df),
        'total_train_spots': int(df['train_spots'].sum()),
        'total_val_spots': int(df['val_spots'].sum()),
        'spot_pearson': {
            'mean': float(df['spot_pearson_mean'].mean()),
            'std': float(df['spot_pearson_mean'].std()),
            'median': float(df['spot_pearson_mean'].median()),
            'min': float(df['spot_pearson_mean'].min()),
            'max': float(df['spot_pearson_mean'].max()),
        },
        'gene_pearson': {
            'mean': float(df['gene_pearson_mean'].mean()),
            'std': float(df['gene_pearson_mean'].std()),
            'median': float(df['gene_pearson_mean'].median()),
            'min': float(df['gene_pearson_mean'].min()),
            'max': float(df['gene_pearson_mean'].max()),
        },
        'folds': all_results
    }
    
    # Save summary
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save CSV
    csv_file = output_file.with_suffix('.csv')
    df.to_csv(csv_file, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("Loki PredEx 10-Fold Cross-Validation Results")
    print("="*70)
    print(f"Folds completed: {len(df)}/10")
    print(f"Total val spots: {summary['total_val_spots']:,}")
    print()
    print("Spot-wise Pearson Correlation:")
    print(f"  Mean   : {summary['spot_pearson']['mean']:.4f} ± {summary['spot_pearson']['std']:.4f}")
    print(f"  Median : {summary['spot_pearson']['median']:.4f}")
    print(f"  Range  : [{summary['spot_pearson']['min']:.4f}, {summary['spot_pearson']['max']:.4f}]")
    print()
    print("Gene-wise Pearson Correlation:")
    print(f"  Mean   : {summary['gene_pearson']['mean']:.4f} ± {summary['gene_pearson']['std']:.4f}")
    print(f"  Median : {summary['gene_pearson']['median']:.4f}")
    print(f"  Range  : [{summary['gene_pearson']['min']:.4f}, {summary['gene_pearson']['max']:.4f}]")
    print("="*70)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spot-wise correlation
    axes[0].bar(df['fold'], df['spot_pearson_mean'], color='steelblue', alpha=0.8)
    axes[0].axhline(summary['spot_pearson']['mean'], color='red', ls='--', 
                    label=f"Mean: {summary['spot_pearson']['mean']:.4f}")
    axes[0].set_xlabel("Fold", fontsize=12)
    axes[0].set_ylabel("Spot-wise Pearson Correlation", fontsize=12)
    axes[0].set_title("Loki PredEx: Spot-wise Correlation (10-Fold CV)", 
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0.6, 0.75)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Gene-wise correlation
    axes[1].bar(df['fold'], df['gene_pearson_mean'], color='coral', alpha=0.8)
    axes[1].axhline(summary['gene_pearson']['mean'], color='red', ls='--',
                    label=f"Mean: {summary['gene_pearson']['mean']:.4f}")
    axes[1].set_xlabel("Fold", fontsize=12)
    axes[1].set_ylabel("Gene-wise Pearson Correlation", fontsize=12)
    axes[1].set_title("Loki PredEx: Gene-wise Correlation (10-Fold CV)", 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig_file = output_file.parent / "10fold_correlation_plot.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_file}")
    
    print(f"✓ Summary saved: {output_file}")
    print(f"✓ CSV saved: {csv_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, 
                   help="Directory containing fold_01, fold_02, ..., fold_10")
    p.add_argument("--output_file", required=True,
                   help="Output JSON file path")
    
    args = p.parse_args()
    main(args)