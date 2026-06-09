#!/usr/bin/env python3
"""
compare_gene_aggregation_methods.py
===================================
Gene-wise Pearson에 대한 다양한 aggregation 방법 비교

Spot-wise와 동일한 7가지 방법 적용:
1. Gene-level (weighted by sample size)
2. Sample-level mean
3. Sample-level median
4. Fisher's Z-transformation
5. Harmonic mean
6. Trimmed mean
7. Min/Max range

Usage:
    python compare_gene_aggregation_methods.py \
        --results_dir /path/to/loki_predex_10fold \
        --output_file gene_aggregation_comparison.json
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")


def load_fold_results(results_dir):
    """Load all fold results"""
    results_dir = Path(results_dir)
    fold_results = []
    
    for fold in range(1, 11):
        result_file = results_dir / f"fold_{fold:02d}" / "loki_predex_results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                fold_results.append({
                    'fold': fold,
                    'gene_pearson_mean': data['gene_pearson_mean'],
                    'gene_pearson_median': data['gene_pearson_median'],
                    'gene_pearson_std': data['gene_pearson_std'],
                    'val_spots': data['val_spots'],
                })
    
    return fold_results


def aggregate_gene_level(fold_results):
    """Gene-level weighted by sample size"""
    total_spots = sum(r['val_spots'] for r in fold_results)
    weighted = sum(r['gene_pearson_mean'] * r['val_spots'] for r in fold_results) / total_spots
    
    return {
        'method': 'Gene-level (Weighted by sample size)',
        'gene_pearson': weighted,
        'note': 'Larger folds have more influence'
    }


def aggregate_sample_mean(fold_results):
    """Sample-level mean"""
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    return {
        'method': 'Sample-level Mean',
        'gene_pearson': np.mean(gene_corrs),
        'gene_std': np.std(gene_corrs),
        'note': 'All folds weighted equally'
    }


def aggregate_sample_median(fold_results):
    """Sample-level median"""
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    return {
        'method': 'Sample-level Median',
        'gene_pearson': np.median(gene_corrs),
        'note': 'Robust to outliers'
    }


def aggregate_fisher_z(fold_results):
    """Fisher's Z-transformation"""
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    # Fisher's Z (handle values close to 0)
    gene_z = [np.arctanh(max(-0.9999, min(0.9999, r))) if abs(r) > 0.001 else 0 
              for r in gene_corrs]
    
    mean_gene_z = np.mean(gene_z)
    gene_corr = np.tanh(mean_gene_z)
    
    return {
        'method': "Fisher's Z-transformation",
        'gene_pearson': gene_corr,
        'note': 'Statistically more accurate for averaging correlations'
    }


def aggregate_harmonic_mean(fold_results):
    """Harmonic mean (only positive values)"""
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    # Only positive
    gene_pos = [c for c in gene_corrs if c > 0.001]
    
    if len(gene_pos) > 0:
        gene_harmonic = stats.hmean(gene_pos)
    else:
        gene_harmonic = 0
    
    return {
        'method': 'Harmonic Mean',
        'gene_pearson': gene_harmonic,
        'note': f'Based on {len(gene_pos)}/{len(gene_corrs)} positive values'
    }


def aggregate_trimmed_mean(fold_results, trim_percent=10):
    """Trimmed mean"""
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    gene_trimmed = stats.trim_mean(gene_corrs, trim_percent/100)
    
    return {
        'method': f'Trimmed Mean ({trim_percent}%)',
        'gene_pearson': gene_trimmed,
        'note': f'Removes top and bottom {trim_percent}% before averaging'
    }


def aggregate_minmax(fold_results):
    """Min/Max range"""
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    return {
        'method': 'Min/Max Range',
        'gene_min': np.min(gene_corrs),
        'gene_max': np.max(gene_corrs),
        'gene_range': np.max(gene_corrs) - np.min(gene_corrs),
        'note': 'Performance range across folds'
    }


def plot_gene_aggregation_comparison(results, output_file):
    """Visualize gene aggregation methods"""
    methods = []
    gene_values = []
    
    for r in results:
        if 'gene_pearson' in r:
            methods.append(r['method'])
            gene_values.append(r['gene_pearson'])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = np.arange(len(methods))
    bars = ax.barh(x, gene_values, color='coral', alpha=0.7)
    
    # Color code by value
    for i, (bar, val) in enumerate(zip(bars, gene_values)):
        if val > 0.01:
            bar.set_color('lightgreen')
        elif val > 0:
            bar.set_color('coral')
        else:
            bar.set_color('lightcoral')
    
    ax.set_yticks(x)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel('Gene-wise Pearson Correlation', fontsize=12)
    ax.set_title('Comparison of Aggregation Methods\n(Gene-wise Performance)', 
                fontsize=14, fontweight='bold')
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    
    # Add values
    for i, v in enumerate(gene_values):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gene aggregation plot saved: {output_file}")


def create_summary_table(results, output_csv):
    """Create summary table"""
    data = []
    for r in results:
        row = {
            'Method': r['method'],
            'Gene Pearson': r.get('gene_pearson', r.get('gene_min', 'N/A')),
            'Note': r.get('note', ''),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✓ Summary table saved: {output_csv}")
    
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)


def main(args):
    results_dir = Path(args.results_dir)
    
    # Load fold results
    print("\n[1] Loading fold results...")
    fold_results = load_fold_results(results_dir)
    print(f"Loaded {len(fold_results)} fold results")
    
    # Apply all aggregation methods
    all_results = []
    
    print("\n" + "="*80)
    print("Computing different aggregation methods for Gene-wise Pearson...")
    print("="*80)
    
    # 1. Gene-level
    result = aggregate_gene_level(fold_results)
    all_results.append(result)
    print(f"\n1. {result['method']}")
    print(f"   Gene: {result['gene_pearson']:.4f}")
    
    # 2. Sample-level mean
    result = aggregate_sample_mean(fold_results)
    all_results.append(result)
    print(f"\n2. {result['method']}")
    print(f"   Gene: {result['gene_pearson']:.4f} ± {result['gene_std']:.4f}")
    
    # 3. Sample-level median
    result = aggregate_sample_median(fold_results)
    all_results.append(result)
    print(f"\n3. {result['method']}")
    print(f"   Gene: {result['gene_pearson']:.4f}")
    
    # 4. Fisher's Z
    result = aggregate_fisher_z(fold_results)
    all_results.append(result)
    print(f"\n4. {result['method']}")
    print(f"   Gene: {result['gene_pearson']:.4f}")
    
    # 5. Harmonic mean
    result = aggregate_harmonic_mean(fold_results)
    all_results.append(result)
    print(f"\n5. {result['method']}")
    print(f"   Gene: {result['gene_pearson']:.4f}")
    
    # 6. Trimmed mean
    result = aggregate_trimmed_mean(fold_results, trim_percent=10)
    all_results.append(result)
    print(f"\n6. {result['method']}")
    print(f"   Gene: {result['gene_pearson']:.4f}")
    
    # 7. Min/Max
    result = aggregate_minmax(fold_results)
    all_results.append(result)
    print(f"\n7. {result['method']}")
    print(f"   Gene range: [{result['gene_min']:.4f}, {result['gene_max']:.4f}]")
    
    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'aggregation_methods': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved: {output_file}")
    
    # Create summary table
    csv_file = output_file.with_suffix('.csv')
    create_summary_table(all_results, csv_file)
    
    # Visualize comparison
    fig_file = output_file.parent / 'gene_aggregation_methods_comparison.png'
    plot_gene_aggregation_comparison(all_results, fig_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Directory with 10-fold results")
    p.add_argument("--output_file", required=True,
                   help="Output JSON file")
    
    args = p.parse_args()
    main(args)