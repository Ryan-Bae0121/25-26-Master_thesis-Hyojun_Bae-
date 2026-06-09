#!/usr/bin/env python3
"""
compare_aggregation_methods.py
==============================
다양한 aggregation 방법 비교

Methods:
1. Spot-level (concatenate all)
2. Sample-level mean
3. Sample-level median
4. Weighted average
5. Fisher's Z-transformation
6. Harmonic mean
7. Min/Max

Usage:
    python compare_aggregation_methods.py \
        --results_dir /path/to/loki_predex_10fold \
        --output_file aggregation_comparison.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")


def aggregate_spot_level(fold_results):
    """
    방법 1: Spot-level (모든 spot을 동등하게)
    전체 spot에 대한 가중 평균
    """
    total_spots = sum(r['val_spots'] for r in fold_results)
    
    spot_weighted = sum(r['spot_pearson_mean'] * r['val_spots'] 
                       for r in fold_results) / total_spots
    gene_weighted = sum(r['gene_pearson_mean'] * r['val_spots'] 
                       for r in fold_results) / total_spots
    
    return {
        'method': 'Spot-level (Weighted by sample size)',
        'spot_pearson': spot_weighted,
        'gene_pearson': gene_weighted,
        'note': 'Larger folds have more influence'
    }


def aggregate_sample_mean(fold_results):
    """
    방법 2: Sample-level Mean (각 fold를 동등하게)
    """
    spot_corrs = [r['spot_pearson_mean'] for r in fold_results]
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    return {
        'method': 'Sample-level Mean',
        'spot_pearson': np.mean(spot_corrs),
        'spot_std': np.std(spot_corrs),
        'gene_pearson': np.mean(gene_corrs),
        'gene_std': np.std(gene_corrs),
        'note': 'All folds weighted equally'
    }


def aggregate_sample_median(fold_results):
    """
    방법 3: Sample-level Median (극단값에 robust)
    """
    spot_corrs = [r['spot_pearson_mean'] for r in fold_results]
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    return {
        'method': 'Sample-level Median',
        'spot_pearson': np.median(spot_corrs),
        'gene_pearson': np.median(gene_corrs),
        'note': 'Robust to outliers'
    }


def aggregate_fisher_z(fold_results):
    """
    방법 4: Fisher's Z-transformation
    통계적으로 더 정확한 correlation 평균
    """
    spot_corrs = [r['spot_pearson_mean'] for r in fold_results]
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    # Fisher's Z transformation
    spot_z = [np.arctanh(max(-0.9999, min(0.9999, r))) for r in spot_corrs]
    gene_z = [np.arctanh(max(-0.9999, min(0.9999, r))) for r in gene_corrs]
    
    # Mean of Z scores
    mean_spot_z = np.mean(spot_z)
    mean_gene_z = np.mean(gene_z)
    
    # Transform back
    spot_corr = np.tanh(mean_spot_z)
    gene_corr = np.tanh(mean_gene_z)
    
    return {
        'method': "Fisher's Z-transformation",
        'spot_pearson': spot_corr,
        'gene_pearson': gene_corr,
        'note': 'Statistically more accurate for averaging correlations'
    }


def aggregate_harmonic_mean(fold_results):
    """
    방법 5: Harmonic Mean
    낮은 값에 더 민감 (worst-case 성능 중시)
    """
    spot_corrs = [r['spot_pearson_mean'] for r in fold_results]
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    # Harmonic mean (only for positive values)
    spot_pos = [c for c in spot_corrs if c > 0]
    gene_pos = [c for c in gene_corrs if c > 0]
    
    spot_harmonic = stats.hmean(spot_pos) if spot_pos else 0
    gene_harmonic = stats.hmean(gene_pos) if gene_pos else 0
    
    return {
        'method': 'Harmonic Mean',
        'spot_pearson': spot_harmonic,
        'gene_pearson': gene_harmonic,
        'note': 'More sensitive to low values (worst-case performance)'
    }


def aggregate_minmax(fold_results):
    """
    방법 6: Min/Max
    성능 범위 확인
    """
    spot_corrs = [r['spot_pearson_mean'] for r in fold_results]
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    return {
        'method': 'Min/Max Range',
        'spot_min': np.min(spot_corrs),
        'spot_max': np.max(spot_corrs),
        'spot_range': np.max(spot_corrs) - np.min(spot_corrs),
        'gene_min': np.min(gene_corrs),
        'gene_max': np.max(gene_corrs),
        'gene_range': np.max(gene_corrs) - np.min(gene_corrs),
        'note': 'Performance range across folds'
    }


def aggregate_trimmed_mean(fold_results, trim_percent=10):
    """
    방법 7: Trimmed Mean
    상하위 X% 제거 후 평균
    """
    spot_corrs = [r['spot_pearson_mean'] for r in fold_results]
    gene_corrs = [r['gene_pearson_mean'] for r in fold_results]
    
    spot_trimmed = stats.trim_mean(spot_corrs, trim_percent/100)
    gene_trimmed = stats.trim_mean(gene_corrs, trim_percent/100)
    
    return {
        'method': f'Trimmed Mean ({trim_percent}%)',
        'spot_pearson': spot_trimmed,
        'gene_pearson': gene_trimmed,
        'note': f'Removes top and bottom {trim_percent}% before averaging'
    }


def visualize_methods_comparison(results, output_fig):
    """모든 방법 비교 시각화"""
    methods = []
    spot_values = []
    gene_values = []
    
    for r in results:
        if 'spot_pearson' in r:
            methods.append(r['method'])
            spot_values.append(r['spot_pearson'])
            gene_values.append(r.get('gene_pearson', 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Spot-wise
    x = np.arange(len(methods))
    axes[0].barh(x, spot_values, color='steelblue', alpha=0.7)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(methods, fontsize=10)
    axes[0].set_xlabel('Spot-wise Pearson Correlation', fontsize=12)
    axes[0].set_title('Comparison of Aggregation Methods\n(Spot-wise)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add values
    for i, v in enumerate(spot_values):
        axes[0].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)
    
    # Gene-wise
    axes[1].barh(x, gene_values, color='coral', alpha=0.7)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(methods, fontsize=10)
    axes[1].set_xlabel('Gene-wise Pearson Correlation', fontsize=12)
    axes[1].set_title('Comparison of Aggregation Methods\n(Gene-wise)', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add values
    for i, v in enumerate(gene_values):
        axes[1].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison figure saved: {output_fig}")


def create_summary_table(results, output_csv):
    """결과를 표로 정리"""
    data = []
    for r in results:
        row = {
            'Method': r['method'],
            'Spot Pearson': r.get('spot_pearson', r.get('spot_min', 'N/A')),
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
    fold_results = []
    for fold in range(1, 11):
        result_file = results_dir / f"fold_{fold:02d}" / "loki_predex_results.json"
        if result_file.exists():
            with open(result_file) as f:
                fold_results.append(json.load(f))
    
    print(f"Loaded {len(fold_results)} fold results")
    
    # Apply all aggregation methods
    all_results = []
    
    print("\n" + "="*80)
    print("Computing different aggregation methods...")
    print("="*80)
    
    # 1. Spot-level
    result = aggregate_spot_level(fold_results)
    all_results.append(result)
    print(f"\n1. {result['method']}")
    print(f"   Spot: {result['spot_pearson']:.4f}, Gene: {result['gene_pearson']:.4f}")
    
    # 2. Sample-level mean
    result = aggregate_sample_mean(fold_results)
    all_results.append(result)
    print(f"\n2. {result['method']}")
    print(f"   Spot: {result['spot_pearson']:.4f} ± {result['spot_std']:.4f}")
    print(f"   Gene: {result['gene_pearson']:.4f} ± {result['gene_std']:.4f}")
    
    # 3. Sample-level median
    result = aggregate_sample_median(fold_results)
    all_results.append(result)
    print(f"\n3. {result['method']}")
    print(f"   Spot: {result['spot_pearson']:.4f}, Gene: {result['gene_pearson']:.4f}")
    
    # 4. Fisher's Z
    result = aggregate_fisher_z(fold_results)
    all_results.append(result)
    print(f"\n4. {result['method']}")
    print(f"   Spot: {result['spot_pearson']:.4f}, Gene: {result['gene_pearson']:.4f}")
    
    # 5. Harmonic mean
    result = aggregate_harmonic_mean(fold_results)
    all_results.append(result)
    print(f"\n5. {result['method']}")
    print(f"   Spot: {result['spot_pearson']:.4f}, Gene: {result['gene_pearson']:.4f}")
    
    # 6. Trimmed mean
    result = aggregate_trimmed_mean(fold_results, trim_percent=10)
    all_results.append(result)
    print(f"\n6. {result['method']}")
    print(f"   Spot: {result['spot_pearson']:.4f}, Gene: {result['gene_pearson']:.4f}")
    
    # 7. Min/Max
    result = aggregate_minmax(fold_results)
    all_results.append(result)
    print(f"\n7. {result['method']}")
    print(f"   Spot range: [{result['spot_min']:.4f}, {result['spot_max']:.4f}]")
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
    fig_file = output_file.parent / 'aggregation_methods_comparison.png'
    visualize_methods_comparison(all_results, fig_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Directory with 10-fold results")
    p.add_argument("--output_file", required=True,
                   help="Output JSON file")
    
    args = p.parse_args()
    main(args)