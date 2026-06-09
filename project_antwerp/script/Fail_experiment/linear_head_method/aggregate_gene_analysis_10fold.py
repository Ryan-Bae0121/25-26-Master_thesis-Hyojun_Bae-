#!/usr/bin/env python3
"""
aggregate_gene_analysis_10fold.py
=================================
10-fold gene 분석 결과 통합

각 fold의 per-gene correlation을 평균내서 
최종 gene-level 성능 평가

Usage:
    python aggregate_gene_analysis_10fold.py \
        --results_dir /path/to/loki_predex_with_preds \
        --output_dir ./gene_analysis_10fold
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set_style("whitegrid")


def load_fold_gene_results(fold_dir):
    """Load gene correlation results from one fold"""
    csv_file = fold_dir / 'gene_correlations.csv'
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return None


def aggregate_gene_results(results_dir, hvg_file):
    """10개 fold의 gene correlation 평균"""
    results_dir = Path(results_dir)
    
    # Load all folds
    all_folds = []
    for fold in range(1, 11):
        fold_dir = results_dir / f"gene_analysis_fold_{fold:02d}"
        df = load_fold_gene_results(fold_dir)
        
        if df is not None:
            df['fold'] = fold
            all_folds.append(df)
            print(f"✓ Loaded fold {fold:02d}: {len(df)} genes")
        else:
            print(f"⚠️  Fold {fold:02d}: Not found")
    
    if not all_folds:
        print("❌ No fold results found!")
        return None
    
    # Concatenate
    combined_df = pd.concat(all_folds, ignore_index=True)
    
    # Aggregate by gene
    gene_summary = combined_df.groupby('gene').agg({
        'pearson': ['mean', 'std', 'median', 'min', 'max'],
        'spearman': ['mean', 'std'],
        'mean_expression': ['mean', 'std'],
        'p_value_pearson': lambda x: (x < 0.05).sum(),  # Count significant
    }).reset_index()
    
    # Flatten column names
    gene_summary.columns = ['gene', 
                           'pearson_mean', 'pearson_std', 'pearson_median', 
                           'pearson_min', 'pearson_max',
                           'spearman_mean', 'spearman_std',
                           'expression_mean', 'expression_std',
                           'n_significant_folds']
    
    # Sort by mean correlation
    gene_summary = gene_summary.sort_values('pearson_mean', ascending=False)
    
    return gene_summary, combined_df


def plot_10fold_gene_analysis(gene_summary, output_dir):
    """10-fold gene 분석 시각화"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Mean correlation distribution
    axes[0, 0].hist(gene_summary['pearson_mean'], bins=50, 
                    color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(gene_summary['pearson_mean'].mean(), 
                      color='red', linestyle='--', linewidth=2,
                      label=f"Mean: {gene_summary['pearson_mean'].mean():.3f}")
    axes[0, 0].axvline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 0].set_xlabel('Mean Pearson Correlation (10-fold)', fontsize=12)
    axes[0, 0].set_ylabel('Number of Genes', fontsize=12)
    axes[0, 0].set_title('Gene-wise Correlation Distribution\n(Averaged across 10 folds)', 
                        fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Expression vs Correlation
    axes[0, 1].scatter(gene_summary['expression_mean'], 
                      gene_summary['pearson_mean'],
                      alpha=0.4, s=30, c=gene_summary['pearson_std'],
                      cmap='RdYlGn_r')
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Std across folds', fontsize=10)
    axes[0, 1].set_xlabel('Mean Expression Level', fontsize=12)
    axes[0, 1].set_ylabel('Mean Pearson Correlation', fontsize=12)
    axes[0, 1].set_title('Expression vs Prediction Performance', 
                        fontsize=13, fontweight='bold')
    axes[0, 1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 1].grid(alpha=0.3)
    
    r, p = pearsonr(gene_summary['expression_mean'], gene_summary['pearson_mean'])
    axes[0, 1].text(0.05, 0.95, f'r = {r:.3f}, p = {p:.2e}',
                   transform=axes[0, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='top')
    
    # 3. Consistency across folds (std)
    axes[0, 2].scatter(gene_summary['pearson_mean'], 
                      gene_summary['pearson_std'],
                      alpha=0.4, s=30, color='coral')
    axes[0, 2].set_xlabel('Mean Pearson Correlation', fontsize=12)
    axes[0, 2].set_ylabel('Std across folds', fontsize=12)
    axes[0, 2].set_title('Prediction Consistency\n(Lower std = more consistent)', 
                        fontsize=13, fontweight='bold')
    axes[0, 2].axvline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Top genes
    top_20 = gene_summary.head(20)
    y_pos = np.arange(20)
    axes[1, 0].barh(y_pos, top_20['pearson_mean'].values[::-1],
                   xerr=top_20['pearson_std'].values[::-1],
                   color='steelblue', alpha=0.7, capsize=3)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(top_20['gene'].values[::-1], fontsize=8)
    axes[1, 0].set_xlabel('Mean Pearson (± Std)', fontsize=12)
    axes[1, 0].set_title('Top 20 Best Predicted Genes\n(10-fold average)', 
                        fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 5. Bottom genes
    bottom_20 = gene_summary.tail(20)
    y_pos = np.arange(20)
    axes[1, 1].barh(y_pos, bottom_20['pearson_mean'].values,
                   xerr=bottom_20['pearson_std'].values,
                   color='coral', alpha=0.7, capsize=3)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(bottom_20['gene'].values, fontsize=8)
    axes[1, 1].set_xlabel('Mean Pearson (± Std)', fontsize=12)
    axes[1, 1].set_title('Top 20 Worst Predicted Genes\n(10-fold average)', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].axvline(0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    # 6. Significant genes
    sig_counts = gene_summary['n_significant_folds'].value_counts().sort_index()
    axes[1, 2].bar(sig_counts.index, sig_counts.values, 
                  color='mediumseagreen', alpha=0.7)
    axes[1, 2].set_xlabel('Number of significant folds (p < 0.05)', fontsize=12)
    axes[1, 2].set_ylabel('Number of Genes', fontsize=12)
    axes[1, 2].set_title('Consistency of Statistical Significance', 
                        fontsize=13, fontweight='bold')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_analysis_10fold.png', dpi=300, bbox_inches='tight')
    print(f"✓ 10-fold gene analysis plot saved")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[1] Loading gene results from all folds...")
    gene_summary, combined_df = aggregate_gene_results(args.results_dir, args.hvg_file)
    
    if gene_summary is None:
        return
    
    # Save full results
    gene_summary.to_csv(output_dir / 'gene_summary_10fold.csv', index=False)
    combined_df.to_csv(output_dir / 'gene_all_folds.csv', index=False)
    
    print(f"\n✓ Saved: {output_dir / 'gene_summary_10fold.csv'}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("10-Fold Gene-wise Analysis Summary")
    print("="*70)
    print(f"Total genes: {len(gene_summary)}")
    print(f"Mean Pearson:   {gene_summary['pearson_mean'].mean():.4f} ± {gene_summary['pearson_mean'].std():.4f}")
    print(f"Median Pearson: {gene_summary['pearson_mean'].median():.4f}")
    print(f"Range: [{gene_summary['pearson_mean'].min():.4f}, {gene_summary['pearson_mean'].max():.4f}]")
    print(f"Positive: {(gene_summary['pearson_mean'] > 0).sum()} ({(gene_summary['pearson_mean'] > 0).sum()/len(gene_summary)*100:.1f}%)")
    print(f"Consistent (significant in ≥8 folds): {(gene_summary['n_significant_folds'] >= 8).sum()}")
    print("="*70)
    
    # Top/Bottom genes
    print("\nTop 15 Best Predicted Genes (10-fold average):")
    top_15 = gene_summary.head(15)
    print(top_15[['gene', 'pearson_mean', 'pearson_std', 'expression_mean']].to_string(index=False))
    
    print("\nTop 15 Worst Predicted Genes (10-fold average):")
    bottom_15 = gene_summary.tail(15)
    print(bottom_15[['gene', 'pearson_mean', 'pearson_std', 'expression_mean']].to_string(index=False))
    
    # Save lists
    top_15.to_csv(output_dir / 'top_15_genes_10fold.csv', index=False)
    bottom_15.to_csv(output_dir / 'bottom_15_genes_10fold.csv', index=False)
    
    # Visualization
    print("\n[2] Creating 10-fold visualization...")
    plot_10fold_gene_analysis(gene_summary, output_dir)
    
    # Summary JSON
    summary = {
        'n_genes': len(gene_summary),
        'n_folds': 10,
        'mean_pearson': float(gene_summary['pearson_mean'].mean()),
        'median_pearson': float(gene_summary['pearson_mean'].median()),
        'std_pearson': float(gene_summary['pearson_mean'].std()),
        'min_pearson': float(gene_summary['pearson_mean'].min()),
        'max_pearson': float(gene_summary['pearson_mean'].max()),
        'n_positive': int((gene_summary['pearson_mean'] > 0).sum()),
        'n_consistent_significant': int((gene_summary['n_significant_folds'] >= 8).sum()),
        'top_15_genes': top_15['gene'].tolist(),
        'bottom_15_genes': bottom_15['gene'].tolist(),
    }
    
    with open(output_dir / 'summary_10fold.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Directory containing gene_analysis_fold_XX")
    p.add_argument("--hvg_file", default="/project_antwerp/hbae/HVG_genelist.txt")
    p.add_argument("--output_dir", default="./gene_analysis_10fold")
    
    args = p.parse_args()
    main(args)