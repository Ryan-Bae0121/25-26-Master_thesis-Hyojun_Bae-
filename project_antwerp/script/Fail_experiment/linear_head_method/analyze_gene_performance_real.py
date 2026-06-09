#!/usr/bin/env python3
"""
analyze_gene_performance_real.py
================================
실제 저장된 예측값으로 per-gene 분석

Usage:
    python analyze_gene_performance_real.py \
        --predictions_file fold01/predictions.npy \
        --ground_truth_file fold01/ground_truth.npy \
        --hvg_file HVG_genelist.txt \
        --output_dir gene_analysis
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

sns.set_style("whitegrid")


def compute_gene_correlations(predictions, ground_truth, hvg_genes):
    """
    각 유전자별 correlation 계산
    
    Args:
        predictions: (n_spots, n_genes)
        ground_truth: (n_spots, n_genes)
        hvg_genes: list of gene names
    """
    n_genes = predictions.shape[1]
    
    gene_stats = []
    for i, gene in enumerate(hvg_genes):
        pred_g = predictions[:, i]
        gt_g = ground_truth[:, i]
        
        # Skip if no variance
        if gt_g.std() < 1e-8:
            continue
        
        # Pearson correlation
        r_pearson, p_pearson = pearsonr(pred_g, gt_g)
        r_spearman, p_spearman = spearmanr(pred_g, gt_g)
        
        gene_stats.append({
            'gene': gene,
            'pearson': r_pearson,
            'spearman': r_spearman,
            'p_value_pearson': p_pearson,
            'p_value_spearman': p_spearman,
            'mean_expression': gt_g.mean(),
            'std_expression': gt_g.std(),
            'mean_prediction': pred_g.mean(),
            'std_prediction': pred_g.std(),
        })
    
    df = pd.DataFrame(gene_stats)
    return df


def plot_gene_analysis(gene_df, output_dir):
    """유전자 분석 시각화"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Correlation distribution
    axes[0, 0].hist(gene_df['pearson'], bins=50, color='steelblue', 
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(gene_df['pearson'].mean(), color='red', 
                      linestyle='--', linewidth=2,
                      label=f"Mean: {gene_df['pearson'].mean():.3f}")
    axes[0, 0].axvline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 0].set_xlabel('Pearson Correlation', fontsize=12)
    axes[0, 0].set_ylabel('Number of Genes', fontsize=12)
    axes[0, 0].set_title('Gene-wise Correlation Distribution', 
                        fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Expression vs Correlation
    axes[0, 1].scatter(gene_df['mean_expression'], gene_df['pearson'],
                      alpha=0.4, s=20, color='coral')
    axes[0, 1].set_xlabel('Mean Expression Level', fontsize=12)
    axes[0, 1].set_ylabel('Pearson Correlation', fontsize=12)
    axes[0, 1].set_title('Expression vs Prediction Performance', 
                        fontsize=13, fontweight='bold')
    axes[0, 1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 1].grid(alpha=0.3)
    
    # Add correlation
    r, p = pearsonr(gene_df['mean_expression'], gene_df['pearson'])
    axes[0, 1].text(0.05, 0.95, f'r = {r:.3f}, p = {p:.2e}',
                   transform=axes[0, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='top')
    
    # 3. Variance vs Correlation
    axes[0, 2].scatter(gene_df['std_expression'], gene_df['pearson'],
                      alpha=0.4, s=20, color='mediumseagreen')
    axes[0, 2].set_xlabel('Expression Variance (Std)', fontsize=12)
    axes[0, 2].set_ylabel('Pearson Correlation', fontsize=12)
    axes[0, 2].set_title('Variance vs Prediction Performance', 
                        fontsize=13, fontweight='bold')
    axes[0, 2].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 2].grid(alpha=0.3)
    
    r, p = pearsonr(gene_df['std_expression'], gene_df['pearson'])
    axes[0, 2].text(0.05, 0.95, f'r = {r:.3f}, p = {p:.2e}',
                   transform=axes[0, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='top')
    
    # 4. Top/Bottom genes
    top_20 = gene_df.nlargest(20, 'pearson')
    bottom_20 = gene_df.nsmallest(20, 'pearson')
    
    axes[1, 0].barh(range(20), top_20['pearson'].values[::-1], 
                   color='steelblue', alpha=0.7)
    axes[1, 0].set_yticks(range(20))
    axes[1, 0].set_yticklabels(top_20['gene'].values[::-1], fontsize=8)
    axes[1, 0].set_xlabel('Pearson Correlation', fontsize=12)
    axes[1, 0].set_title('Top 20 Best Predicted Genes', 
                        fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 5. Worst genes
    axes[1, 1].barh(range(20), bottom_20['pearson'].values, 
                   color='coral', alpha=0.7)
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels(bottom_20['gene'].values, fontsize=8)
    axes[1, 1].set_xlabel('Pearson Correlation', fontsize=12)
    axes[1, 1].set_title('Top 20 Worst Predicted Genes', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    # 6. Cumulative distribution
    sorted_corr = np.sort(gene_df['pearson'].values)
    cumulative = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
    axes[1, 2].plot(sorted_corr, cumulative, linewidth=2, color='steelblue')
    axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.5,
                      label='Zero correlation')
    axes[1, 2].axvline(gene_df['pearson'].median(), color='orange', 
                      linestyle='--', alpha=0.5,
                      label=f"Median: {gene_df['pearson'].median():.3f}")
    axes[1, 2].set_xlabel('Pearson Correlation', fontsize=12)
    axes[1, 2].set_ylabel('Cumulative Fraction', fontsize=12)
    axes[1, 2].set_title('Cumulative Distribution', 
                        fontsize=13, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_analysis_full.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gene analysis plot saved")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("[1] Loading predictions and ground truth...")
    predictions = np.load(args.predictions_file)
    ground_truth = np.load(args.ground_truth_file)
    
    print(f"  Predictions: {predictions.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    
    # Load gene list
    with open(args.hvg_file) as f:
        hvg_genes = [line.strip() for line in f]
    
    print(f"  HVG genes: {len(hvg_genes)}")
    
    # Compute gene correlations
    print("\n[2] Computing per-gene correlations...")
    gene_df = compute_gene_correlations(predictions, ground_truth, hvg_genes)
    
    # Save full results
    gene_df.to_csv(output_dir / 'gene_correlations.csv', index=False)
    print(f"  ✓ Saved: {output_dir / 'gene_correlations.csv'}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Gene-wise Correlation Summary")
    print("="*70)
    print(f"Total genes analyzed: {len(gene_df)}")
    print(f"Mean Pearson:   {gene_df['pearson'].mean():.4f} ± {gene_df['pearson'].std():.4f}")
    print(f"Median Pearson: {gene_df['pearson'].median():.4f}")
    print(f"Range: [{gene_df['pearson'].min():.4f}, {gene_df['pearson'].max():.4f}]")
    print(f"Positive: {(gene_df['pearson'] > 0).sum()} ({(gene_df['pearson'] > 0).sum()/len(gene_df)*100:.1f}%)")
    print(f"Negative: {(gene_df['pearson'] <= 0).sum()} ({(gene_df['pearson'] <= 0).sum()/len(gene_df)*100:.1f}%)")
    print(f"Significant (p<0.05): {(gene_df['p_value_pearson'] < 0.05).sum()}")
    print("="*70)
    
    # Top/Bottom genes
    print("\nTop 10 Best Predicted Genes:")
    top_10 = gene_df.nlargest(10, 'pearson')
    print(top_10[['gene', 'pearson', 'mean_expression']].to_string(index=False))
    
    print("\nTop 10 Worst Predicted Genes:")
    bottom_10 = gene_df.nsmallest(10, 'pearson')
    print(bottom_10[['gene', 'pearson', 'mean_expression']].to_string(index=False))
    
    # Save lists
    top_10.to_csv(output_dir / 'top_10_genes.csv', index=False)
    bottom_10.to_csv(output_dir / 'bottom_10_genes.csv', index=False)
    
    # Visualization
    print("\n[3] Creating visualizations...")
    plot_gene_analysis(gene_df, output_dir)
    
    # Summary JSON
    summary = {
        'n_genes': len(gene_df),
        'mean_pearson': float(gene_df['pearson'].mean()),
        'median_pearson': float(gene_df['pearson'].median()),
        'std_pearson': float(gene_df['pearson'].std()),
        'min_pearson': float(gene_df['pearson'].min()),
        'max_pearson': float(gene_df['pearson'].max()),
        'n_positive': int((gene_df['pearson'] > 0).sum()),
        'n_significant': int((gene_df['p_value_pearson'] < 0.05).sum()),
        'top_10_genes': top_10['gene'].tolist(),
        'bottom_10_genes': bottom_10['gene'].tolist(),
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_file", required=True,
                   help="Path to predictions.npy")
    p.add_argument("--ground_truth_file", required=True,
                   help="Path to ground_truth.npy")
    p.add_argument("--hvg_file", required=True,
                   help="HVG gene list file")
    p.add_argument("--output_dir", default="./gene_analysis")
    
    args = p.parse_args()
    main(args)