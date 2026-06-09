#!/usr/bin/env python3
"""
calculate_variance_ratio.py
===========================
HVG 예측에 대한 Variance Ratio 계산

Variance Ratio = Var(predicted) / Var(actual)
- Per-gene: 각 유전자별 variance ratio
- Per-spot: 각 spot별 variance ratio

Usage:
    python calculate_variance_ratio.py \
        --predictions_file predictions.npy \
        --ground_truth_file ground_truth.npy \
        --hvg_file HVG_genelist.txt \
        --output_dir variance_analysis
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


def calculate_variance_ratios(predictions, ground_truth):
    """
    Calculate variance ratios
    
    Args:
        predictions: (n_spots, n_genes)
        ground_truth: (n_spots, n_genes)
    
    Returns:
        gene_var_ratio: (n_genes,) variance ratio per gene
        spot_var_ratio: (n_spots,) variance ratio per spot
    """
    # Per-gene variance ratio
    pred_var_genes = predictions.var(axis=0)  # (n_genes,)
    true_var_genes = ground_truth.var(axis=0)  # (n_genes,)
    
    # Avoid division by zero
    gene_var_ratio = np.where(
        true_var_genes > 1e-8,
        pred_var_genes / true_var_genes,
        np.nan
    )
    
    # Per-spot variance ratio
    pred_var_spots = predictions.var(axis=1)  # (n_spots,)
    true_var_spots = ground_truth.var(axis=1)  # (n_spots,)
    
    spot_var_ratio = np.where(
        true_var_spots > 1e-8,
        pred_var_spots / true_var_spots,
        np.nan
    )
    
    return gene_var_ratio, spot_var_ratio


def analyze_variance_ratio(predictions, ground_truth, gene_names, output_dir):
    """Complete variance ratio analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Variance Ratio Analysis")
    print("="*70)
    
    # Calculate variance ratios
    gene_var_ratio, spot_var_ratio = calculate_variance_ratios(predictions, ground_truth)
    
    # Remove NaN values
    gene_var_ratio_clean = gene_var_ratio[~np.isnan(gene_var_ratio)]
    spot_var_ratio_clean = spot_var_ratio[~np.isnan(spot_var_ratio)]
    
    # Summary statistics
    print("\nGene-wise Variance Ratio:")
    print(f"  Mean:   {gene_var_ratio_clean.mean():.4f}")
    print(f"  Median: {np.median(gene_var_ratio_clean):.4f}")
    print(f"  Std:    {gene_var_ratio_clean.std():.4f}")
    print(f"  Range:  [{gene_var_ratio_clean.min():.4f}, {gene_var_ratio_clean.max():.4f}]")
    print(f"  Perfect (0.9-1.1): {((gene_var_ratio_clean >= 0.9) & (gene_var_ratio_clean <= 1.1)).sum()} / {len(gene_var_ratio_clean)}")
    print(f"  Under-pred (<1.0): {(gene_var_ratio_clean < 1.0).sum()} ({(gene_var_ratio_clean < 1.0).sum()/len(gene_var_ratio_clean)*100:.1f}%)")
    print(f"  Over-pred (>1.0):  {(gene_var_ratio_clean > 1.0).sum()} ({(gene_var_ratio_clean > 1.0).sum()/len(gene_var_ratio_clean)*100:.1f}%)")
    
    print("\nSpot-wise Variance Ratio:")
    print(f"  Mean:   {spot_var_ratio_clean.mean():.4f}")
    print(f"  Median: {np.median(spot_var_ratio_clean):.4f}")
    print(f"  Std:    {spot_var_ratio_clean.std():.4f}")
    print(f"  Range:  [{spot_var_ratio_clean.min():.4f}, {spot_var_ratio_clean.max():.4f}]")
    
    # Per-gene analysis
    gene_df = pd.DataFrame({
        'gene': gene_names,
        'var_ratio': gene_var_ratio,
        'pred_var': predictions.var(axis=0),
        'true_var': ground_truth.var(axis=0),
        'pred_mean': predictions.mean(axis=0),
        'true_mean': ground_truth.mean(axis=0),
    })
    
    # Remove NaN
    gene_df_clean = gene_df.dropna()
    
    # Sort by variance ratio
    gene_df_clean = gene_df_clean.sort_values('var_ratio')
    
    # Save
    gene_df_clean.to_csv(output_dir / 'gene_variance_ratio.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'gene_variance_ratio.csv'}")
    
    # Top/Bottom genes by variance ratio
    print("\nTop 10 Under-predicted (lowest var ratio):")
    bottom_10 = gene_df_clean.head(10)
    print(bottom_10[['gene', 'var_ratio', 'pred_var', 'true_var']].to_string(index=False))
    
    print("\nTop 10 Over-predicted (highest var ratio):")
    top_10 = gene_df_clean.tail(10)
    print(top_10[['gene', 'var_ratio', 'pred_var', 'true_var']].to_string(index=False))
    
    # Correlation between variance ratio and prediction performance
    # Calculate per-gene correlation first
    gene_corrs = []
    for i in range(predictions.shape[1]):
        if ground_truth[:, i].std() > 1e-8:
            r, _ = pearsonr(predictions[:, i], ground_truth[:, i])
            gene_corrs.append(r)
        else:
            gene_corrs.append(np.nan)
    
    gene_df_clean['pearson'] = [gene_corrs[i] for i in gene_df_clean.index]
    
    # Correlation: var_ratio vs pearson
    valid_mask = ~np.isnan(gene_df_clean['var_ratio']) & ~np.isnan(gene_df_clean['pearson'])
    if valid_mask.sum() > 0:
        corr_var_perf, p_val = pearsonr(
            gene_df_clean.loc[valid_mask, 'var_ratio'],
            gene_df_clean.loc[valid_mask, 'pearson']
        )
        print(f"\nCorrelation (Var Ratio vs Pearson): r={corr_var_perf:.4f}, p={p_val:.2e}")
    
    return gene_df_clean, spot_var_ratio_clean


def plot_variance_ratio_analysis(gene_df, spot_var_ratio, output_dir):
    """Visualize variance ratio analysis"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Gene-wise variance ratio distribution
    axes[0, 0].hist(gene_df['var_ratio'], bins=50, color='steelblue', 
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(1.0, color='red', linestyle='--', linewidth=2,
                      label='Perfect (1.0)')
    axes[0, 0].axvline(gene_df['var_ratio'].mean(), color='orange', 
                      linestyle='--', linewidth=2,
                      label=f"Mean: {gene_df['var_ratio'].mean():.3f}")
    axes[0, 0].set_xlabel('Variance Ratio', fontsize=12)
    axes[0, 0].set_ylabel('Number of Genes', fontsize=12)
    axes[0, 0].set_title('Gene-wise Variance Ratio Distribution', 
                        fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlim(0, min(gene_df['var_ratio'].quantile(0.99) * 1.2, 5))
    
    # 2. Predicted vs True variance
    axes[0, 1].scatter(gene_df['true_var'], gene_df['pred_var'],
                      alpha=0.4, s=30, c=gene_df['var_ratio'], 
                      cmap='RdYlGn', vmin=0.5, vmax=1.5)
    max_var = max(gene_df['true_var'].max(), gene_df['pred_var'].max())
    axes[0, 1].plot([0, max_var], [0, max_var], 'r--', linewidth=2,
                   label='Perfect (y=x)')
    axes[0, 1].set_xlabel('True Variance', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Variance', fontsize=12)
    axes[0, 1].set_title('Predicted vs True Variance (Per Gene)', 
                        fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Var Ratio', fontsize=10)
    
    # 3. Variance ratio vs Pearson correlation
    if 'pearson' in gene_df.columns:
        valid_mask = ~np.isnan(gene_df['var_ratio']) & ~np.isnan(gene_df['pearson'])
        axes[0, 2].scatter(gene_df.loc[valid_mask, 'var_ratio'], 
                          gene_df.loc[valid_mask, 'pearson'],
                          alpha=0.4, s=30, color='coral')
        axes[0, 2].axvline(1.0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 2].set_xlabel('Variance Ratio', fontsize=12)
        axes[0, 2].set_ylabel('Pearson Correlation', fontsize=12)
        axes[0, 2].set_title('Var Ratio vs Prediction Performance', 
                            fontsize=13, fontweight='bold')
        axes[0, 2].grid(alpha=0.3)
        
        if valid_mask.sum() > 0:
            r, p = pearsonr(gene_df.loc[valid_mask, 'var_ratio'],
                           gene_df.loc[valid_mask, 'pearson'])
            axes[0, 2].text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}',
                           transform=axes[0, 2].transAxes,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           verticalalignment='top')
    
    # 4. Spot-wise variance ratio distribution
    axes[1, 0].hist(spot_var_ratio, bins=50, color='mediumseagreen', 
                    alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(1.0, color='red', linestyle='--', linewidth=2,
                      label='Perfect (1.0)')
    axes[1, 0].axvline(spot_var_ratio.mean(), color='orange', 
                      linestyle='--', linewidth=2,
                      label=f"Mean: {spot_var_ratio.mean():.3f}")
    axes[1, 0].set_xlabel('Variance Ratio', fontsize=12)
    axes[1, 0].set_ylabel('Number of Spots', fontsize=12)
    axes[1, 0].set_title('Spot-wise Variance Ratio Distribution', 
                        fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Top under-predicted genes
    bottom_20 = gene_df.nsmallest(20, 'var_ratio')
    y_pos = np.arange(20)
    axes[1, 1].barh(y_pos, bottom_20['var_ratio'].values,
                   color='coral', alpha=0.7)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(bottom_20['gene'].values, fontsize=8)
    axes[1, 1].set_xlabel('Variance Ratio', fontsize=12)
    axes[1, 1].set_title('Top 20 Under-predicted Genes\n(Lowest Var Ratio)', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].axvline(1.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    # 6. Top over-predicted genes
    top_20 = gene_df.nlargest(20, 'var_ratio')
    y_pos = np.arange(20)
    axes[1, 2].barh(y_pos, top_20['var_ratio'].values[::-1],
                   color='steelblue', alpha=0.7)
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(top_20['gene'].values[::-1], fontsize=8)
    axes[1, 2].set_xlabel('Variance Ratio', fontsize=12)
    axes[1, 2].set_title('Top 20 Over-predicted Genes\n(Highest Var Ratio)', 
                        fontsize=13, fontweight='bold')
    axes[1, 2].axvline(1.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'variance_ratio_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Variance ratio plot saved: {output_dir / 'variance_ratio_analysis.png'}")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("[1] Loading predictions and ground truth...")
    predictions = np.load(args.predictions_file)
    ground_truth = np.load(args.ground_truth_file)
    
    print(f"  Predictions: {predictions.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    
    # Load gene names
    with open(args.hvg_file) as f:
        gene_names = [line.strip() for line in f]
    
    print(f"  Genes: {len(gene_names)}")
    
    # Analysis
    print("\n[2] Calculating variance ratios...")
    gene_df, spot_var_ratio = analyze_variance_ratio(
        predictions, ground_truth, gene_names, output_dir)
    
    # Visualization
    print("\n[3] Creating visualizations...")
    plot_variance_ratio_analysis(gene_df, spot_var_ratio, output_dir)
    
    # Summary JSON
    summary = {
        'gene_var_ratio': {
            'mean': float(gene_df['var_ratio'].mean()),
            'median': float(gene_df['var_ratio'].median()),
            'std': float(gene_df['var_ratio'].std()),
            'min': float(gene_df['var_ratio'].min()),
            'max': float(gene_df['var_ratio'].max()),
            'n_under_pred': int((gene_df['var_ratio'] < 1.0).sum()),
            'n_over_pred': int((gene_df['var_ratio'] > 1.0).sum()),
            'n_perfect': int(((gene_df['var_ratio'] >= 0.9) & (gene_df['var_ratio'] <= 1.1)).sum()),
        },
        'spot_var_ratio': {
            'mean': float(spot_var_ratio.mean()),
            'median': float(np.median(spot_var_ratio)),
            'std': float(spot_var_ratio.std()),
            'min': float(spot_var_ratio.min()),
            'max': float(spot_var_ratio.max()),
        },
        'top_10_under_pred': gene_df.head(10)['gene'].tolist(),
        'top_10_over_pred': gene_df.tail(10)['gene'].tolist(),
    }
    
    with open(output_dir / 'variance_ratio_summary.json', 'w') as f:
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
    p.add_argument("--output_dir", default="./variance_analysis")
    
    args = p.parse_args()
    main(args)