#!/usr/bin/env python3
"""
Gene Prediction Quality Analysis Script

This script evaluates the performance of gene expression prediction models
by comparing predicted values with ground truth data.

Usage:
    python analyze_gene_prediction_quality.py \
      --pred_csv path/to/predicted.csv \
      --truth_csv path/to/ground_truth.csv \
      --out_dir results \
      --n_perm 1000 \
      --log1p false \
      --seed 42
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants for well-predicted gene criteria
Q_VALUE_THRESHOLD = 0.05
SMAPE_THRESHOLD = 0.30
MIN_PATCHES_PER_GENE = 3
RECOMMENDED_PATCHES_PER_GENE = 10


def load_and_orient_data(csv_path: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Load CSV data and automatically detect gene vs patch orientation.
    
    Returns:
        Tuple of (dataframe, gene_axis, patch_axis)
    """
    logger.info(f"Loading data from {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path, index_col=0)
    
    # Clean index and column names
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    
    # Detect orientation based on typical patterns
    # Genes usually have more unique identifiers than patches
    n_rows, n_cols = df.shape
    
    # Simple heuristic: if rows > cols, assume rows are genes
    if n_rows > n_cols:
        gene_axis = 'index'
        patch_axis = 'columns'
        logger.info(f"Detected orientation: rows=genes ({n_rows}), cols=patches ({n_cols})")
    else:
        gene_axis = 'columns'
        patch_axis = 'index'
        logger.info(f"Detected orientation: rows=patches ({n_rows}), cols=genes ({n_cols})")
    
    # Ensure genes are rows and patches are columns
    if gene_axis == 'columns':
        df = df.T
        logger.info("Transposed data to make genes rows and patches columns")
    
    return df, 'index', 'columns'


def intersect_and_align(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find common genes and patches, align dataframes.
    
    Returns:
        Tuple of aligned (pred_df, truth_df)
    """
    logger.info("Finding common genes and patches...")
    
    # Get common genes and patches
    common_genes = set(pred_df.index) & set(truth_df.index)
    common_patches = set(pred_df.columns) & set(truth_df.columns)
    
    logger.info(f"Common genes: {len(common_genes)}")
    logger.info(f"Common patches: {len(common_patches)}")
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found between prediction and truth data")
    if len(common_patches) == 0:
        raise ValueError("No common patches found between prediction and truth data")
    
    # Align dataframes
    pred_aligned = pred_df.loc[common_genes, common_patches].copy()
    truth_aligned = truth_df.loc[common_genes, common_patches].copy()
    
    # Sort to ensure consistent ordering
    pred_aligned = pred_aligned.sort_index().sort_index(axis=1)
    truth_aligned = truth_aligned.sort_index().sort_index(axis=1)
    
    logger.info(f"Aligned data shape: {pred_aligned.shape}")
    
    return pred_aligned, truth_aligned


def calculate_metrics(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float]:
    """
    Calculate MSE and sMAPE for a single gene.
    
    Args:
        pred: Predicted values (1D array)
        truth: Ground truth values (1D array)
    
    Returns:
        Tuple of (mse, smape)
    """
    # Remove NaN values
    mask = ~(np.isnan(pred) | np.isnan(truth))
    if np.sum(mask) < MIN_PATCHES_PER_GENE:
        return np.nan, np.nan
    
    pred_clean = pred[mask]
    truth_clean = truth[mask]
    
    # MSE
    mse = np.mean((pred_clean - truth_clean) ** 2)
    
    # sMAPE
    eps = 1e-8
    numerator = 2 * np.abs(pred_clean - truth_clean)
    denominator = np.abs(pred_clean) + np.abs(truth_clean) + eps
    smape = np.mean(numerator / denominator)
    
    return mse, smape


def permutation_test(pred: np.ndarray, truth: np.ndarray, n_perm: int, 
                    rng: np.random.Generator) -> float:
    """
    Perform permutation test to assess significance.
    
    Args:
        pred: Predicted values (1D array)
        truth: Ground truth values (1D array)
        n_perm: Number of permutations
        rng: Random number generator
    
    Returns:
        p-value (one-sided)
    """
    # Remove NaN values
    mask = ~(np.isnan(pred) | np.isnan(truth))
    if np.sum(mask) < MIN_PATCHES_PER_GENE:
        return np.nan
    
    pred_clean = pred[mask]
    truth_clean = truth[mask]
    
    # Observed MSE
    mse_obs = np.mean((pred_clean - truth_clean) ** 2)
    
    # Permutation test
    mse_perm = np.zeros(n_perm)
    for i in range(n_perm):
        # Shuffle predicted values
        pred_shuffled = rng.permutation(pred_clean)
        mse_perm[i] = np.mean((pred_shuffled - truth_clean) ** 2)
    
    # One-sided p-value: probability of getting MSE <= observed by chance
    p_value = (1 + np.sum(mse_perm <= mse_obs)) / (n_perm + 1)
    
    return p_value


def analyze_gene_prediction_quality(pred_csv: str, truth_csv: str, out_dir: str,
                                  n_perm: int = 1000, log1p: bool = False, 
                                  seed: int = 42) -> None:
    """
    Main analysis function.
    """
    # Set random seed
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting gene prediction quality analysis...")
    logger.info(f"Parameters: n_perm={n_perm}, log1p={log1p}, seed={seed}")
    
    # Load and orient data
    pred_df, _, _ = load_and_orient_data(pred_csv)
    truth_df, _, _ = load_and_orient_data(truth_csv)
    
    # Intersect and align
    pred_aligned, truth_aligned = intersect_and_align(pred_df, truth_df)
    
    # Optional log transformation
    if log1p:
        logger.info("Applying log1p transformation...")
        pred_aligned = np.log1p(pred_aligned)
        truth_aligned = np.log1p(truth_aligned)
    
    # Calculate metrics for each gene
    logger.info("Calculating metrics for each gene...")
    
    results = []
    for gene in pred_aligned.index:
        pred_values = pred_aligned.loc[gene].values
        truth_values = truth_aligned.loc[gene].values
        
        # Calculate metrics
        mse, smape = calculate_metrics(pred_values, truth_values)
        
        if np.isnan(mse) or np.isnan(smape):
            continue
        
        # Permutation test
        p_value = permutation_test(pred_values, truth_values, n_perm, rng)
        
        if np.isnan(p_value):
            continue
        
        # Count valid patches
        n_patches = np.sum(~(np.isnan(pred_values) | np.isnan(truth_values)))
        
        results.append({
            'gene': gene,
            'n_patches': n_patches,
            'mse': mse,
            'smape': smape,
            'p_value': p_value
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        logger.error("No valid genes found for analysis")
        return
    
    logger.info(f"Analyzed {len(results_df)} genes")
    
    # FDR correction
    logger.info("Applying FDR correction...")
    _, q_values, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['q_value'] = q_values
    
    # Save metrics per gene
    metrics_file = out_path / 'metrics_per_gene.csv'
    results_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Identify well-predicted genes
    well_predicted = results_df[
        (results_df['q_value'] < Q_VALUE_THRESHOLD) & 
        (results_df['smape'] < SMAPE_THRESHOLD)
    ].copy()
    
    well_predicted_file = out_path / 'well_predicted_genes.csv'
    well_predicted.to_csv(well_predicted_file, index=False)
    logger.info(f"Found {len(well_predicted)} well-predicted genes")
    logger.info(f"Saved well-predicted genes to {well_predicted_file}")
    
    # Generate plots
    logger.info("Generating plots...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. MSE histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['mse'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('Distribution of MSE Values')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(out_path / 'hist_mse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. sMAPE histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['smape'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('sMAPE')
    plt.ylabel('Frequency')
    plt.title('Distribution of sMAPE Values')
    plt.tight_layout()
    plt.savefig(out_path / 'hist_smape.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. P-value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(-np.log10(results_df['p_value']), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('-log10(p-value)')
    plt.ylabel('Frequency')
    plt.title('Distribution of -log10(p-values)')
    plt.tight_layout()
    plt.savefig(out_path / 'qq_pvalues.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Volcano-like plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(results_df['smape'], -np.log10(results_df['q_value']), 
                         alpha=0.6, s=20)
    
    # Highlight well-predicted genes
    if len(well_predicted) > 0:
        plt.scatter(well_predicted['smape'], -np.log10(well_predicted['q_value']), 
                   color='red', alpha=0.8, s=30, label='Well-predicted')
    
    # Add threshold lines
    plt.axhline(y=-np.log10(Q_VALUE_THRESHOLD), color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=SMAPE_THRESHOLD, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('sMAPE')
    plt.ylabel('-log10(q-value)')
    plt.title('Volcano-like Plot: sMAPE vs Significance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / 'volcano_like.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Scatter plots for top 5 genes
    top_genes = results_df.nsmallest(5, 'q_value')
    
    for i, (_, row) in enumerate(top_genes.iterrows()):
        gene = row['gene']
        
        # Get data for this gene
        pred_values = pred_aligned.loc[gene].values
        truth_values = truth_aligned.loc[gene].values
        
        # Remove NaN values
        mask = ~(np.isnan(pred_values) | np.isnan(truth_values))
        pred_clean = pred_values[mask]
        truth_clean = truth_values[mask]
        
        if len(pred_clean) < MIN_PATCHES_PER_GENE:
            continue
        
        plt.figure(figsize=(8, 8))
        plt.scatter(truth_clean, pred_clean, alpha=0.6, s=20)
        
        # Add y=x reference line
        min_val = min(np.min(truth_clean), np.min(pred_clean))
        max_val = max(np.max(truth_clean), np.max(pred_clean))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
        
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted')
        plt.title(f'Gene {gene}: True vs Predicted\n'
                 f'MSE={row["mse"]:.4f}, sMAPE={row["smape"]:.4f}, '
                 f'q-value={row["q_value"]:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / f'scatter_top5_true_vs_pred_{gene}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate summary report
    logger.info("Generating summary report...")
    
    summary_file = out_path / 'analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Gene Prediction Quality Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Input files:\n")
        f.write(f"  Prediction: {pred_csv}\n")
        f.write(f"  Ground truth: {truth_csv}\n\n")
        
        f.write(f"Analysis parameters:\n")
        f.write(f"  Permutations: {n_perm}\n")
        f.write(f"  Log transformation: {log1p}\n")
        f.write(f"  Random seed: {seed}\n\n")
        
        f.write(f"Data summary:\n")
        f.write(f"  Total genes analyzed: {len(results_df)}\n")
        f.write(f"  Well-predicted genes: {len(well_predicted)}\n")
        f.write(f"  Well-predicted rate: {len(well_predicted)/len(results_df):.1%}\n\n")
        
        f.write(f"Criteria for well-predicted genes:\n")
        f.write(f"  q-value < {Q_VALUE_THRESHOLD}\n")
        f.write(f"  sMAPE < {SMAPE_THRESHOLD}\n\n")
        
        f.write(f"Top 5 genes by significance:\n")
        for i, (_, row) in enumerate(top_genes.iterrows()):
            f.write(f"  {i+1}. {row['gene']}: q-value={row['q_value']:.4f}, "
                   f"sMAPE={row['smape']:.4f}, MSE={row['mse']:.4f}\n")
        
        f.write(f"\nOutput files:\n")
        f.write(f"  Metrics per gene: {metrics_file}\n")
        f.write(f"  Well-predicted genes: {well_predicted_file}\n")
        f.write(f"  Plots: hist_mse.png, hist_smape.png, qq_pvalues.png, "
               f"volcano_like.png, scatter_top5_*.png\n")
    
    logger.info(f"Analysis complete! Results saved to {out_dir}")
    logger.info(f"Summary report: {summary_file}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Analyze gene prediction quality by comparing predicted vs ground truth data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_gene_prediction_quality.py \\
    --pred_csv predicted.csv \\
    --truth_csv ground_truth.csv \\
    --out_dir results \\
    --n_perm 1000 \\
    --log1p false \\
    --seed 42
        """
    )
    
    parser.add_argument('--pred_csv', required=True,
                       help='Path to predicted gene expression CSV file')
    parser.add_argument('--truth_csv', required=True,
                       help='Path to ground truth gene expression CSV file')
    parser.add_argument('--out_dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--n_perm', type=int, default=1000,
                       help='Number of permutations for significance testing (default: 1000)')
    parser.add_argument('--log1p', action='store_true',
                       help='Apply log1p transformation to data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.pred_csv):
        logger.error(f"Prediction file not found: {args.pred_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.truth_csv):
        logger.error(f"Truth file not found: {args.truth_csv}")
        sys.exit(1)
    
    try:
        analyze_gene_prediction_quality(
            pred_csv=args.pred_csv,
            truth_csv=args.truth_csv,
            out_dir=args.out_dir,
            n_perm=args.n_perm,
            log1p=args.log1p,
            seed=args.seed
        )
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
