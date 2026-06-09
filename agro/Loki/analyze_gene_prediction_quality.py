#!/usr/bin/env python3
"""
Gene Prediction Quality Analysis Script

This script evaluates Omiclip's performance by comparing predicted gene expression
data with ground truth data, calculating metrics (MSE, sMAPE), performing 
permutation tests with FDR correction, and generating comprehensive reports.

Author: AI Assistant
Date: 2024
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup basic logging configuration"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_and_orient_data(csv_path: str, logger) -> Tuple[pd.DataFrame, bool]:
    """
    Load CSV data and automatically detect orientation (genes as rows or columns)
    
    Args:
        csv_path: Path to CSV file
        logger: Logger instance
        
    Returns:
        Tuple of (oriented_dataframe, was_transposed)
    """
    logger.info(f"Loading data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, index_col=0)
    
    # Clean gene names (remove whitespace)
    df.index = df.index.str.strip()
    df.columns = df.columns.str.strip()
    
    # Detect orientation: assume genes are the axis with more unique identifiers
    # that look like gene names (contain letters and numbers)
    n_rows = len(df.index)
    n_cols = len(df.columns)
    
    # Simple heuristic: if rows > cols, assume genes are rows
    # Otherwise, assume genes are columns
    if n_rows > n_cols:
        logger.info(f"Detected orientation: genes as rows ({n_rows} genes, {n_cols} patches)")
        return df, False
    else:
        logger.info(f"Detected orientation: genes as columns ({n_cols} genes, {n_rows} patches)")
        return df.T, True

def intersect_and_align(pred_df: pd.DataFrame, truth_df: pd.DataFrame, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find common genes and patches, align data
    
    Args:
        pred_df: Prediction dataframe (genes x patches)
        truth_df: Ground truth dataframe (genes x patches)
        logger: Logger instance
        
    Returns:
        Tuple of aligned prediction and truth dataframes
    """
    logger.info("Finding common genes and patches...")
    
    # Find common genes and patches
    common_genes = set(pred_df.index) & set(truth_df.index)
    common_patches = set(pred_df.columns) & set(truth_df.columns)
    
    logger.info(f"Common genes: {len(common_genes)}")
    logger.info(f"Common patches: {len(common_patches)}")
    
    if len(common_genes) < 10:
        logger.warning(f"Very few common genes ({len(common_genes)}). Results may be unreliable.")
    
    if len(common_patches) < 10:
        logger.warning(f"Very few common patches ({len(common_patches)}). Results may be unreliable.")
    
    # Align data
    pred_aligned = pred_df.loc[list(common_genes), list(common_patches)]
    truth_aligned = truth_df.loc[list(common_genes), list(common_patches)]
    
    # Ensure same order
    pred_aligned = pred_aligned.sort_index()
    truth_aligned = truth_aligned.sort_index()
    
    logger.info(f"Final aligned data: {pred_aligned.shape[0]} genes x {pred_aligned.shape[1]} patches")
    
    return pred_aligned, truth_aligned

def calculate_metrics(pred_data: np.ndarray, truth_data: np.ndarray, eps: float = 1e-8) -> Tuple[float, float]:
    """
    Calculate MSE and sMAPE for a single gene
    
    Args:
        pred_data: Predicted values (1D array)
        truth_data: Ground truth values (1D array)
        eps: Small constant to avoid division by zero
        
    Returns:
        Tuple of (MSE, sMAPE)
    """
    # Remove NaN values
    mask = ~(np.isnan(pred_data) | np.isnan(truth_data))
    if np.sum(mask) < 3:
        return np.nan, np.nan
    
    pred_clean = pred_data[mask]
    truth_clean = truth_data[mask]
    
    # MSE
    mse = np.mean((pred_clean - truth_clean) ** 2)
    
    # sMAPE
    numerator = 2 * np.abs(pred_clean - truth_clean)
    denominator = np.abs(pred_clean) + np.abs(truth_clean) + eps
    smape = np.mean(numerator / denominator)
    
    return mse, smape

def permutation_test(pred_data: np.ndarray, truth_data: np.ndarray, n_perm: int, 
                    rng: np.random.Generator) -> float:
    """
    Perform permutation test for significance
    
    Args:
        pred_data: Predicted values (1D array)
        truth_data: Ground truth values (1D array)
        n_perm: Number of permutations
        rng: Random number generator
        
    Returns:
        p-value (one-sided)
    """
    # Remove NaN values
    mask = ~(np.isnan(pred_data) | np.isnan(truth_data))
    if np.sum(mask) < 3:
        return 1.0
    
    pred_clean = pred_data[mask]
    truth_clean = truth_data[mask]
    
    # Observed MSE
    mse_obs = np.mean((pred_clean - truth_clean) ** 2)
    
    # Permutation test
    mse_perm = np.zeros(n_perm)
    for i in range(n_perm):
        pred_shuffled = rng.permutation(pred_clean)
        mse_perm[i] = np.mean((pred_shuffled - truth_clean) ** 2)
    
    # One-sided p-value: probability of getting MSE as good or better by chance
    p_value = (1 + np.sum(mse_perm <= mse_obs)) / (n_perm + 1)
    
    return p_value

def analyze_gene_prediction_quality(pred_csv: str, truth_csv: str, out_dir: str, 
                                  n_perm: int = 1000, log1p: bool = False, 
                                  seed: int = 42) -> None:
    """
    Main analysis function
    
    Args:
        pred_csv: Path to prediction CSV file
        truth_csv: Path to ground truth CSV file
        out_dir: Output directory
        n_perm: Number of permutations for significance testing
        log1p: Whether to apply log1p transformation
        seed: Random seed for reproducibility
    """
    logger = setup_logging()
    
    # Set random seed
    rng = np.random.default_rng(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    
    # Load and orient data
    pred_df, pred_transposed = load_and_orient_data(pred_csv, logger)
    truth_df, truth_transposed = load_and_orient_data(truth_csv, logger)
    
    # Intersect and align
    pred_aligned, truth_aligned = intersect_and_align(pred_df, truth_df, logger)
    
    # Apply transformation if requested
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
        
        # Calculate MSE and sMAPE
        mse, smape = calculate_metrics(pred_values, truth_values)
        
        if np.isnan(mse) or np.isnan(smape):
            logger.warning(f"Skipping gene {gene}: insufficient valid data")
            continue
        
        # Permutation test
        p_value = permutation_test(pred_values, truth_values, n_perm, rng)
        
        # Count valid patches
        mask = ~(np.isnan(pred_values) | np.isnan(truth_values))
        n_patches = np.sum(mask)
        
        results.append({
            'gene': gene,
            'n_patches': n_patches,
            'mse': mse,
            'smape': smape,
            'p_value': p_value
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    logger.info(f"Analyzed {len(results_df)} genes")
    
    # FDR correction
    logger.info("Performing FDR correction...")
    q_values = multipletests(results_df['p_value'], method='fdr_bh')[1]
    results_df['q_value'] = q_values
    
    # Save metrics
    metrics_path = os.path.join(out_dir, 'metrics_per_gene.csv')
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Identify well-predicted genes
    well_predicted = results_df[
        (results_df['q_value'] < 0.05) & 
        (results_df['smape'] < 0.30)
    ].copy()
    
    well_predicted_path = os.path.join(out_dir, 'well_predicted_genes.csv')
    well_predicted.to_csv(well_predicted_path, index=False)
    logger.info(f"Well-predicted genes saved to: {well_predicted_path}")
    logger.info(f"Found {len(well_predicted)} well-predicted genes")
    
    # Generate plots
    logger.info("Generating plots...")
    generate_plots(results_df, well_predicted, pred_aligned, truth_aligned, out_dir)
    
    # Generate summary report
    generate_summary_report(results_df, well_predicted, out_dir, logger)

def generate_plots(results_df: pd.DataFrame, well_predicted: pd.DataFrame,
                  pred_aligned: pd.DataFrame, truth_aligned: pd.DataFrame, 
                  out_dir: str) -> None:
    """Generate all analysis plots"""
    
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
    plt.savefig(os.path.join(out_dir, 'hist_mse.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. sMAPE histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['smape'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('sMAPE')
    plt.ylabel('Frequency')
    plt.title('Distribution of sMAPE Values')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hist_smape.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. P-values Q-Q plot
    plt.figure(figsize=(10, 6))
    stats.probplot(results_df['p_value'], dist="uniform", plot=plt)
    plt.title('Q-Q Plot of P-values (Uniform Distribution)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'qq_pvalues.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Volcano-like plot
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['smape'], -np.log10(results_df['q_value']), 
                alpha=0.6, s=20, c='blue', label='All genes')
    
    if len(well_predicted) > 0:
        plt.scatter(well_predicted['smape'], -np.log10(well_predicted['q_value']), 
                   alpha=0.8, s=30, c='red', label='Well-predicted genes')
        
        # Label top 10 well-predicted genes
        top_genes = well_predicted.nsmallest(10, 'q_value')
        for _, row in top_genes.iterrows():
            plt.annotate(row['gene'], 
                        (row['smape'], -np.log10(row['q_value'])),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    plt.xlabel('sMAPE')
    plt.ylabel('-log10(q-value)')
    plt.title('Volcano-like Plot: sMAPE vs Significance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'volcano_like.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Scatter plots for top 5 genes
    top_5_genes = results_df.nsmallest(5, 'q_value')
    
    for _, row in top_5_genes.iterrows():
        gene = row['gene']
        pred_values = pred_aligned.loc[gene].values
        truth_values = truth_aligned.loc[gene].values
        
        # Remove NaN values
        mask = ~(np.isnan(pred_values) | np.isnan(truth_values))
        pred_clean = pred_values[mask]
        truth_clean = truth_values[mask]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(truth_clean, pred_clean, alpha=0.6, s=20)
        
        # Add y=x reference line
        min_val = min(np.min(truth_clean), np.min(pred_clean))
        max_val = max(np.max(truth_clean), np.max(pred_clean))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='y=x')
        
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted')
        plt.title(f'True vs Predicted: {gene}\nMSE={row["mse"]:.4f}, sMAPE={row["smape"]:.4f}, q-value={row["q_value"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'scatter_top5_true_vs_pred_{gene}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def generate_summary_report(results_df: pd.DataFrame, well_predicted: pd.DataFrame,
                          out_dir: str, logger) -> None:
    """Generate summary report"""
    
    report_path = os.path.join(out_dir, 'analysis_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("Gene Prediction Quality Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total genes analyzed: {len(results_df)}\n")
        f.write(f"Well-predicted genes: {len(well_predicted)}\n")
        f.write(f"Success rate: {len(well_predicted)/len(results_df)*100:.2f}%\n\n")
        
        f.write("MSE Statistics:\n")
        f.write(f"  Mean: {results_df['mse'].mean():.6f}\n")
        f.write(f"  Median: {results_df['mse'].median():.6f}\n")
        f.write(f"  Std: {results_df['mse'].std():.6f}\n")
        f.write(f"  Min: {results_df['mse'].min():.6f}\n")
        f.write(f"  Max: {results_df['mse'].max():.6f}\n\n")
        
        f.write("sMAPE Statistics:\n")
        f.write(f"  Mean: {results_df['smape'].mean():.6f}\n")
        f.write(f"  Median: {results_df['smape'].median():.6f}\n")
        f.write(f"  Std: {results_df['smape'].std():.6f}\n")
        f.write(f"  Min: {results_df['smape'].min():.6f}\n")
        f.write(f"  Max: {results_df['smape'].max():.6f}\n\n")
        
        f.write("Top 10 Well-predicted Genes:\n")
        if len(well_predicted) > 0:
            top_10 = well_predicted.nsmallest(10, 'q_value')
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"  {i:2d}. {row['gene']:15s} - MSE: {row['mse']:.6f}, sMAPE: {row['smape']:.6f}, q-value: {row['q_value']:.6f}\n")
        else:
            f.write("  No well-predicted genes found.\n")
    
    logger.info(f"Summary report saved to: {report_path}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Analyze gene prediction quality by comparing predicted vs ground truth data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_gene_prediction_quality.py \\
    --pred_csv gene_expression_maps_TCGA-QK-A6IH-01Z-00-DX1.csv \\
    --truth_csv ground_truth_ST_matrix_TCGA-QK-A6IH-01Z-00-DX1.csv \\
    --out_dir results_TCGA_QK_A6IH \\
    --n_perm 1000 \\
    --log1p false \\
    --seed 42
        """
    )
    
    parser.add_argument('--pred_csv', required=True,
                       help='Path to prediction CSV file')
    parser.add_argument('--truth_csv', required=True,
                       help='Path to ground truth CSV file')
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
        print(f"Error: Prediction CSV file not found: {args.pred_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.truth_csv):
        print(f"Error: Ground truth CSV file not found: {args.truth_csv}")
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
        print(f"\nAnalysis completed successfully! Results saved to: {args.out_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
