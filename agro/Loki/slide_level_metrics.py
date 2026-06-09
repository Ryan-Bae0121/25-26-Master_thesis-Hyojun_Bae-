#!/usr/bin/env python3
"""
Slide-level Metrics Calculator for Loki Predictions vs TCGA Ground Truth

This script aggregates tile-level Loki predictions to slide-level and evaluates 
against TCGA bulk ground truth using MSE, sMAPE, and Pearson correlation.
It is robust to gene/ID orientation and sparsity.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
from scipy import stats
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def detect_orientation(df):
    """
    Detect if dataframe is gene×patch or patch×gene orientation.
    Returns: 'genes_as_rows' or 'genes_as_cols'
    """
    n_rows, n_cols = df.shape
    
    # Heuristic: pick the axis with larger cardinality as genes
    if n_rows > n_cols:
        # More rows than cols - likely genes as rows
        orientation = 'genes_as_rows'
    else:
        # More cols than rows - likely genes as cols
        orientation = 'genes_as_cols'
    
    # Additional check: look for patch-like identifiers
    if orientation == 'genes_as_rows':
        # Check if column names look like patch identifiers
        col_sample = df.columns[:min(10, len(df.columns))]
        patch_like_cols = sum(1 for col in col_sample if 
                             re.match(r'^\d+_\d+$', str(col)) or 
                             str(col).startswith('patch_') or
                             str(col).isdigit())
        
        if patch_like_cols > len(col_sample) * 0.7:
            orientation = 'genes_as_rows'  # Confirmed
        else:
            orientation = 'genes_as_cols'  # Switch
    
    else:  # genes_as_cols
        # Check if row names look like patch identifiers
        row_sample = df.index[:min(10, len(df.index))]
        patch_like_rows = sum(1 for row in row_sample if 
                             re.match(r'^\d+_\d+$', str(row)) or 
                             str(row).startswith('patch_') or
                             str(row).isdigit())
        
        if patch_like_rows > len(row_sample) * 0.7:
            orientation = 'genes_as_cols'  # Confirmed
        else:
            orientation = 'genes_as_rows'  # Switch
    
    return orientation

def normalize_gene_ids(gene_ids):
    """
    Normalize gene IDs for matching.
    - Strip version suffixes (e.g., ENSG00000123456.1 -> ENSG00000123456)
    - Strip rna_ prefix for TCGA data
    - Uppercase symbols
    - Trim spaces
    """
    normalized = []
    for gene_id in gene_ids:
        gene_str = str(gene_id).strip()
        
        # Strip rna_ prefix for TCGA data
        if gene_str.startswith('rna_'):
            gene_str = gene_str[4:]  # Remove 'rna_' prefix
        
        # Strip version suffix for Ensembl IDs
        if '.' in gene_str and gene_str.split('.')[0].startswith('ENS'):
            gene_str = gene_str.split('.')[0]
        
        # Uppercase for gene symbols
        gene_str = gene_str.upper()
        normalized.append(gene_str)
    
    return normalized

def extract_base_sample_id(slide_id):
    """Extract base sample ID from slide ID (first 3 hyphen-separated fields)"""
    parts = slide_id.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    return slide_id

def find_matching_sample(tcga_df, slide_id, sample_col=None):
    """
    Find matching sample in TCGA dataframe.
    Returns: (sample_id, sample_data)
    """
    base_id = extract_base_sample_id(slide_id)
    print(f"Looking for sample matching base ID: {base_id}")
    
    # Determine sample column
    if sample_col is None:
        # Try to find sample identifier column
        possible_cols = ['wsi_file_name', 'sample_id', 'sample', 'Sample', 'SAMPLE_ID', 'Sample_ID']
        sample_col = None
        for col in possible_cols:
            if col in tcga_df.columns:
                sample_col = col
                break
        
        if sample_col is None:
            # Use second column as sample identifier (first is usually index)
            if len(tcga_df.columns) > 1:
                sample_col = tcga_df.columns[1]
            else:
                sample_col = tcga_df.columns[0]
            print(f"No explicit sample column found, using column: {sample_col}")
    
    print(f"Using sample column: {sample_col}")
    
    # Get all sample identifiers
    sample_ids = tcga_df[sample_col].astype(str).str.upper()
    
    # Try exact match first
    exact_matches = sample_ids[sample_ids == base_id.upper()]
    if len(exact_matches) > 0:
        sample_id = exact_matches.index[0]
        print(f"Found exact match: {tcga_df.loc[sample_id, sample_col]}")
        return tcga_df.loc[sample_id, sample_col], tcga_df.loc[sample_id]
    
    # Try substring match
    substring_matches = sample_ids[sample_ids.str.contains(base_id.upper(), na=False)]
    if len(substring_matches) > 0:
        sample_id = substring_matches.index[0]
        print(f"Found substring match: {tcga_df.loc[sample_id, sample_col]}")
        return tcga_df.loc[sample_id, sample_col], tcga_df.loc[sample_id]
    
    # Try fuzzy matching
    print("No exact or substring match found, trying fuzzy matching...")
    similarities = []
    for idx, sample_id in sample_ids.items():
        similarity = SequenceMatcher(None, base_id.upper(), sample_id).ratio()
        similarities.append((similarity, idx, sample_id))
    
    similarities.sort(reverse=True)
    
    if similarities[0][0] > 0.3:  # Minimum similarity threshold
        best_match_idx = similarities[0][1]
        best_match_id = similarities[0][2]
        print(f"Found fuzzy match: {best_match_id} (similarity: {similarities[0][0]:.3f})")
        return best_match_id, tcga_df.loc[best_match_idx]
    
    # No good match found
    print("WARNING: No good match found!")
    print("Top 5 candidates:")
    for i, (sim, idx, sample_id) in enumerate(similarities[:5]):
        print(f"  {i+1}. {sample_id} (similarity: {sim:.3f})")
    
    # Return the best match anyway
    best_match_idx = similarities[0][1]
    best_match_id = similarities[0][2]
    return best_match_id, tcga_df.loc[best_match_idx]

def determine_aggregation_method(pred_data, threshold=0.6):
    """
    Determine aggregation method based on zero fraction.
    If zero fraction > threshold, use 'sum', else 'mean'.
    """
    zero_fraction = (pred_data == 0).sum().sum() / (pred_data.shape[0] * pred_data.shape[1])
    
    if zero_fraction > threshold:
        method = 'sum'
        print(f"High zero fraction ({zero_fraction:.3f} > {threshold}), using SUM aggregation")
    else:
        method = 'mean'
        print(f"Low zero fraction ({zero_fraction:.3f} <= {threshold}), using MEAN aggregation")
    
    return method, zero_fraction

def aggregate_to_slide_level(pred_data, method):
    """Aggregate tile-level predictions to slide level"""
    if method == 'sum':
        return pred_data.sum(axis=1)  # Sum across patches for each gene
    elif method == 'mean':
        return pred_data.mean(axis=1)  # Mean across patches for each gene
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def calculate_metrics(pred_slide, gt_slide, log1p=False):
    """
    Calculate MSE, sMAPE, and Pearson correlation.
    Returns: dict with metrics
    """
    # Remove any NaN values
    valid_mask = ~(np.isnan(pred_slide) | np.isnan(gt_slide))
    pred_valid = pred_slide[valid_mask]
    gt_valid = gt_slide[valid_mask]
    
    if len(pred_valid) == 0:
        return {'MSE': np.nan, 'sMAPE': np.nan, 'Pearson_r': np.nan, 'Pearson_p': np.nan}
    
    # Apply log1p transformation if requested
    if log1p:
        pred_valid = np.log1p(pred_valid)
        gt_valid = np.log1p(gt_valid)
        suffix = '_log1p'
    else:
        suffix = '_raw'
    
    # MSE
    mse = np.mean((pred_valid - gt_valid) ** 2)
    
    # sMAPE
    numerator = 2 * np.abs(pred_valid - gt_valid)
    denominator = np.abs(pred_valid) + np.abs(gt_valid) + 1e-8
    smape = np.mean(numerator / denominator)
    
    # Pearson correlation
    if len(pred_valid) > 1 and (np.std(pred_valid) > 1e-8 or np.std(gt_valid) > 1e-8):
        pearson_r, pearson_p = stats.pearsonr(pred_valid, gt_valid)
    else:
        print(f"WARNING: Cannot compute Pearson correlation (constant vectors)")
        pearson_r, pearson_p = np.nan, np.nan
    
    return {
        f'MSE{suffix}': mse,
        f'sMAPE{suffix}': smape,
        f'Pearson_r{suffix}': pearson_r,
        f'Pearson_p{suffix}': pearson_p
    }

def create_scatter_plot(pred_slide, gt_slide, output_path, log1p=False):
    """Create scatter plot of predictions vs ground truth"""
    plt.figure(figsize=(8, 8))
    
    # Remove NaN values
    valid_mask = ~(np.isnan(pred_slide) | np.isnan(gt_slide))
    pred_valid = pred_slide[valid_mask]
    gt_valid = gt_slide[valid_mask]
    
    # Check if we have any valid data
    if len(pred_valid) == 0:
        plt.text(0.5, 0.5, 'No valid data points for plotting', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Predicted vs Ground Truth Expression (No Data)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    if log1p:
        pred_valid = np.log1p(pred_valid)
        gt_valid = np.log1p(gt_valid)
        xlabel = 'Log1p(Predicted Expression)'
        ylabel = 'Log1p(Ground Truth Expression)'
        title_suffix = ' (Log1p Transformed)'
    else:
        xlabel = 'Predicted Expression'
        ylabel = 'Ground Truth Expression'
        title_suffix = ' (Raw)'
    
    plt.scatter(pred_valid, gt_valid, alpha=0.6, s=20)
    
    # Add diagonal line
    min_val = min(pred_valid.min(), gt_valid.min())
    max_val = max(pred_valid.max(), gt_valid.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect correlation')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Predicted vs Ground Truth Expression{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add correlation info
    if len(pred_valid) > 1 and (np.std(pred_valid) > 1e-8 or np.std(gt_valid) > 1e-8):
        corr, p_val = stats.pearsonr(pred_valid, gt_valid)
        plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {p_val:.3e}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Calculate slide-level metrics for Loki predictions vs TCGA ground truth")
    parser.add_argument("--pred_csv", required=True, help="Path to gene expression predictions CSV")
    parser.add_argument("--tcga_csv", required=True, help="Path to TCGA bulk expression CSV")
    parser.add_argument("--slide_id", required=True, help="Slide ID to match (e.g., TCGA-QK-A6IH-01Z-00-DX1)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--agg", default="auto", choices=["sum", "mean", "auto"], help="Aggregation method")
    parser.add_argument("--log1p", default="false", choices=["true", "false"], help="Apply log1p transformation")
    parser.add_argument("--gene_id_type", default="auto", help="Gene ID type (auto-detect)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=== Slide-level Metrics Calculator ===")
    print(f"Prediction CSV: {args.pred_csv}")
    print(f"TCGA CSV: {args.tcga_csv}")
    print(f"Slide ID: {args.slide_id}")
    print(f"Output directory: {args.out_dir}")
    print()
    
    # 1. Load prediction data
    print("Loading prediction data...")
    pred_df = pd.read_csv(args.pred_csv, index_col=0)
    print(f"Prediction data shape: {pred_df.shape}")
    
    # 2. Detect orientation
    print("Detecting data orientation...")
    orientation = detect_orientation(pred_df)
    print(f"Detected orientation: {orientation}")
    
    # 3. Coerce to genes as rows, patches as columns
    if orientation == 'genes_as_cols':
        print("Transposing data to genes as rows...")
        pred_df = pred_df.T
    
    print(f"Final prediction data shape: {pred_df.shape} (genes × patches)")
    
    # 4. Determine aggregation method
    if args.agg == "auto":
        agg_method, zero_fraction = determine_aggregation_method(pred_df)
    else:
        agg_method = args.agg
        zero_fraction = (pred_df == 0).sum().sum() / (pred_df.shape[0] * pred_df.shape[1])
        print(f"Using specified aggregation method: {agg_method}")
    
    # 5. Aggregate to slide level
    print(f"Aggregating to slide level using {agg_method}...")
    pred_slide = aggregate_to_slide_level(pred_df, agg_method)
    
    # 6. Load TCGA data and find matching sample
    print("Loading TCGA data...")
    tcga_df = pd.read_csv(args.tcga_csv, index_col=0)
    print(f"TCGA data shape: {tcga_df.shape}")
    
    matched_sample_id, matched_sample_data = find_matching_sample(tcga_df, args.slide_id)
    print(f"Matched sample: {matched_sample_id}")
    
    # 7. Normalize gene IDs for alignment
    print("Normalizing gene IDs...")
    pred_genes = normalize_gene_ids(pred_slide.index)
    tcga_genes = normalize_gene_ids(matched_sample_data.index)
    
    # Create mapping
    pred_gene_map = dict(zip(pred_genes, pred_slide.index))
    tcga_gene_map = dict(zip(tcga_genes, matched_sample_data.index))
    
    # Find common genes
    common_genes = set(pred_genes) & set(tcga_genes)
    print(f"Found {len(common_genes)} common genes")
    
    if len(common_genes) < 100:
        print("⚠️  LOW_GENE_OVERLAP: Less than 100 genes aligned!")
    
    # 8. Align data
    aligned_pred = []
    aligned_gt = []
    aligned_gene_names = []
    
    for gene in common_genes:
        pred_idx = pred_gene_map[gene]
        tcga_idx = tcga_gene_map[gene]
        
        aligned_pred.append(pred_slide[pred_idx])
        aligned_gt.append(matched_sample_data[tcga_idx])
        aligned_gene_names.append(gene)
    
    aligned_pred = np.array(aligned_pred)
    aligned_gt = np.array(aligned_gt)
    
    print(f"Aligned data: {len(aligned_pred)} genes")
    
    # 9. Calculate metrics
    print("Calculating metrics...")
    metrics = {}
    
    # Raw metrics
    raw_metrics = calculate_metrics(aligned_pred, aligned_gt, log1p=False)
    metrics.update(raw_metrics)
    
    # Log1p metrics if requested
    if args.log1p.lower() == "true":
        log1p_metrics = calculate_metrics(aligned_pred, aligned_gt, log1p=True)
        metrics.update(log1p_metrics)
    
    # Add metadata
    metrics.update({
        'n_genes_used': len(aligned_pred),
        'agg_method': agg_method,
        'zero_fraction': zero_fraction,
        'matched_sample_id': matched_sample_id,
        'slide_id': args.slide_id
    })
    
    # 10. Create outputs
    print("Creating outputs...")
    
    # Save aligned gene table
    aligned_df = pd.DataFrame({
        'gene': aligned_gene_names,
        'pred_slide': aligned_pred,
        'gt_slide': aligned_gt
    })
    aligned_df.to_csv(os.path.join(args.out_dir, 'aligned_gene_table.csv'), index=False)
    
    # Save metrics
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create scatter plots
    create_scatter_plot(aligned_pred, aligned_gt, 
                       os.path.join(args.out_dir, 'scatter_pred_vs_gt.png'), 
                       log1p=False)
    
    if args.log1p.lower() == "true":
        create_scatter_plot(aligned_pred, aligned_gt, 
                           os.path.join(args.out_dir, 'scatter_pred_vs_gt_log1p.png'), 
                           log1p=True)
    
    # Create diagnostics
    diagnostics = f"""
Slide-level Metrics Diagnostics
==============================

Input Files:
- Prediction CSV: {args.pred_csv}
- TCGA CSV: {args.tcga_csv}
- Slide ID: {args.slide_id}

Data Processing:
- Original prediction shape: {pred_df.shape}
- Detected orientation: {orientation}
- Final shape (genes × patches): {pred_df.shape if orientation == 'genes_as_rows' else pred_df.T.shape}
- Aggregation method: {agg_method} (zero fraction: {zero_fraction:.3f})

Sample Matching:
- Target base ID: {extract_base_sample_id(args.slide_id)}
- Matched sample: {matched_sample_id}
- TCGA data shape: {tcga_df.shape}

Gene Alignment:
- Prediction genes: {len(pred_genes)}
- TCGA genes: {len(tcga_genes)}
- Common genes: {len(common_genes)}
- Final aligned genes: {len(aligned_pred)}

Metrics Summary:
- MSE (raw): {metrics.get('MSE_raw', 'N/A'):.6f}
- sMAPE (raw): {metrics.get('sMAPE_raw', 'N/A'):.6f}
- Pearson r (raw): {metrics.get('Pearson_r_raw', 'N/A'):.6f}
"""
    
    if args.log1p.lower() == "true":
        diagnostics += f"""
- MSE (log1p): {metrics.get('MSE_log1p', 'N/A'):.6f}
- sMAPE (log1p): {metrics.get('sMAPE_log1p', 'N/A'):.6f}
- Pearson r (log1p): {metrics.get('Pearson_r_log1p', 'N/A'):.6f}
"""
    
    with open(os.path.join(args.out_dir, 'diagnostics.txt'), 'w') as f:
        f.write(diagnostics)
    
    # 11. Print summary
    print("\n=== Results Summary ===")
    print(f"Matched sample: {matched_sample_id}")
    print(f"Genes aligned: {len(aligned_pred)}")
    print(f"Aggregation method: {agg_method}")
    print(f"Zero fraction: {zero_fraction:.3f}")
    print(f"MSE (raw): {metrics.get('MSE_raw', 'N/A'):.6f}")
    print(f"sMAPE (raw): {metrics.get('sMAPE_raw', 'N/A'):.6f}")
    print(f"Pearson r (raw): {metrics.get('Pearson_r_raw', 'N/A'):.6f}")
    
    if args.log1p.lower() == "true":
        print(f"MSE (log1p): {metrics.get('MSE_log1p', 'N/A'):.6f}")
        print(f"sMAPE (log1p): {metrics.get('sMAPE_log1p', 'N/A'):.6f}")
        print(f"Pearson r (log1p): {metrics.get('Pearson_r_log1p', 'N/A'):.6f}")
    
    print(f"\nResults saved to: {args.out_dir}")

if __name__ == "__main__":
    main()

