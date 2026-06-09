#!/usr/bin/env python3
"""
Slide-level Calibrated Metrics Calculator

This script aligns genes, aggregates by SUM and MEAN, applies two scalings 
(alpha_LS and alpha_L1), and reports MSE, sMAPE, Pearson/Spearman on raw, 
alpha-scaled, log1p, and zero-aware variants.
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
        orientation = 'genes_as_rows'
    else:
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
    
    print("Top 5 candidates:")
    for i, (sim, idx, sample_id) in enumerate(similarities[:5]):
        print(f"  {i+1}. {sample_id} (similarity: {sim:.3f})")
    
    if similarities[0][0] > 0.3:  # Minimum similarity threshold
        best_match_idx = similarities[0][1]
        best_match_id = similarities[0][2]
        print(f"Found fuzzy match: {best_match_id} (similarity: {similarities[0][0]:.3f})")
        return best_match_id, tcga_df.loc[best_match_idx]
    
    # No good match found
    print("WARNING: No good match found!")
    # Return the best match anyway
    best_match_idx = similarities[0][1]
    best_match_id = similarities[0][2]
    return best_match_id, tcga_df.loc[best_match_idx]

def compute_scaling_factors(pred_vector, gt_vector):
    """
    Compute scaling factors alpha_LS and alpha_L1.
    alpha_LS = (p·g)/(p·p)  # Least squares
    alpha_L1 = (sum g)/(sum p)  # L1 scaling
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(pred_vector) | np.isnan(gt_vector))
    pred_valid = pred_vector[valid_mask]
    gt_valid = gt_vector[valid_mask]
    
    if len(pred_valid) == 0:
        return np.nan, np.nan
    
    # alpha_LS = (p·g)/(p·p)
    dot_product = np.dot(pred_valid, gt_valid)
    pred_norm_squared = np.dot(pred_valid, pred_valid)
    alpha_LS = dot_product / pred_norm_squared if pred_norm_squared > 0 else np.nan
    
    # alpha_L1 = (sum g)/(sum p)
    gt_sum = np.sum(gt_valid)
    pred_sum = np.sum(pred_valid)
    alpha_L1 = gt_sum / pred_sum if pred_sum > 0 else np.nan
    
    return alpha_LS, alpha_L1

def calculate_metrics(pred_vector, gt_vector, variant_name=""):
    """
    Calculate comprehensive metrics for a pair of vectors.
    Returns metrics for raw, log1p, and zero-aware variants.
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(pred_vector) | np.isnan(gt_vector))
    pred_valid = pred_vector[valid_mask]
    gt_valid = gt_vector[valid_mask]
    
    if len(pred_valid) == 0:
        return {
            'n_genes': 0,
            'MSE_raw': np.nan, 'sMAPE_raw': np.nan, 'Pearson_r_raw': np.nan, 'Spearman_r_raw': np.nan,
            'MSE_log1p': np.nan, 'sMAPE_log1p': np.nan, 'Pearson_r_log1p': np.nan, 'Spearman_r_log1p': np.nan,
            'MSE_zero_aware': np.nan, 'sMAPE_zero_aware': np.nan, 'Pearson_r_zero_aware': np.nan, 'Spearman_r_zero_aware': np.nan
        }
    
    metrics = {'n_genes': len(pred_valid)}
    
    # Raw metrics
    mse_raw = np.mean((pred_valid - gt_valid) ** 2)
    numerator = 2 * np.abs(pred_valid - gt_valid)
    denominator = np.abs(pred_valid) + np.abs(gt_valid) + 1e-8
    smape_raw = np.mean(numerator / denominator)
    
    if len(pred_valid) > 1 and (np.std(pred_valid) > 1e-8 or np.std(gt_valid) > 1e-8):
        pearson_r_raw, pearson_p_raw = stats.pearsonr(pred_valid, gt_valid)
        spearman_r_raw, spearman_p_raw = stats.spearmanr(pred_valid, gt_valid)
    else:
        pearson_r_raw, spearman_r_raw = np.nan, np.nan
    
    metrics.update({
        'MSE_raw': mse_raw,
        'sMAPE_raw': smape_raw,
        'Pearson_r_raw': pearson_r_raw,
        'Spearman_r_raw': spearman_r_raw
    })
    
    # Log1p metrics
    pred_log1p = np.log1p(pred_valid)
    gt_log1p = np.log1p(gt_valid)
    
    mse_log1p = np.mean((pred_log1p - gt_log1p) ** 2)
    numerator_log1p = 2 * np.abs(pred_log1p - gt_log1p)
    denominator_log1p = np.abs(pred_log1p) + np.abs(gt_log1p) + 1e-8
    smape_log1p = np.mean(numerator_log1p / denominator_log1p)
    
    if len(pred_log1p) > 1 and (np.std(pred_log1p) > 1e-8 or np.std(gt_log1p) > 1e-8):
        pearson_r_log1p, pearson_p_log1p = stats.pearsonr(pred_log1p, gt_log1p)
        spearman_r_log1p, spearman_p_log1p = stats.spearmanr(pred_log1p, gt_log1p)
    else:
        pearson_r_log1p, spearman_r_log1p = np.nan, np.nan
    
    metrics.update({
        'MSE_log1p': mse_log1p,
        'sMAPE_log1p': smape_log1p,
        'Pearson_r_log1p': pearson_r_log1p,
        'Spearman_r_log1p': spearman_r_log1p
    })
    
    # Zero-aware metrics (only non-zero values)
    zero_aware_mask = (pred_valid > 0) | (gt_valid > 0)
    if zero_aware_mask.sum() > 0:
        pred_zero_aware = pred_valid[zero_aware_mask]
        gt_zero_aware = gt_valid[zero_aware_mask]
        
        mse_zero_aware = np.mean((pred_zero_aware - gt_zero_aware) ** 2)
        numerator_za = 2 * np.abs(pred_zero_aware - gt_zero_aware)
        denominator_za = np.abs(pred_zero_aware) + np.abs(gt_zero_aware) + 1e-8
        smape_zero_aware = np.mean(numerator_za / denominator_za)
        
        if len(pred_zero_aware) > 1 and (np.std(pred_zero_aware) > 1e-8 or np.std(gt_zero_aware) > 1e-8):
            pearson_r_zero_aware, pearson_p_zero_aware = stats.pearsonr(pred_zero_aware, gt_zero_aware)
            spearman_r_zero_aware, spearman_p_zero_aware = stats.spearmanr(pred_zero_aware, gt_zero_aware)
        else:
            pearson_r_zero_aware, spearman_r_zero_aware = np.nan, np.nan
        
        metrics.update({
            'MSE_zero_aware': mse_zero_aware,
            'sMAPE_zero_aware': smape_zero_aware,
            'Pearson_r_zero_aware': pearson_r_zero_aware,
            'Spearman_r_zero_aware': spearman_r_zero_aware
        })
    else:
        metrics.update({
            'MSE_zero_aware': np.nan,
            'sMAPE_zero_aware': np.nan,
            'Pearson_r_zero_aware': np.nan,
            'Spearman_r_zero_aware': np.nan
        })
    
    return metrics

def create_scatter_plot(pred_vector, gt_vector, output_path, title_suffix="", log1p=False):
    """Create scatter plot of predictions vs ground truth"""
    plt.figure(figsize=(8, 8))
    
    # Remove NaN values
    valid_mask = ~(np.isnan(pred_vector) | np.isnan(gt_vector))
    pred_valid = pred_vector[valid_mask]
    gt_valid = gt_vector[valid_mask]
    
    # Check if we have any valid data
    if len(pred_valid) == 0:
        plt.text(0.5, 0.5, 'No valid data points for plotting', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'Predicted vs Ground Truth Expression{title_suffix} (No Data)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    if log1p:
        pred_valid = np.log1p(pred_valid)
        gt_valid = np.log1p(gt_valid)
        xlabel = 'Log1p(Predicted Expression)'
        ylabel = 'Log1p(Ground Truth Expression)'
        plot_suffix = ' (Log1p Transformed)'
    else:
        xlabel = 'Predicted Expression'
        ylabel = 'Ground Truth Expression'
        plot_suffix = ''
    
    plt.scatter(pred_valid, gt_valid, alpha=0.6, s=20)
    
    # Add diagonal line
    min_val = min(pred_valid.min(), gt_valid.min())
    max_val = max(pred_valid.max(), gt_valid.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect correlation')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Predicted vs Ground Truth Expression{title_suffix}{plot_suffix}')
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
    parser = argparse.ArgumentParser(description="Calculate slide-level calibrated metrics")
    parser.add_argument("--pred_csv", required=True, help="Path to gene expression predictions CSV")
    parser.add_argument("--tcga_csv", required=True, help="Path to TCGA bulk expression CSV")
    parser.add_argument("--slide_id", required=True, help="Slide ID to match")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=== Slide-level Calibrated Metrics Calculator ===")
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
    
    # 4. Aggregate to slide level (both SUM and MEAN)
    print("Aggregating to slide level...")
    pred_sum = pred_df.sum(axis=1)  # Sum across patches for each gene
    pred_mean = pred_df.mean(axis=1)  # Mean across patches for each gene
    
    print(f"SUM aggregation: {pred_sum.shape}")
    print(f"MEAN aggregation: {pred_mean.shape}")
    
    # 5. Load TCGA data and find matching sample
    print("Loading TCGA data...")
    tcga_df = pd.read_csv(args.tcga_csv, index_col=0)
    print(f"TCGA data shape: {tcga_df.shape}")
    
    matched_sample_id, matched_sample_data = find_matching_sample(tcga_df, args.slide_id)
    print(f"Matched sample: {matched_sample_id}")
    
    # 6. Normalize gene IDs for alignment
    print("Normalizing gene IDs...")
    pred_genes_sum = normalize_gene_ids(pred_sum.index)
    pred_genes_mean = normalize_gene_ids(pred_mean.index)
    tcga_genes = normalize_gene_ids(matched_sample_data.index)
    
    # Create mapping
    pred_gene_map_sum = dict(zip(pred_genes_sum, pred_sum.index))
    pred_gene_map_mean = dict(zip(pred_genes_mean, pred_mean.index))
    tcga_gene_map = dict(zip(tcga_genes, matched_sample_data.index))
    
    # Find common genes
    common_genes_sum = set(pred_genes_sum) & set(tcga_genes)
    common_genes_mean = set(pred_genes_mean) & set(tcga_genes)
    
    print(f"Found {len(common_genes_sum)} common genes for SUM")
    print(f"Found {len(common_genes_mean)} common genes for MEAN")
    
    # 7. Build aligned vectors
    print("Building aligned vectors...")
    
    # SUM vectors
    aligned_pred_sum = []
    aligned_gt_sum = []
    aligned_gene_names_sum = []
    
    for gene in common_genes_sum:
        pred_idx = pred_gene_map_sum[gene]
        tcga_idx = tcga_gene_map[gene]
        
        aligned_pred_sum.append(pred_sum[pred_idx])
        aligned_gt_sum.append(matched_sample_data[tcga_idx])
        aligned_gene_names_sum.append(gene)
    
    aligned_pred_sum = np.array(aligned_pred_sum)
    aligned_gt_sum = np.array(aligned_gt_sum)
    
    # MEAN vectors
    aligned_pred_mean = []
    aligned_gt_mean = []
    aligned_gene_names_mean = []
    
    for gene in common_genes_mean:
        pred_idx = pred_gene_map_mean[gene]
        tcga_idx = tcga_gene_map[gene]
        
        aligned_pred_mean.append(pred_mean[pred_idx])
        aligned_gt_mean.append(matched_sample_data[tcga_idx])
        aligned_gene_names_mean.append(gene)
    
    aligned_pred_mean = np.array(aligned_pred_mean)
    aligned_gt_mean = np.array(aligned_gt_mean)
    
    print(f"SUM aligned data: {len(aligned_pred_sum)} genes")
    print(f"MEAN aligned data: {len(aligned_pred_mean)} genes")
    
    # 8. Compute scaling factors
    print("Computing scaling factors...")
    
    alpha_LS_sum, alpha_L1_sum = compute_scaling_factors(aligned_pred_sum, aligned_gt_sum)
    alpha_LS_mean, alpha_L1_mean = compute_scaling_factors(aligned_pred_mean, aligned_gt_mean)
    
    print(f"SUM - alpha_LS: {alpha_LS_sum:.6f}, alpha_L1: {alpha_L1_sum:.6f}")
    print(f"MEAN - alpha_LS: {alpha_LS_mean:.6f}, alpha_L1: {alpha_L1_mean:.6f}")
    
    # 9. Calculate metrics for all combinations
    print("Calculating metrics...")
    
    metrics_results = []
    
    # SUM variants
    metrics_sum_raw = calculate_metrics(aligned_pred_sum, aligned_gt_sum, "SUM_raw")
    metrics_sum_raw['setting'] = 'SUM_raw'
    metrics_results.append(metrics_sum_raw)
    
    if not np.isnan(alpha_LS_sum):
        pred_sum_ls = aligned_pred_sum * alpha_LS_sum
        metrics_sum_ls = calculate_metrics(pred_sum_ls, aligned_gt_sum, "SUM_alpha_LS")
        metrics_sum_ls['setting'] = 'SUM_alpha_LS'
        metrics_results.append(metrics_sum_ls)
    
    if not np.isnan(alpha_L1_sum):
        pred_sum_l1 = aligned_pred_sum * alpha_L1_sum
        metrics_sum_l1 = calculate_metrics(pred_sum_l1, aligned_gt_sum, "SUM_alpha_L1")
        metrics_sum_l1['setting'] = 'SUM_alpha_L1'
        metrics_results.append(metrics_sum_l1)
    
    # MEAN variants
    metrics_mean_raw = calculate_metrics(aligned_pred_mean, aligned_gt_mean, "MEAN_raw")
    metrics_mean_raw['setting'] = 'MEAN_raw'
    metrics_results.append(metrics_mean_raw)
    
    if not np.isnan(alpha_LS_mean):
        pred_mean_ls = aligned_pred_mean * alpha_LS_mean
        metrics_mean_ls = calculate_metrics(pred_mean_ls, aligned_gt_mean, "MEAN_alpha_LS")
        metrics_mean_ls['setting'] = 'MEAN_alpha_LS'
        metrics_results.append(metrics_mean_ls)
    
    if not np.isnan(alpha_L1_mean):
        pred_mean_l1 = aligned_pred_mean * alpha_L1_mean
        metrics_mean_l1 = calculate_metrics(pred_mean_l1, aligned_gt_mean, "MEAN_alpha_L1")
        metrics_mean_l1['setting'] = 'MEAN_alpha_L1'
        metrics_results.append(metrics_mean_l1)
    
    # 10. Create metrics table
    print("Creating metrics table...")
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df = metrics_df.set_index('setting')
    metrics_df.to_csv(os.path.join(args.out_dir, 'metrics_table.csv'))
    
    # 11. Create scatter plots for best settings
    print("Creating scatter plots...")
    
    # Find best settings based on Pearson correlation (raw)
    best_settings = []
    for setting in metrics_df.index:
        if 'raw' in setting and not np.isnan(metrics_df.loc[setting, 'Pearson_r_raw']):
            best_settings.append((setting, metrics_df.loc[setting, 'Pearson_r_raw']))
    
    best_settings.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if len(best_settings) >= 2:
        best_setting1 = best_settings[0][0]
        best_setting2 = best_settings[1][0]
        
        print(f"Best settings for plotting: {best_setting1}, {best_setting2}")
        
        # Create plots for best settings
        if 'SUM' in best_setting1:
            if 'alpha_LS' in best_setting1:
                pred_plot = aligned_pred_sum * alpha_LS_sum
            elif 'alpha_L1' in best_setting1:
                pred_plot = aligned_pred_sum * alpha_L1_sum
            else:
                pred_plot = aligned_pred_sum
            gt_plot = aligned_gt_sum
        else:  # MEAN
            if 'alpha_LS' in best_setting1:
                pred_plot = aligned_pred_mean * alpha_LS_mean
            elif 'alpha_L1' in best_setting1:
                pred_plot = aligned_pred_mean * alpha_L1_mean
            else:
                pred_plot = aligned_pred_mean
            gt_plot = aligned_gt_mean
        
        create_scatter_plot(pred_plot, gt_plot, 
                           os.path.join(args.out_dir, f'scatter_{best_setting1}.png'), 
                           f' ({best_setting1})', log1p=False)
        create_scatter_plot(pred_plot, gt_plot, 
                           os.path.join(args.out_dir, f'scatter_{best_setting1}_log1p.png'), 
                           f' ({best_setting1})', log1p=True)
        
        # Second best setting
        if 'SUM' in best_setting2:
            if 'alpha_LS' in best_setting2:
                pred_plot2 = aligned_pred_sum * alpha_LS_sum
            elif 'alpha_L1' in best_setting2:
                pred_plot2 = aligned_pred_sum * alpha_L1_sum
            else:
                pred_plot2 = aligned_pred_sum
            gt_plot2 = aligned_gt_sum
        else:  # MEAN
            if 'alpha_LS' in best_setting2:
                pred_plot2 = aligned_pred_mean * alpha_LS_mean
            elif 'alpha_L1' in best_setting2:
                pred_plot2 = aligned_pred_mean * alpha_L1_mean
            else:
                pred_plot2 = aligned_pred_mean
            gt_plot2 = aligned_gt_mean
        
        create_scatter_plot(pred_plot2, gt_plot2, 
                           os.path.join(args.out_dir, f'scatter_{best_setting2}.png'), 
                           f' ({best_setting2})', log1p=False)
        create_scatter_plot(pred_plot2, gt_plot2, 
                           os.path.join(args.out_dir, f'scatter_{best_setting2}_log1p.png'), 
                           f' ({best_setting2})', log1p=True)
    
    # 12. Create diagnostics
    print("Creating diagnostics...")
    
    diagnostics = f"""
Slide-level Calibrated Metrics Diagnostics
=========================================

Input Files:
- Prediction CSV: {args.pred_csv}
- TCGA CSV: {args.tcga_csv}
- Slide ID: {args.slide_id}

Data Processing:
- Original prediction shape: {pred_df.shape}
- Detected orientation: {orientation}
- Final shape (genes × patches): {pred_df.shape}

Sample Matching:
- Target base ID: {extract_base_sample_id(args.slide_id)}
- Matched sample: {matched_sample_id}
- TCGA data shape: {tcga_df.shape}

Gene Alignment:
- Prediction genes: {len(pred_genes_sum)}
- TCGA genes: {len(tcga_genes)}
- Common genes (SUM): {len(common_genes_sum)}
- Common genes (MEAN): {len(common_genes_mean)}

Scaling Factors:
- SUM - alpha_LS: {alpha_LS_sum:.6f}
- SUM - alpha_L1: {alpha_L1_sum:.6f}
- MEAN - alpha_LS: {alpha_LS_mean:.6f}
- MEAN - alpha_L1: {alpha_L1_mean:.6f}

Zero Fractions:
- SUM prediction: {(aligned_pred_sum == 0).mean():.3f}
- MEAN prediction: {(aligned_pred_mean == 0).mean():.3f}
- GT: {(aligned_gt_sum == 0).mean():.3f}

Best Settings (by Pearson r):
"""
    
    for i, (setting, corr) in enumerate(best_settings[:5]):
        diagnostics += f"- {i+1}. {setting}: r = {corr:.6f}\n"
    
    with open(os.path.join(args.out_dir, 'diagnostics.txt'), 'w') as f:
        f.write(diagnostics)
    
    # 13. Print summary
    print("\n=== Results Summary ===")
    print(f"Matched sample: {matched_sample_id}")
    print(f"Genes aligned (SUM): {len(aligned_pred_sum)}")
    print(f"Genes aligned (MEAN): {len(aligned_pred_mean)}")
    print(f"Scaling factors computed successfully")
    
    print(f"\nBest settings by Pearson correlation:")
    for i, (setting, corr) in enumerate(best_settings[:3]):
        print(f"  {i+1}. {setting}: r = {corr:.6f}")
    
    print(f"\nResults saved to: {args.out_dir}")
    print(f"Files created:")
    print(f"- metrics_table.csv")
    print(f"- scatter_*.png")
    print(f"- diagnostics.txt")

if __name__ == "__main__":
    main()
