#!/usr/bin/env python3
"""
Evaluate predictions using top high-variance genes from bulk RNA-seq.

This script:
1. Computes gene-wise variance from bulk RNA-seq (ground truth)
2. Selects top K genes by variance (K = 300, 500, 1000, 2000, All)
3. Evaluates predictions on selected genes for each K
4. Computes metrics: Overall PCC, Mean slide-wise PCC, Mean/Median gene-wise PCC
5. Generates comparison tables and visualizations

Usage:
    python evaluate_top_variance_genes.py \
        --pred_file results_predex/Y_slide.npy \
        --bulk_file ref_file_for_eval.csv \
        --gene_names results_predex/fold_01/gene_names.npy \
        --output_dir top_variance_eval_results
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_data(pred_file, bulk_file, gene_names_file):
    """
    Load prediction data, bulk RNA-seq data, and gene names.
    Aligns by slide IDs and gene names to find intersection.
    
    Returns:
        Y_pred_aligned: (N_slides, N_genes) prediction array (aligned)
        Y_bulk_aligned: (N_slides, N_genes) bulk RNA-seq array (aligned)
        gene_names_aligned: (N_genes,) common gene name array
        slide_ids: (N_slides,) common slide ID array
    """
    print("=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    # Load predictions
    print(f"[1] Loading predictions: {pred_file}")
    Y_pred = np.load(pred_file).astype(np.float32)
    print(f"    [Pred] Shape: {Y_pred.shape}")
    
    # Load slide IDs from predictions
    pred_file_path = Path(pred_file)
    slide_ids_file = pred_file_path.parent / "slide_ids.npy"
    
    if slide_ids_file.exists():
        print(f"[2] Loading slide IDs: {slide_ids_file}")
        slide_ids_pred = np.load(slide_ids_file, allow_pickle=True).astype(str)
        print(f"    [Pred] Slides: {len(slide_ids_pred)}")
    else:
        raise FileNotFoundError(f"slide_ids.npy not found in {pred_file_path.parent}")
    
    # Load gene names
    print(f"[3] Loading gene names: {gene_names_file}")
    gene_names_pred = np.load(gene_names_file, allow_pickle=True).astype(str)
    print(f"    [Pred] Genes: {len(gene_names_pred)}")
    
    # Load bulk RNA-seq (already preprocessed, rna_ prefix removed)
    print(f"[4] Loading bulk RNA-seq: {bulk_file}")
    bulk_df = pd.read_csv(bulk_file)
    
    # Handle slide_id column (should already be 'slide_id')
    if 'slide_id' not in bulk_df.columns:
        if "wsi_file_name" in bulk_df.columns:
            bulk_df["slide_id"] = bulk_df["wsi_file_name"].astype(str)
        else:
            first_col = bulk_df.columns[0]
            bulk_df = bulk_df.rename(columns={first_col: "slide_id"})
    
    bulk_df['slide_id'] = bulk_df['slide_id'].astype(str)
    bulk_df = bulk_df.set_index('slide_id')
    
    print(f"    [Bulk] Shape: {bulk_df.shape}")
    
    # Find common slides
    print("\n[5] Finding common slides...")
    common_slides = np.intersect1d(slide_ids_pred, bulk_df.index.values)
    print(f"    [Common] Slides: {len(common_slides)}")
    
    if len(common_slides) == 0:
        raise RuntimeError("No common slides found! Check slide_id normalization.")
    
    # Find common genes
    print("\n[6] Finding common genes...")
    bulk_gene_names = bulk_df.columns.values.astype(str)
    common_genes = np.intersect1d(gene_names_pred, bulk_gene_names)
    print(f"    [Common] Genes: {len(common_genes)}")
    
    if len(common_genes) == 0:
        raise RuntimeError("No common genes found! Check gene name matching.")
    
    # Align predictions to common slides and genes
    print("\n[7] Aligning predictions to common slides and genes...")
    pred_slide_idx = np.array([np.where(slide_ids_pred == s)[0][0] for s in common_slides])
    pred_gene_idx = np.array([np.where(gene_names_pred == g)[0][0] for g in common_genes])
    
    Y_pred_aligned = Y_pred[np.ix_(pred_slide_idx, pred_gene_idx)]
    print(f"    [Aligned] Y_pred: {Y_pred_aligned.shape}")
    
    # Align bulk to common slides and genes
    print("\n[8] Aligning bulk to common slides and genes...")
    Y_bulk_aligned = bulk_df.loc[common_slides, common_genes].values.astype(np.float32)
    print(f"    [Aligned] Y_bulk: {Y_bulk_aligned.shape}")
    
    # Check for NaN/Inf
    print(f"\n[9] Data quality check:")
    print(f"    Pred NaN: {np.isnan(Y_pred_aligned).sum()}, Inf: {np.isinf(Y_pred_aligned).sum()}")
    print(f"    Bulk NaN: {np.isnan(Y_bulk_aligned).sum()}, Inf: {np.isinf(Y_bulk_aligned).sum()}")
    
    gene_names_aligned = np.array(common_genes)
    slide_ids = np.array(common_slides)
    
    print(f"\n✅ Data loading complete!")
    print(f"    Final shape: {Y_pred_aligned.shape}")
    print(f"    Common slides: {len(slide_ids)}")
    print(f"    Common genes: {len(gene_names_aligned)}")
    
    return Y_pred_aligned, Y_bulk_aligned, gene_names_aligned, slide_ids


def compute_gene_variances(Y_bulk, gene_names):
    """
    Compute gene-wise variance from bulk RNA-seq.
    
    Args:
        Y_bulk: (N_slides, N_genes) bulk RNA-seq array
        gene_names: (N_genes,) gene name array
    
    Returns:
        gene_variances: (N_genes,) variance array
        sorted_indices: Indices sorted by variance (high to low)
    """
    print("\n" + "=" * 80)
    print("Computing Gene Variances")
    print("=" * 80)
    
    gene_variances = np.var(Y_bulk, axis=0)
    sorted_indices = np.argsort(gene_variances)[::-1]  # High to low
    
    print(f"Variance range: [{gene_variances.min():.6f}, {gene_variances.max():.6f}]")
    print(f"Mean variance: {gene_variances.mean():.6f}")
    print(f"Median variance: {np.median(gene_variances):.6f}")
    
    print(f"\nTop 10 genes by variance:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"  {i+1:2d}. {gene_names[idx]:15s}: variance={gene_variances[idx]:.6f}")
    
    return gene_variances, sorted_indices


def select_top_k_genes(gene_variances, K):
    """
    Select top K genes by variance.
    
    Args:
        gene_variances: (N_genes,) variance array
        K: Number of genes to select (if K >= len(gene_variances), returns all)
    
    Returns:
        top_k_indices: Indices of top K genes
    """
    sorted_indices = np.argsort(gene_variances)[::-1]  # High to low
    
    if K >= len(gene_variances):
        return sorted_indices
    
    return sorted_indices[:K]


def evaluate(Y_pred, Y_bulk, gene_names, K, min_var=1e-8):
    """
    Evaluate predictions against ground truth.
    
    Args:
        Y_pred: (N_slides, N_genes) prediction array
        Y_bulk: (N_slides, N_genes) ground truth array
        gene_names: (N_genes,) gene name array
        K: K value (for display)
        min_var: Minimum variance threshold for PCC computation
    
    Returns:
        Dictionary with evaluation metrics
    """
    n_slides, n_genes = Y_pred.shape
    
    # Overall PCC (flatten)
    pred_flat = Y_pred.flatten()
    bulk_flat = Y_bulk.flatten()
    
    # Remove NaN/Inf
    valid_mask = np.isfinite(pred_flat) & np.isfinite(bulk_flat)
    if valid_mask.sum() == 0:
        overall_pcc = np.nan
    else:
        overall_pcc, _ = pearsonr(pred_flat[valid_mask], bulk_flat[valid_mask])
    
    # Slide-wise PCC
    slide_pccs = []
    for i in range(n_slides):
        y_pred_slide = Y_pred[i, :]
        y_bulk_slide = Y_bulk[i, :]
        
        # Check variance
        if np.var(y_pred_slide) < min_var or np.var(y_bulk_slide) < min_var:
            slide_pccs.append(np.nan)
        else:
            valid_mask = np.isfinite(y_pred_slide) & np.isfinite(y_bulk_slide)
            if valid_mask.sum() < 2:
                slide_pccs.append(np.nan)
            else:
                try:
                    pcc, _ = pearsonr(y_pred_slide[valid_mask], y_bulk_slide[valid_mask])
                    if not np.isnan(pcc):
                        slide_pccs.append(pcc)
                    else:
                        slide_pccs.append(np.nan)
                except:
                    slide_pccs.append(np.nan)
    
    slide_pccs = np.array(slide_pccs)
    valid_slide_pccs = slide_pccs[~np.isnan(slide_pccs)]
    
    # Gene-wise PCC
    gene_pccs = []
    gene_pcc_dict = {}
    for j in range(n_genes):
        y_pred_gene = Y_pred[:, j]
        y_bulk_gene = Y_bulk[:, j]
        
        # Check variance
        if np.var(y_pred_gene) < min_var or np.var(y_bulk_gene) < min_var:
            gene_pccs.append(np.nan)
        else:
            valid_mask = np.isfinite(y_pred_gene) & np.isfinite(y_bulk_gene)
            if valid_mask.sum() < 2:
                gene_pccs.append(np.nan)
            else:
                try:
                    pcc, _ = pearsonr(y_pred_gene[valid_mask], y_bulk_gene[valid_mask])
                    if not np.isnan(pcc):
                        gene_pccs.append(pcc)
                        gene_pcc_dict[gene_names[j]] = pcc
                    else:
                        gene_pccs.append(np.nan)
                except:
                    gene_pccs.append(np.nan)
    
    gene_pccs = np.array(gene_pccs)
    valid_gene_pccs = gene_pccs[~np.isnan(gene_pccs)]
    
    mean_gene_pcc = np.nanmean(gene_pccs) if len(valid_gene_pccs) > 0 else np.nan
    median_gene_pcc = np.nanmedian(gene_pccs) if len(valid_gene_pccs) > 0 else np.nan
    
    # Top 10 genes by PCC
    top10_genes = sorted(gene_pcc_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n[Results for K={K}]")
    print(f"  Overall PCC: {overall_pcc:.4f}")
    print(f"  Mean Slide PCC: {np.nanmean(slide_pccs):.4f}")
    print(f"  Mean Gene PCC: {mean_gene_pcc:.4f}")
    print(f"  Median Gene PCC: {median_gene_pcc:.4f}")
    print(f"  Valid slides: {len(valid_slide_pccs)}/{n_slides}")
    print(f"  Valid genes: {len(valid_gene_pccs)}/{n_genes}")
    
    if len(top10_genes) > 0:
        print(f"\n  Top 10 genes by PCC:")
        for idx, (gene, pcc) in enumerate(top10_genes, 1):
            print(f"    {idx:2d}. {gene:15s}: {pcc:.4f}")
    
    return {
        'overall_pcc': overall_pcc,
        'mean_slide_pcc': np.nanmean(slide_pccs),
        'median_slide_pcc': np.nanmedian(slide_pccs),
        'std_slide_pcc': np.nanstd(slide_pccs),
        'mean_gene_pcc': mean_gene_pcc,
        'median_gene_pcc': median_gene_pcc,
        'std_gene_pcc': np.nanstd(gene_pccs),
        'n_valid_slides': len(valid_slide_pccs),
        'n_valid_genes': len(valid_gene_pccs),
        'slide_wise_pccs': slide_pccs,
        'gene_wise_pccs': gene_pccs,
    }


def plot_gene_wise_pcc_distribution(gene_pccs, K, n_genes, output_path):
    """Plot gene-wise PCC distribution as histogram."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    valid_pccs = gene_pccs[~np.isnan(gene_pccs)]
    
    if len(valid_pccs) > 0:
        ax.hist(valid_pccs, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(np.mean(valid_pccs), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(valid_pccs):.3f}')
        ax.axvline(np.median(valid_pccs), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(valid_pccs):.3f}')
        ax.set_xlabel('Gene-wise PCC', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Gene-wise PCC Distribution (K={K}, {n_genes} genes)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f'N: {len(valid_pccs)}\nStd: {np.std(valid_pccs):.4f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No valid PCCs', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_variance_vs_pcc(gene_variances, gene_pccs, K, output_path):
    """Plot gene variance vs gene PCC scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_mask = ~np.isnan(gene_pccs)
    valid_variances = gene_variances[valid_mask]
    valid_pccs = gene_pccs[valid_mask]
    
    if len(valid_pccs) > 0:
        scatter = ax.scatter(valid_variances, valid_pccs, alpha=0.5, s=20)
        ax.set_xlabel('Gene Variance (Bulk RNA-seq)', fontsize=12)
        ax.set_ylabel('Gene-wise PCC', fontsize=12)
        ax.set_title(f'Gene Variance vs Gene-wise PCC (K={K})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        if len(valid_pccs) > 1:
            corr, _ = pearsonr(valid_variances, valid_pccs)
            ax.text(0.02, 0.98, f'Correlation: {corr:.4f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Log scale for variance if needed
        if valid_variances.max() / valid_variances.min() > 100:
            ax.set_xscale('log')
    else:
        ax.text(0.5, 0.5, 'No valid PCCs', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_across_K(results_dict, output_path):
    """Plot comparison bar chart across K values."""
    # Sort K values: numeric first (ascending), then "All" at the end
    def sort_key(x):
        if x == "All":
            return (1, float('inf'))
        try:
            return (0, int(x))
        except:
            return (0, 0)
    
    K_values = sorted(results_dict.keys(), key=sort_key)
    
    metrics = ['mean_gene_pcc', 'median_gene_pcc', 'mean_slide_pcc']
    metric_labels = ['Mean Gene-wise PCC', 'Median Gene-wise PCC', 'Mean Slide-wise PCC']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [results_dict[k][metric] for k in K_values]
        
        bars = ax.bar(range(len(K_values)), values, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_xticks(range(len(K_values)))
        ax.set_xticklabels([f'K={k}' for k in K_values], rotation=45, ha='right')
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val,
                       f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
    
    plt.suptitle('Evaluation Metrics Comparison Across K Values', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions using top high-variance genes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred_file", type=str, required=True,
                       help="Path to Y_slide.npy prediction file")
    parser.add_argument("--bulk_file", type=str, required=True,
                       help="Path to ref_file_for_eval.csv bulk RNA-seq file")
    parser.add_argument("--gene_names", type=str, required=True,
                       help="Path to gene_names.npy file")
    parser.add_argument("--output_dir", type=str, default="top_variance_eval_results",
                       help="Output directory for results")
    parser.add_argument("--K_values", type=int, nargs='+', default=[300, 500, 1000, 2000],
                       help="K values to evaluate (default: 300 500 1000 2000)")
    parser.add_argument("--include_all", action="store_true",
                       help="Include evaluation on all genes (baseline)")
    parser.add_argument("--min_var", type=float, default=1e-8,
                       help="Minimum variance threshold for PCC (default: 1e-8)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print("=" * 80)
        print("Top Variance Genes Evaluation")
        print("=" * 80)
        print(f"Prediction file: {args.pred_file}")
        print(f"Bulk file: {args.bulk_file}")
        print(f"Gene names file: {args.gene_names}")
        print(f"Output directory: {output_dir}")
        print(f"K values: {args.K_values}")
        print(f"Include all genes: {args.include_all}")
        print("=" * 80)
        print()
    
    # Load data
    Y_pred, Y_bulk, gene_names, slide_ids = load_data(
        args.pred_file, args.bulk_file, args.gene_names
    )
    
    # Compute gene variances
    gene_variances, sorted_indices = compute_gene_variances(Y_bulk, gene_names)
    
    # Determine K values to evaluate
    K_values = list(args.K_values)
    if args.include_all:
        K_values.append(len(gene_names))
    
    # Store results
    results_dict = {}
    
    print("\n" + "=" * 80)
    print("Evaluating for Different K Values")
    print("=" * 80)
    
    # Evaluate for each K
    for K in K_values:
        if isinstance(K, int) and K > len(gene_names):
            print(f"\n⚠️  K={K} exceeds number of genes ({len(gene_names)}), skipping...")
            continue
        
        if K == len(gene_names):
            K_label = "All"
            top_k_indices = np.arange(len(gene_names))
        else:
            K_label = str(K)
            top_k_indices = sorted_indices[:K]
        
        print(f"\n{'=' * 80}")
        print(f"Evaluating K={K_label} (Top {K} variance genes)")
        print(f"{'=' * 80}")
        
        # Select subset
        Y_pred_subset = Y_pred[:, top_k_indices]
        Y_bulk_subset = Y_bulk[:, top_k_indices]
        gene_names_subset = gene_names[top_k_indices]
        gene_variances_subset = gene_variances[top_k_indices]
        
        print(f"Subset shape: {Y_pred_subset.shape}")
        
        # Evaluate
        results = evaluate(Y_pred_subset, Y_bulk_subset, gene_names_subset, K_label, min_var=args.min_var)
        results['K'] = K_label
        results['n_genes'] = len(top_k_indices)
        results['selected_gene_names'] = gene_names_subset.tolist()
        results['selected_gene_variances'] = gene_variances_subset.tolist()
        
        results_dict[K_label] = results
        
        # Save selected gene names
        gene_names_file = output_dir / f"top_{K_label}_variance_gene_names.txt"
        with open(gene_names_file, 'w') as f:
            for gene in gene_names_subset:
                f.write(f"{gene}\n")
        print(f"\n  ✅ Saved gene names to: {gene_names_file}")
        
        # Visualizations
        print(f"\n  Generating visualizations...")
        
        # Gene-wise PCC distribution (histogram)
        hist_path = output_dir / f"gene_wise_pcc_distribution_K{K_label}.png"
        plot_gene_wise_pcc_distribution(results['gene_wise_pccs'], K_label, len(top_k_indices), hist_path)
        print(f"    ✅ Saved: {hist_path}")
        
        # Variance vs PCC scatter (only for K=300)
        if K == 300:
            scatter_path = output_dir / f"variance_vs_pcc_K{K_label}.png"
            plot_variance_vs_pcc(gene_variances_subset, results['gene_wise_pccs'], K_label, scatter_path)
            print(f"    ✅ Saved: {scatter_path}")
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("Creating Comparison Table")
    print("=" * 80)
    
    comparison_data = []
    # Sort K values: numeric first (ascending), then "All" at the end
    def sort_key(x):
        if x == "All":
            return (1, float('inf'))
        try:
            return (0, int(x))
        except:
            return (0, 0)
    
    for K_label in sorted(results_dict.keys(), key=sort_key):
        r = results_dict[K_label]
        comparison_data.append({
            'K': K_label,
            'N_Genes': r['n_genes'],
            'Overall_PCC': r['overall_pcc'],
            'Mean_Slide_PCC': r['mean_slide_pcc'],
            'Mean_Gene_PCC': r['mean_gene_pcc'],
            'Median_Gene_PCC': r['median_gene_pcc'],
            'Std_Gene_PCC': r['std_gene_pcc'],
            'N_Valid_Slides': r['n_valid_slides'],
            'N_Valid_Genes': r['n_valid_genes'],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = output_dir / "top_variance_genes_evaluation.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n✅ Comparison table saved to: {comparison_csv}")
    print("\n" + comparison_df.to_string(index=False))
    
    # Comparison visualization
    comparison_plot_path = output_dir / "comparison_across_K.png"
    plot_comparison_across_K(results_dict, comparison_plot_path)
    print(f"\n✅ Comparison plot saved to: {comparison_plot_path}")
    
    # Save detailed results as JSON
    results_json_path = output_dir / "detailed_results.json"
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for K_label, r in results_dict.items():
        json_results[K_label] = {}
        for k, v in r.items():
            if k in ['slide_wise_pccs', 'gene_wise_pccs']:
                # Skip large arrays to save space
                continue
            elif isinstance(v, np.ndarray):
                json_results[K_label][k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                json_results[K_label][k] = float(v)
            else:
                json_results[K_label][k] = v
    
    with open(results_json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✅ Detailed results saved to: {results_json_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Results directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - top_variance_genes_evaluation.csv (comparison table)")
    print(f"  - comparison_across_K.png (comparison plots)")
    print(f"  - detailed_results.json (detailed metrics)")
    print(f"  - top_*_variance_gene_names.txt (selected gene lists)")
    print(f"  - gene_wise_pcc_distribution_K*.png (PCC distributions)")
    print(f"  - variance_vs_pcc_K300.png (variance vs PCC scatter plot)")
    print("=" * 80)


if __name__ == "__main__":
    main()
