#!/usr/bin/env python3
"""
analyze_sample_level_gene_pearson.py
=====================================
Calculate gene-wise Pearson correlation within each sample separately

This tests if mixing different samples reduces gene-wise correlation.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import json
from tqdm import tqdm

def load_spot_to_sample_mapping(csv_file):
    """
    Load mapping from spot index to sample ID
    
    Returns:
        dict: {index: sample_id}, list of sample_ids in order
    """
    # Try reading as CSV first, then TSV
    try:
        df = pd.read_csv(csv_file)
    except:
        df = pd.read_csv(csv_file, sep='\t')
    
    # Check if filepath column exists
    if 'filepath' not in df.columns:
        raise ValueError(f"CSV file must have 'filepath' column. Found: {df.columns.tolist()}")
    
    idx_to_sample = {}
    sample_ids_ordered = []
    
    for idx, row in df.iterrows():
        filepath = row['filepath']
        
        # Extract sample ID from filepath
        # Example: /path/to/GSM5494475/patches/AAAC...png
        parts = filepath.split('/')
        
        # Find GSM... part
        sample_id = None
        for part in parts:
            if part.startswith('GSM'):
                sample_id = part
                break
        
        if sample_id is None:
            # Fallback: use directory name
            sample_id = parts[-3] if len(parts) >= 3 else "UNKNOWN"
        
        idx_to_sample[idx] = sample_id
        sample_ids_ordered.append(sample_id)
    
    return idx_to_sample, sample_ids_ordered


def calculate_sample_level_gene_pearson(predictions_file, ground_truth_file, 
                                        csv_file, output_dir):
    """
    Calculate gene-wise Pearson correlation within each sample
    """
    print("="*70)
    print("Sample-Level Gene-wise Pearson Analysis")
    print("="*70)
    
    # Load data
    print("\n[1] Loading predictions and ground truth...")
    predictions = np.load(predictions_file)  # (N_spots, N_genes)
    ground_truth = np.load(ground_truth_file)  # (N_spots, N_genes)
    
    N_spots, N_genes = predictions.shape
    print(f"  Predictions: {predictions.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    
    # Load spot to sample mapping
    print("\n[2] Loading spot to sample mapping...")
    idx_to_sample, sample_ids_ordered = load_spot_to_sample_mapping(csv_file)
    
    # Handle mismatch
    if len(sample_ids_ordered) != N_spots:
        print(f"  Warning: CSV has {len(sample_ids_ordered)} spots, data has {N_spots}")
        print(f"  Using first {N_spots} spots from CSV")
        sample_ids_ordered = sample_ids_ordered[:N_spots]
        idx_to_sample = {i: sample_ids_ordered[i] for i in range(N_spots)}
    
    # Get unique samples
    samples = sorted(set(sample_ids_ordered))
    print(f"  Found {len(samples)} unique samples:")
    for sample in samples:
        n = sample_ids_ordered.count(sample)
        print(f"    - {sample}: {n} spots")
    
    # Calculate gene-wise correlation per sample
    print("\n[3] Calculating gene-wise Pearson per sample...")
    
    sample_results = []
    all_fold_gene_corrs = []  # For fold-level comparison
    
    for sample_id in tqdm(samples, desc="Samples"):
        # Get spot indices for this sample
        sample_indices = [i for i, s in enumerate(sample_ids_ordered) if s == sample_id]
        
        if len(sample_indices) < 10:
            print(f"  Warning: {sample_id} has only {len(sample_indices)} spots, skipping")
            continue
        
        # Get predictions and ground truth for this sample
        sample_pred = predictions[sample_indices]
        sample_gt = ground_truth[sample_indices]
        
        # Calculate gene-wise correlation
        gene_corrs = []
        for g in range(N_genes):
            if sample_gt[:, g].std() > 1e-8:
                r, p = pearsonr(sample_pred[:, g], sample_gt[:, g])
                if np.isfinite(r):
                    gene_corrs.append(r)
        
        if len(gene_corrs) > 0:
            sample_results.append({
                'sample_id': sample_id,
                'n_spots': len(sample_indices),
                'n_genes': len(gene_corrs),
                'gene_pearson_mean': float(np.mean(gene_corrs)),
                'gene_pearson_median': float(np.median(gene_corrs)),
                'gene_pearson_std': float(np.std(gene_corrs))
            })
    
    # Calculate fold-level (all spots mixed)
    print("\n[4] Calculating fold-level gene-wise Pearson (for comparison)...")
    fold_gene_corrs = []
    for g in tqdm(range(N_genes), desc="Genes"):
        if ground_truth[:, g].std() > 1e-8:
            r, p = pearsonr(predictions[:, g], ground_truth[:, g])
            if np.isfinite(r):
                fold_gene_corrs.append(r)
    
    fold_mean = float(np.mean(fold_gene_corrs))
    fold_median = float(np.median(fold_gene_corrs))
    
    # Overall statistics
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    
    print("\nPer-Sample Gene-wise Pearson:")
    for result in sample_results:
        print(f"  {result['sample_id']:30s}: "
              f"n_spots={result['n_spots']:5d}, "
              f"Gene Pearson={result['gene_pearson_mean']:7.4f} ± {result['gene_pearson_std']:.4f}")
    
    # Average across samples
    if len(sample_results) > 0:
        sample_level_mean = np.mean([r['gene_pearson_mean'] for r in sample_results])
        sample_level_std = np.std([r['gene_pearson_mean'] for r in sample_results])
        
        print("\n" + "="*70)
        print(f"Sample-level Gene Pearson: {sample_level_mean:.4f} ± {sample_level_std:.4f}")
        print(f"  (Average of {len(sample_results)} samples)")
        print(f"\nFold-level Gene Pearson:   {fold_mean:.4f}")
        print(f"  (All {N_spots} spots mixed)")
        print("\n" + "="*70)
        
        # Comparison
        improvement = sample_level_mean - fold_mean
        pct_improvement = (improvement / abs(fold_mean)) * 100 if fold_mean != 0 else 0
        
        print(f"\nComparison:")
        print(f"  Difference: {improvement:+.4f} ({pct_improvement:+.1f}%)")
        
        if sample_level_mean > fold_mean + 0.01:
            print(f"  → Sample-level is HIGHER!")
            print(f"  → Sample mixing reduces gene-wise correlation")
            print(f"  → Batch effect may be present")
        elif abs(sample_level_mean - fold_mean) < 0.01:
            print(f"  → Similar performance")
            print(f"  → Sample mixing effect is minimal")
        else:
            print(f"  → Fold-level is higher (unexpected)")
        
        print("="*70)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'sample_results': sample_results,
        'sample_level_mean': float(sample_level_mean) if len(sample_results) > 0 else None,
        'sample_level_std': float(sample_level_std) if len(sample_results) > 0 else None,
        'fold_level_mean': fold_mean,
        'fold_level_median': fold_median,
        'n_samples': len(sample_results),
        'n_spots_total': N_spots,
        'n_genes': N_genes
    }
    
    with open(output_dir / 'sample_level_gene_pearson.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Save CSV
    df_results = pd.DataFrame(sample_results)
    df_results.to_csv(output_dir / 'sample_level_results.csv', index=False)
    
    print(f"\n✓ Results saved to {output_dir}")
    print(f"  - sample_level_gene_pearson.json")
    print(f"  - sample_level_results.csv")


def main(args):
    calculate_sample_level_gene_pearson(
        args.predictions_file,
        args.ground_truth_file,
        args.csv_file,
        args.output_dir
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_file", required=True,
                   help="Path to predictions.npy")
    p.add_argument("--ground_truth_file", required=True,
                   help="Path to ground_truth.npy")
    p.add_argument("--csv_file", required=True,
                   help="Path to CSV file with filepath column")
    p.add_argument("--output_dir", default="./sample_level_analysis")
    
    args = p.parse_args()
    main(args)