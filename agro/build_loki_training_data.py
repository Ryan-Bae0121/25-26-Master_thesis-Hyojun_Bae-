#!/usr/bin/env python3
"""
Loki PredEx Training Data Builder

This script constructs the following 4 files required for Loki Original PredEx:
1. combined_expression_matrix.npy
2. combined_obs.npy
3. all_shared_genes.txt
4. train_df.csv

It processes 36 Spatial Transcriptomics samples and filters spots based on meta.csv.
"""

import argparse
import glob
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from tqdm import tqdm
from typing import List, Set, Tuple


def find_all_h5ad_files(root_dir: str) -> List[str]:
    """Find all st_norm_noHK.h5ad files recursively."""
    pattern = str(Path(root_dir) / "**" / "st_norm_noHK.h5ad")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} h5ad files")
    return sorted(files)


def filter_h5ad_files_by_meta(h5ad_files: List[str], meta_df: pd.DataFrame) -> List[str]:
    """
    Keep only samples that appear in meta.csv.
    This prevents loading/examining samples that are fully excluded (0 spots).
    """
    allowed_samples = set(meta_df["sample_name"].astype(str).unique())
    kept = []
    for f in h5ad_files:
        sample = extract_sample_name_from_path(f)
        if sample in allowed_samples:
            kept.append(f)
    removed = len(h5ad_files) - len(kept)
    if removed:
        print(f"Filtered h5ad files by meta.csv sample_name: {len(h5ad_files)} -> {len(kept)} (removed {removed})")
    return kept


def extract_sample_name_from_path(file_path: str) -> str:
    """Extract sample name from file path.
    
    Examples:
        .../GSE181300/GSM5494475/st_norm_noHK.h5ad -> GSM5494475
        .../GSE208253/GSM6339631_s1/st_norm_noHK.h5ad -> GSM6339631_s1
    """
    parts = Path(file_path).parts
    # Find the second-to-last part (the sample directory name)
    if len(parts) >= 2:
        return parts[-2]
    raise ValueError(f"Cannot extract sample name from path: {file_path}")


def get_shared_genes(h5ad_files: List[str]) -> List[str]:
    """Find genes that are present in ALL h5ad files."""
    print("\n=== Step 1: Finding shared genes across all samples ===")
    
    all_gene_sets = []
    for h5ad_file in tqdm(h5ad_files, desc="Loading gene sets"):
        try:
            adata = sc.read_h5ad(h5ad_file)
            gene_set = set(adata.var_names)
            all_gene_sets.append(gene_set)
            print(f"  {Path(h5ad_file).parts[-2]}: {len(gene_set)} genes")
        except Exception as e:
            print(f"ERROR loading {h5ad_file}: {e}")
            raise
    
    # Compute intersection
    shared_genes = set(all_gene_sets[0])
    for gene_set in all_gene_sets[1:]:
        shared_genes = shared_genes.intersection(gene_set)
    
    shared_genes_list = sorted(list(shared_genes))
    print(f"\nShared genes across all {len(h5ad_files)} samples: {len(shared_genes_list)}")
    
    return shared_genes_list


def count_train_spots(h5ad_files: List[str], meta_df: pd.DataFrame, 
                     shared_genes: List[str]) -> Tuple[int, dict]:
    """Count how many spots will be included and build mapping.
    
    Returns:
        total_spots: Total number of spots to include
        sample_spot_mapping: Dict mapping sample_name -> list of (barcode, img_idx) tuples
    """
    print("\n=== Step 2: Counting train spots (first pass) ===")
    
    train_spots_set = set(meta_df["img_idx"].values)
    sample_spot_mapping = {}
    total_spots = 0
    
    shared_genes_set = set(shared_genes)
    
    for h5ad_file in tqdm(h5ad_files, desc="Counting spots"):
        sample_name = extract_sample_name_from_path(h5ad_file)
        adata = sc.read_h5ad(h5ad_file)
        
        # Check if all shared genes are present
        adata_genes = set(adata.var_names)
        np.save("combined_genes.npy", adata_genes)
        missing_genes = shared_genes_set - adata_genes
        if missing_genes:
            print(f"WARNING: {sample_name} missing {len(missing_genes)} shared genes")
        
        # Filter spots: only include those whose img_idx is in meta.csv
        matching_spots = []
        for barcode in adata.obs_names:
            img_idx = f"{sample_name}_{barcode}_hires"
            if img_idx in train_spots_set:
                matching_spots.append((barcode, img_idx))
        
        sample_spot_mapping[sample_name] = matching_spots
        total_spots += len(matching_spots)
        print(f"  {sample_name}: {len(matching_spots)} spots from {len(adata.obs_names)} total")
    
    print(f"\nTotal train spots to include: {total_spots}")
    return total_spots, sample_spot_mapping


def build_combined_matrix(h5ad_files: List[str], sample_spot_mapping: dict,
                         shared_genes: List[str], output_dir: Path) -> None:
    """Build the combined expression matrix (second pass)."""
    print("\n=== Step 3: Building combined expression matrix (second pass) ===")
    
    # Count total spots
    total_spots = sum(len(spots) for spots in sample_spot_mapping.values())
    n_genes = len(shared_genes)
    
    # Create output dir early so we can write progressively
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the matrix directly as a .npy memmap to avoid huge in-memory arrays
    matrix_path = output_dir / "combined_expression_matrix.npy"
    print(f"Creating memmap matrix: ({total_spots}, {n_genes}) -> {matrix_path}")
    combined_matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(total_spots, n_genes)
    )
    combined_obs: list[str] = []

    shared_genes_set = set(shared_genes)

    row_idx = 0
    for h5ad_file in tqdm(h5ad_files, desc="Loading expression data"):
        sample_name = extract_sample_name_from_path(h5ad_file)
        matching_spots = sample_spot_mapping.get(sample_name, [])
        if len(matching_spots) == 0:
            continue

        # Load once per sample
        adata = sc.read_h5ad(h5ad_file)

        # Validate genes exist (should, due to shared_genes)
        missing = shared_genes_set - set(adata.var_names)
        if missing:
            raise ValueError(f"{sample_name} missing {len(missing)} shared genes (unexpected).")

        # Block load: subset to barcodes + shared genes in desired order
        barcodes = [b for b, _ in matching_spots]
        img_idx_list = [img for _, img in matching_spots]
        ad_sub = adata[barcodes, shared_genes]
        X = ad_sub.X
        X_block = X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)

        n = X_block.shape[0]
        combined_matrix[row_idx:row_idx + n, :] = X_block
        combined_obs.extend(img_idx_list)
        row_idx += n
    
    print(f"\nFinal matrix shape: ({total_spots}, {n_genes})")
    print(f"Total spots collected: {len(combined_obs)}")
    
    # Assertions
    assert row_idx == total_spots, f"Row count mismatch: {row_idx} != {total_spots}"
    assert len(combined_obs) == total_spots, f"Obs count mismatch: {len(combined_obs)} != {total_spots}"
    
    # Save outputs (obs + genes). Matrix already written.
    print("\n=== Step 4: Saving output files ===")
    print(f"Saved: {matrix_path} ({total_spots}, {n_genes}, dtype=float32)")
    
    # Save combined_obs.npy
    obs_array = np.array(combined_obs, dtype=object)
    obs_path = output_dir / "combined_obs.npy"
    np.save(obs_path, obs_array)
    print(f"Saved: {obs_path} ({len(combined_obs)} spots)")

    # Save combined_obs_fixed.npy (string array for portability / no allow_pickle)
    obs_fixed = np.array(combined_obs, dtype=np.str_)
    obs_fixed_path = output_dir / "combined_obs_fixed.npy"
    np.save(obs_fixed_path, obs_fixed)
    print(f"Saved: {obs_fixed_path} ({len(obs_fixed)} spots)")

    # Save all_shared_genes.txt
    genes_path = output_dir / "all_shared_genes.txt"
    with open(genes_path, 'w') as f:
        f.write('\n'.join(shared_genes))
    print(f"Saved: {genes_path} ({len(shared_genes)} genes)")
    
    print("\n✅ All files saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Build Loki PredEx training data from HNSCC ST samples"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/students/hbae/data/Processed_Data",
        help="Root directory containing processed_loki subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/students/hbae/data/Processed_Data/training_data",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--meta_csv",
        type=str,
        default="/home/students/hbae/data/Processed_Data/training_data/meta.csv",
        help="Path to meta.csv file"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    meta_csv_path = Path(args.meta_csv)
    
    print("=" * 70)
    print("Loki PredEx Training Data Builder")
    print("=" * 70)
    print(f"Root directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Meta CSV: {meta_csv_path}")
    print()
    
    # Load meta.csv
    print("Loading meta.csv...")
    meta_df = pd.read_csv(meta_csv_path)
    print(f"Loaded {len(meta_df)} entries from meta.csv")
    print(f"Unique img_idx: {meta_df['img_idx'].nunique()}")
    
    # Find all h5ad files
    h5ad_files = find_all_h5ad_files(root_dir)
    if len(h5ad_files) == 0:
        raise ValueError(f"No h5ad files found in {root_dir}")
    h5ad_files = filter_h5ad_files_by_meta(h5ad_files, meta_df)
    
    # Step 1: Find shared genes
    shared_genes = get_shared_genes(h5ad_files)
    
    # Step 2: Count train spots (first pass)
    total_spots, sample_spot_mapping = count_train_spots(
        h5ad_files, meta_df, shared_genes
    )
    
    if total_spots == 0:
        raise ValueError("No matching spots found! Check img_idx format in meta.csv")
    
    # Step 3: Build combined matrix (second pass)
    build_combined_matrix(h5ad_files, sample_spot_mapping, shared_genes, output_dir)
    
    # Step 4: Save train_df.csv (copy of meta.csv with img_idx as index)
    print("\n=== Step 5: Saving train_df.csv ===")
    train_df = meta_df.copy()
    train_df = train_df.set_index('img_idx')
    train_df_path = output_dir / "train_df.csv"
    train_df.to_csv(train_df_path)
    print(f"Saved: {train_df_path} ({len(train_df)} entries)")
    
    print("\n" + "=" * 70)
    print("✅ Loki PredEx training data construction complete!")
    print("=" * 70)
    print(f"\nOutput files in {output_dir}:")
    print("  - combined_expression_matrix.npy")
    print("  - combined_obs.npy")
    print("  - combined_obs_fixed.npy")
    print("  - all_shared_genes.txt")
    print("  - train_df.csv")


if __name__ == "__main__":
    main()

