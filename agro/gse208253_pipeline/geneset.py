"""
Module for creating common gene sets across samples
and combining expression matrices
"""

import numpy as np
import pandas as pd
import anndata
from typing import List, Dict, Set
from pathlib import Path


def find_common_genes(adata_dict: Dict[str, anndata.AnnData]) -> List[str]:
    """
    Find common genes across all samples (intersection)
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData
        
    Returns:
        common_genes: List of common gene names (sorted)
    """
    if not adata_dict:
        return []
    
    # Start with first sample's genes
    gene_sets = [set(adata.var_names) for adata in adata_dict.values()]
    common_genes = set.intersection(*gene_sets)
    
    # Sort for consistency
    common_genes = sorted(list(common_genes))
    
    print(f"Found {len(common_genes)} common genes across {len(adata_dict)} samples")
    
    return common_genes


def subset_to_common_genes(
    adata_dict: Dict[str, anndata.AnnData],
    common_genes: List[str]
) -> Dict[str, anndata.AnnData]:
    """
    Subset all samples to common genes in consistent order
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData
        common_genes: List of common genes
        
    Returns:
        subset_dict: Dictionary with subsetted AnnData objects
    """
    subset_dict = {}
    
    for sample_id, adata in adata_dict.items():
        # Subset and reorder genes
        adata_subset = adata[:, common_genes].copy()
        subset_dict[sample_id] = adata_subset
    
    return subset_dict


def combine_expression_matrices(
    adata_dict: Dict[str, anndata.AnnData],
    layer: str = None
) -> np.ndarray:
    """
    Combine expression matrices from multiple samples vertically
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData (must have same genes in same order)
        layer: Which layer to use (None for .X)
        
    Returns:
        combined_matrix: Combined expression matrix (n_total_spots, n_genes)
    """
    matrices = []
    
    for sample_id, adata in adata_dict.items():
        if layer is None:
            mat = adata.X
        else:
            mat = adata.layers[layer]
        
        # Convert to dense if sparse
        if hasattr(mat, 'toarray'):
            mat = mat.toarray()
        
        matrices.append(mat)
    
    # Vertical stack
    combined_matrix = np.vstack(matrices)
    
    print(f"Combined matrix shape: {combined_matrix.shape}")
    
    return combined_matrix


def create_combined_obs(
    adata_dict: Dict[str, anndata.AnnData]
) -> pd.DataFrame:
    """
    Create combined observation metadata
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData
        
    Returns:
        combined_obs: Combined observation DataFrame
    """
    obs_list = []
    
    for sample_id, adata in adata_dict.items():
        obs_df = adata.obs.copy()
        obs_df['sample_id'] = sample_id
        obs_list.append(obs_df)
    
    combined_obs = pd.concat(obs_list, axis=0)
    
    return combined_obs


def save_geneset_and_expression(
    common_genes: List[str],
    combined_matrix: np.ndarray,
    output_dir: Path,
    combined_obs: pd.DataFrame = None
) -> None:
    """
    Save common gene set and combined expression matrix
    
    Args:
        common_genes: List of common genes
        combined_matrix: Combined expression matrix
        output_dir: Output directory
        combined_obs: Optional observation metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save gene list
    gene_file = output_dir / 'all_shared_genes.txt'
    with open(gene_file, 'w') as f:
        for gene in common_genes:
            f.write(f"{gene}\n")
    
    print(f"Saved {len(common_genes)} genes to {gene_file}")
    
    # Save expression matrix
    expr_file = output_dir / 'combined_expression.npy'
    np.save(expr_file, combined_matrix)
    
    print(f"Saved combined expression matrix {combined_matrix.shape} to {expr_file}")
    
    # Optionally save obs metadata
    if combined_obs is not None:
        obs_file = output_dir / 'combined_obs.csv'
        combined_obs.to_csv(obs_file)
        print(f"Saved combined observation metadata to {obs_file}")


def verify_consistency(
    common_genes: List[str],
    combined_matrix: np.ndarray
) -> bool:
    """
    Verify that gene list and expression matrix are consistent
    
    Args:
        common_genes: List of genes
        combined_matrix: Expression matrix
        
    Returns:
        is_consistent: Whether dimensions match
    """
    n_genes_list = len(common_genes)
    n_genes_matrix = combined_matrix.shape[1]
    
    if n_genes_list != n_genes_matrix:
        print(f"ERROR: Gene list has {n_genes_list} genes but matrix has {n_genes_matrix} columns!")
        return False
    
    print(f"✓ Consistency check passed: {n_genes_list} genes")
    return True


def process_all_samples(
    adata_dict: Dict[str, anndata.AnnData],
    output_dir: Path,
    layer: str = None
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    Complete pipeline: find common genes, subset, combine, and save
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData
        output_dir: Output directory
        layer: Which layer to use for expression
        
    Returns:
        common_genes: List of common genes
        combined_matrix: Combined expression matrix
        combined_obs: Combined observation metadata
    """
    from typing import Tuple
    
    # Find common genes
    common_genes = find_common_genes(adata_dict)
    
    # Subset to common genes
    adata_subset = subset_to_common_genes(adata_dict, common_genes)
    
    # Combine expression matrices
    combined_matrix = combine_expression_matrices(adata_subset, layer=layer)
    
    # Create combined obs
    combined_obs = create_combined_obs(adata_subset)
    
    # Verify consistency
    verify_consistency(common_genes, combined_matrix)
    
    # Save outputs
    save_geneset_and_expression(common_genes, combined_matrix, output_dir, combined_obs)
    
    return common_genes, combined_matrix, combined_obs



