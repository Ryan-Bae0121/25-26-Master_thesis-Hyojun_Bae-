"""
Module for expression normalization and gene selection
Uses Scanpy standard pipeline: normalize_total -> log1p -> HVG selection
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from typing import List, Optional, Set
from pathlib import Path


# Housekeeping genes to remove (common contaminants)
HOUSEKEEPING_GENES = {
    'ACTB', 'GAPDH', 'B2M', 'PPIA', 'RPLP0', 'RPL13A', 'HPRT1',
    'TBP', 'UBC', 'PGK1', 'GUSB', 'TFRC', 'HMBS', 'YWHAZ',
    # Add common ribosomal and mitochondrial patterns will be handled separately
}


def ensembl_to_symbol(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Convert Ensembl IDs to Gene Symbols if available
    
    Args:
        adata: AnnData with potentially Ensembl gene names
        
    Returns:
        adata: AnnData with gene symbols
    """
    # Check if we have gene_symbols in var
    if 'gene_symbols' in adata.var.columns:
        # Use gene symbols, fall back to original ID if missing
        adata.var['original_id'] = adata.var_names
        adata.var_names = adata.var['gene_symbols'].fillna(adata.var_names)
        adata.var_names_make_unique()
    elif 'gene_ids' in adata.var.columns and not adata.var_names[0].startswith('ENS'):
        # Already symbols, keep as is
        pass
    else:
        # Try to detect if we have Ensembl IDs (start with 'ENS')
        if adata.var_names[0].startswith('ENS'):
            print(f"Warning: Ensembl IDs detected but no gene_symbols column. Keeping IDs.")
    
    return adata


def filter_housekeeping_genes(
    adata: anndata.AnnData,
    housekeeping_set: Set[str] = HOUSEKEEPING_GENES,
    remove_mt: bool = True,
    remove_ribo: bool = False
) -> anndata.AnnData:
    """
    Remove housekeeping and unwanted genes
    
    Args:
        adata: AnnData object
        housekeeping_set: Set of housekeeping gene symbols
        remove_mt: Remove mitochondrial genes (MT-)
        remove_ribo: Remove ribosomal genes (RPL*, RPS*)
        
    Returns:
        adata: Filtered AnnData
    """
    genes_to_keep = []
    
    for gene in adata.var_names:
        # Check housekeeping
        if gene in housekeeping_set:
            continue
        
        # Check mitochondrial
        if remove_mt and gene.startswith('MT-'):
            continue
        
        # Check ribosomal (optional)
        if remove_ribo and (gene.startswith('RPL') or gene.startswith('RPS')):
            continue
        
        genes_to_keep.append(gene)
    
    n_removed = adata.n_vars - len(genes_to_keep)
    if n_removed > 0:
        print(f"Removed {n_removed} housekeeping/unwanted genes")
    
    return adata[:, genes_to_keep].copy()


def normalize_sample(
    adata: anndata.AnnData,
    target_sum: float = 1e4,
    n_top_genes: int = 1000,
    remove_housekeeping: bool = True,
    log_transform: bool = True
) -> anndata.AnnData:
    """
    Normalize expression using Scanpy standard pipeline
    
    Args:
        adata: Raw count AnnData
        target_sum: Target sum for normalization (default 10000)
        n_top_genes: Number of highly variable genes to select
        remove_housekeeping: Whether to remove housekeeping genes
        log_transform: Whether to apply log1p transformation
        
    Returns:
        adata: Normalized AnnData
    """
    adata = adata.copy()
    
    # Convert IDs to symbols
    adata = ensembl_to_symbol(adata)
    
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()
    
    # Normalize total counts per spot
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Log transform
    if log_transform:
        sc.pp.log1p(adata)
        adata.layers['log1p'] = adata.X.copy()
    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_top_genes, 
        flavor='seurat_v3',
        layer='counts'
    )
    
    # Remove housekeeping genes
    if remove_housekeeping:
        adata = filter_housekeeping_genes(adata, remove_mt=True, remove_ribo=False)
    
    return adata


def select_hvg(
    adata: anndata.AnnData,
    n_top_genes: Optional[int] = None
) -> anndata.AnnData:
    """
    Select highly variable genes
    
    Args:
        adata: AnnData with HVG annotation
        n_top_genes: Number of genes to select (if None, use all marked HVG)
        
    Returns:
        adata: AnnData with only HVG
    """
    if 'highly_variable' not in adata.var.columns:
        raise ValueError("HVG not computed. Run normalize_sample first.")
    
    if n_top_genes is not None:
        # Select top N genes by variance
        var_scores = adata.var['highly_variable_rank'].copy()
        var_scores[~adata.var['highly_variable']] = np.inf
        top_genes = var_scores.nsmallest(n_top_genes).index
        adata_hvg = adata[:, top_genes].copy()
    else:
        adata_hvg = adata[:, adata.var['highly_variable']].copy()
    
    return adata_hvg


def batch_normalize_samples(
    samples_dict: dict,
    target_sum: float = 1e4,
    n_top_genes: int = 1000,
    remove_housekeeping: bool = True
) -> dict:
    """
    Normalize multiple samples
    
    Args:
        samples_dict: Dictionary of sample_id -> AnnData
        target_sum: Target sum for normalization
        n_top_genes: Number of HVG to identify
        remove_housekeeping: Remove housekeeping genes
        
    Returns:
        normalized_dict: Dictionary of normalized AnnData objects
    """
    normalized_dict = {}
    
    for sample_id, adata in samples_dict.items():
        print(f"Normalizing {sample_id}...")
        adata_norm = normalize_sample(
            adata, 
            target_sum=target_sum,
            n_top_genes=n_top_genes,
            remove_housekeeping=remove_housekeeping
        )
        normalized_dict[sample_id] = adata_norm
    
    return normalized_dict


def save_normalized_sample(
    adata: anndata.AnnData,
    output_path: Path,
    compression: str = 'gzip'
) -> None:
    """
    Save normalized AnnData to h5ad
    
    Args:
        adata: Normalized AnnData
        output_path: Output file path
        compression: Compression method
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression=compression)



