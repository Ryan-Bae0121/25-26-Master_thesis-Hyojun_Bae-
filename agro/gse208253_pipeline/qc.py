"""
Quality control module for Visium samples
Filters samples and spots based on resolution and gene detection
"""

import numpy as np
import pandas as pd
import anndata
from typing import Tuple, Dict, List
from pathlib import Path


def check_image_resolution(image: np.ndarray, min_size: int = 2000) -> Tuple[bool, str]:
    """
    Check if image meets minimum resolution requirement
    
    Args:
        image: Image array (H, W, C)
        min_size: Minimum size for both dimensions
        
    Returns:
        passed: Whether image passes QC
        reason: Reason for pass/fail
    """
    h, w = image.shape[:2]
    
    if h < min_size or w < min_size:
        return False, f"Resolution {h}x{w} < {min_size}x{min_size}"
    
    return True, f"Resolution {h}x{w} OK"


def filter_spots_by_genes(
    adata: anndata.AnnData,
    min_genes: int = 200,
    in_tissue_only: bool = True
) -> Tuple[anndata.AnnData, Dict]:
    """
    Filter spots based on gene detection and tissue location
    
    Args:
        adata: AnnData object
        min_genes: Minimum number of genes detected per spot
        in_tissue_only: Only keep spots marked as in_tissue
        
    Returns:
        adata_filtered: Filtered AnnData
        stats: Dictionary with filtering statistics
    """
    n_spots_initial = adata.n_obs
    
    # Calculate genes per spot (n_genes_by_counts)
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if hasattr(adata.X, 'A1') else (adata.X > 0).sum(axis=1)
    adata.obs['total_counts'] = adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else adata.X.sum(axis=1)
    
    # Create filter mask
    mask = adata.obs['n_genes'] > min_genes
    
    if in_tissue_only and 'in_tissue' in adata.obs.columns:
        mask = mask & (adata.obs['in_tissue'] == 1)
    
    # Apply filter
    adata_filtered = adata[mask].copy()
    
    # Calculate statistics
    n_spots_final = adata_filtered.n_obs
    n_spots_removed = n_spots_initial - n_spots_final
    
    stats = {
        'n_spots_initial': n_spots_initial,
        'n_spots_final': n_spots_final,
        'n_spots_removed': n_spots_removed,
        'pass_rate': n_spots_final / n_spots_initial if n_spots_initial > 0 else 0,
        'median_umi': np.median(adata_filtered.obs['total_counts']) if n_spots_final > 0 else 0,
        'median_genes': np.median(adata_filtered.obs['n_genes']) if n_spots_final > 0 else 0,
    }
    
    return adata_filtered, stats


def qc_sample(
    adata: anndata.AnnData,
    image: np.ndarray,
    sample_id: str,
    min_image_size: int = 2000,
    min_genes: int = 200,
    in_tissue_only: bool = True
) -> Tuple[anndata.AnnData, Dict, bool]:
    """
    Perform complete QC on a sample
    
    Args:
        adata: AnnData object
        image: High-resolution image
        sample_id: Sample identifier
        min_image_size: Minimum image resolution
        min_genes: Minimum genes per spot
        in_tissue_only: Filter for in_tissue spots
        
    Returns:
        adata_qc: QC-passed AnnData (or None if failed)
        qc_report: Dictionary with QC results
        passed: Whether sample passed QC
    """
    qc_report = {
        'sample_id': sample_id,
        'image_resolution': f"{image.shape[0]}x{image.shape[1]}",
    }
    
    # Check image resolution
    img_pass, img_reason = check_image_resolution(image, min_image_size)
    qc_report['image_qc_passed'] = img_pass
    qc_report['image_qc_reason'] = img_reason
    
    if not img_pass:
        qc_report['overall_passed'] = False
        qc_report['exclusion_reason'] = img_reason
        return None, qc_report, False
    
    # Filter spots
    adata_qc, spot_stats = filter_spots_by_genes(adata, min_genes, in_tissue_only)
    qc_report.update(spot_stats)
    
    # Check if we have enough spots
    if adata_qc.n_obs == 0:
        qc_report['overall_passed'] = False
        qc_report['exclusion_reason'] = "No spots passed filtering"
        return None, qc_report, False
    
    qc_report['overall_passed'] = True
    qc_report['exclusion_reason'] = None
    
    return adata_qc, qc_report, True


def batch_qc_samples(
    sample_ids: List[str],
    raw_root: str,
    min_image_size: int = 2000,
    min_genes: int = 200,
    in_tissue_only: bool = True
) -> Tuple[Dict[str, anndata.AnnData], pd.DataFrame]:
    """
    Run QC on multiple samples
    
    Args:
        sample_ids: List of sample identifiers
        raw_root: Root directory with raw data
        min_image_size: Minimum image resolution
        min_genes: Minimum genes per spot
        in_tissue_only: Filter for in_tissue spots
        
    Returns:
        passed_samples: Dictionary of sample_id -> AnnData for passed samples
        qc_summary: DataFrame with QC summary for all samples
    """
    from . import io_visium
    
    passed_samples = {}
    qc_reports = []
    
    for sample_id in sample_ids:
        try:
            # Load sample
            adata = io_visium.load_complete_sample(sample_id, raw_root, prefix=sample_id)
            image = adata.uns['spatial']['hires_image']
            
            # Run QC
            adata_qc, qc_report, passed = qc_sample(
                adata, image, sample_id, 
                min_image_size, min_genes, in_tissue_only
            )
            
            if passed:
                passed_samples[sample_id] = adata_qc
            
            qc_reports.append(qc_report)
            
        except Exception as e:
            qc_reports.append({
                'sample_id': sample_id,
                'overall_passed': False,
                'exclusion_reason': f"Error loading: {str(e)}"
            })
    
    qc_summary = pd.DataFrame(qc_reports)
    
    return passed_samples, qc_summary



