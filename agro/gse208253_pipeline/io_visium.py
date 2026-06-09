"""
Module for loading Visium spatial transcriptomics data
Supports both h5 and mtx formats
"""

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, Tuple, Optional
import anndata


def load_visium_data(
    sample_id: str,
    raw_root: str,
    prefix: str = None
) -> Tuple[anndata.AnnData, Dict, pd.DataFrame]:
    """
    Load Visium data from flat file structure
    
    Args:
        sample_id: Sample identifier (e.g., 'GSM6339631_s1')
        raw_root: Root directory containing all files
        prefix: Optional prefix for file names (defaults to sample_id)
    
    Returns:
        adata: AnnData object with counts
        scalefactors: Dictionary with spatial scale factors
        positions: DataFrame with barcode positions
    """
    if prefix is None:
        prefix = sample_id
    
    raw_root = Path(raw_root)
    
    # Load count matrix
    h5_path = raw_root / f"{prefix}_filtered_feature_bc_matrix.h5"
    
    if h5_path.exists():
        adata = sc.read_10x_h5(h5_path)
    else:
        # Try mtx format
        mtx_dir = raw_root / f"{prefix}_counts"
        if mtx_dir.exists():
            adata = sc.read_10x_mtx(mtx_dir)
        else:
            raise FileNotFoundError(f"Count matrix not found for {sample_id}")
    
    # Make variable names unique
    adata.var_names_make_unique()
    
    # Load scalefactors
    scale_path = raw_root / f"{prefix}_scalefactors_json.json"
    with open(scale_path, 'r') as f:
        scalefactors = json.load(f)
    
    # Load tissue positions
    pos_path = raw_root / f"{prefix}_tissue_positions_list.csv"
    
    # Read without header (Visium format)
    positions = pd.read_csv(pos_path, header=None)
    
    # Assign column names based on number of columns
    if positions.shape[1] == 6:
        # Format: barcode, has_image, in_tissue, array_row, array_col, pxl_row, pxl_col
        positions.columns = ['barcode', 'has_image', 'in_tissue', 
                            'array_row', 'array_col', 'pxl_row_fullres', 'pxl_col_fullres']
    elif positions.shape[1] == 5:
        # Alternative format without has_image
        positions.columns = ['barcode', 'in_tissue', 'array_row', 
                            'array_col', 'pxl_row_fullres', 'pxl_col_fullres']
    else:
        raise ValueError(f"Unexpected position file format with {positions.shape[1]} columns")
    
    positions.set_index('barcode', inplace=True)
    
    return adata, scalefactors, positions


def load_hires_image(sample_id: str, raw_root: str, prefix: str = None) -> np.ndarray:
    """
    Load high-resolution tissue image
    
    Args:
        sample_id: Sample identifier
        raw_root: Root directory
        prefix: Optional prefix for file names
        
    Returns:
        image: RGB image as numpy array
    """
    from PIL import Image
    
    if prefix is None:
        prefix = sample_id
    
    raw_root = Path(raw_root)
    img_path = raw_root / f"{prefix}_tissue_hires_image.png"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = Image.open(img_path)
    return np.array(img)


def attach_spatial_info(
    adata: anndata.AnnData,
    positions: pd.DataFrame,
    scalefactors: Dict,
    image: np.ndarray
) -> anndata.AnnData:
    """
    Attach spatial information to AnnData object
    
    Args:
        adata: AnnData object
        positions: Position DataFrame
        scalefactors: Scale factors dictionary
        image: High-resolution image
        
    Returns:
        adata: AnnData with spatial info attached
    """
    # Filter positions to match adata barcodes
    common_barcodes = adata.obs_names.intersection(positions.index)
    adata = adata[common_barcodes].copy()
    positions = positions.loc[common_barcodes]
    
    # Add position info to obs
    adata.obs['in_tissue'] = positions['in_tissue'].values
    adata.obs['array_row'] = positions['array_row'].values
    adata.obs['array_col'] = positions['array_col'].values
    adata.obs['pxl_row_fullres'] = positions['pxl_row_fullres'].values
    adata.obs['pxl_col_fullres'] = positions['pxl_col_fullres'].values
    
    # Calculate hires coordinates
    scale = scalefactors['tissue_hires_scalef']
    adata.obs['pxl_row_hires'] = (positions['pxl_row_fullres'] * scale).values
    adata.obs['pxl_col_hires'] = (positions['pxl_col_fullres'] * scale).values
    
    # Store spatial data in uns
    adata.uns['spatial'] = {
        'scalefactors': scalefactors,
        'hires_image': image
    }
    
    return adata


def load_complete_sample(
    sample_id: str,
    raw_root: str,
    prefix: str = None
) -> anndata.AnnData:
    """
    Load complete Visium sample with all spatial information
    
    Args:
        sample_id: Sample identifier
        raw_root: Root directory
        prefix: Optional file prefix
        
    Returns:
        adata: Complete AnnData object with spatial info
    """
    adata, scalefactors, positions = load_visium_data(sample_id, raw_root, prefix)
    image = load_hires_image(sample_id, raw_root, prefix)
    adata = attach_spatial_info(adata, positions, scalefactors, image)
    adata.obs['sample_id'] = sample_id
    
    return adata



