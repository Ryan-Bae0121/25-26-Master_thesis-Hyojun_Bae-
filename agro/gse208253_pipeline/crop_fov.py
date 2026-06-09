"""
Module for cropping Field-of-View (FOV) patches around each spot
Uses Space Ranger coordinates and scale factors for accurate cropping
"""

import numpy as np
from PIL import Image
import anndata
from pathlib import Path
from typing import Tuple, Dict
import os


def calculate_fov_size(
    spot_diameter_fullres: float,
    tissue_hires_scalef: float,
    k_fov: float = 1.3
) -> Tuple[float, int]:
    """
    Calculate FOV size based on spot diameter
    
    Args:
        spot_diameter_fullres: Spot diameter in full resolution pixels
        tissue_hires_scalef: Scale factor for hires image
        k_fov: Multiplier for FOV size (default 1.3x spot diameter)
        
    Returns:
        d_hires: Spot diameter in hires pixels
        fov_size: FOV square side length in hires pixels
    """
    d_hires = spot_diameter_fullres * tissue_hires_scalef
    fov_size = int(np.round(k_fov * d_hires))
    
    return d_hires, fov_size


def crop_spot_fov(
    image: np.ndarray,
    center_row: float,
    center_col: float,
    fov_size: int,
    pad_mode: str = 'reflect'
) -> np.ndarray:
    """
    Crop a square FOV around a spot center with padding if needed
    
    Args:
        image: Full hires image (H, W, C)
        center_row: Row coordinate of spot center
        center_col: Column coordinate of spot center
        fov_size: Size of square FOV
        pad_mode: Padding mode for boundary spots ('reflect', 'edge', 'constant')
        
    Returns:
        fov_patch: Cropped FOV patch (fov_size, fov_size, C)
    """
    h, w = image.shape[:2]
    half_size = fov_size // 2
    
    # Calculate crop boundaries
    row_min = int(center_row - half_size)
    row_max = int(center_row + half_size)
    col_min = int(center_col - half_size)
    col_max = int(center_col + half_size)
    
    # Check if padding is needed
    need_padding = (row_min < 0 or row_max > h or col_min < 0 or col_max > w)
    
    if need_padding:
        # Calculate padding amounts
        pad_top = max(0, -row_min)
        pad_bottom = max(0, row_max - h)
        pad_left = max(0, -col_min)
        pad_right = max(0, col_max - w)
        
        # Pad image
        if len(image.shape) == 3:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        
        image_padded = np.pad(image, padding, mode=pad_mode)
        
        # Adjust coordinates for padded image
        row_min += pad_top
        row_max += pad_top
        col_min += pad_left
        col_max += pad_left
        
        fov_patch = image_padded[row_min:row_max, col_min:col_max]
    else:
        fov_patch = image[row_min:row_max, col_min:col_max]
    
    # Ensure correct size (may need to adjust by 1 pixel due to rounding)
    if fov_patch.shape[0] != fov_size or fov_patch.shape[1] != fov_size:
        fov_patch = fov_patch[:fov_size, :fov_size]
    
    return fov_patch


def resize_patch(patch: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resize patch to target size
    
    Args:
        patch: Input patch (H, W, C)
        target_size: Target size (will be square)
        
    Returns:
        resized: Resized patch (target_size, target_size, C)
    """
    pil_img = Image.fromarray(patch.astype(np.uint8))
    pil_resized = pil_img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(pil_resized)


def process_sample_patches(
    adata: anndata.AnnData,
    image: np.ndarray,
    scalefactors: Dict,
    out_dir: Path,
    sample_id: str,
    k_fov: float = 1.3,
    target_size: int = 224,
    pad_mode: str = 'reflect'
) -> Tuple[Dict, float, int]:
    """
    Process all spots in a sample and save FOV patches
    
    Args:
        adata: AnnData with spatial info
        image: Hires image
        scalefactors: Scale factors dictionary
        out_dir: Output directory for patches
        sample_id: Sample identifier
        k_fov: FOV size multiplier
        target_size: Target patch size
        pad_mode: Padding mode
        
    Returns:
        patch_paths: Dictionary mapping barcode to patch file path
        d_hires: Spot diameter in hires pixels
        fov_size: FOV size in hires pixels
    """
    # Create output directory
    sample_dir = out_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate FOV size
    d_hires, fov_size = calculate_fov_size(
        scalefactors['spot_diameter_fullres'],
        scalefactors['tissue_hires_scalef'],
        k_fov
    )
    
    patch_paths = {}
    
    # Process each spot
    for barcode in adata.obs_names:
        center_row = adata.obs.loc[barcode, 'pxl_row_hires']
        center_col = adata.obs.loc[barcode, 'pxl_col_hires']
        
        # Crop FOV
        fov_patch = crop_spot_fov(image, center_row, center_col, fov_size, pad_mode)
        
        # Resize to target size
        patch_resized = resize_patch(fov_patch, target_size)
        
        # Save patch
        patch_filename = f"{barcode}.png"
        patch_path = sample_dir / patch_filename
        Image.fromarray(patch_resized.astype(np.uint8)).save(patch_path)
        
        # Store relative path
        patch_paths[barcode] = str(patch_path)
    
    return patch_paths, d_hires, fov_size


def add_patch_info_to_adata(
    adata: anndata.AnnData,
    patch_paths: Dict,
    d_hires: float,
    fov_size: int
) -> anndata.AnnData:
    """
    Add patch information to AnnData obs
    
    Args:
        adata: AnnData object
        patch_paths: Dictionary of barcode -> patch path
        d_hires: Spot diameter in hires pixels
        fov_size: FOV size in hires pixels
        
    Returns:
        adata: Updated AnnData
    """
    adata.obs['patch_path'] = [patch_paths.get(bc, '') for bc in adata.obs_names]
    adata.uns['fov_info'] = {
        'd_hires': d_hires,
        'fov_size': fov_size
    }
    
    return adata



