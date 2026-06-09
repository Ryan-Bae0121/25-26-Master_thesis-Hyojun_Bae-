#!/usr/bin/env python3
"""
Extract tiles from HDF5 file to PNG images for zero-shot pipeline.
"""

import argparse
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def extract_tiles(hdf5_path, out_dir, max_tiles=None):
    """Extract tiles from HDF5 to PNG files"""
    
    hdf5_path = Path(hdf5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Opening HDF5: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check available keys
        print(f"   Keys: {list(f.keys())}")
        
        # Try different possible keys
        if 'img' in f:
            images = f['img']
        elif 'images' in f:
            images = f['images']
        elif 'tiles' in f:
            images = f['tiles']
        else:
            raise KeyError(f"No image data found. Available keys: {list(f.keys())}")
        
        print(f"   ✅ Found images: shape={images.shape}, dtype={images.dtype}")
        
        # Get coordinates if available
        coords = None
        if 'coords' in f:
            coords = f['coords'][:]
            print(f"   ✅ Found coords: shape={coords.shape}")
        
        # Determine number of tiles to extract
        n_tiles = len(images)
        if max_tiles and max_tiles < n_tiles:
            n_tiles = max_tiles
            print(f"   ⚠️  Limiting to {max_tiles} tiles (out of {len(images)})")
        
        print(f"\n🖼️  Extracting {n_tiles} tiles to {out_dir}...")
        
        # Extract tiles
        for i in tqdm(range(n_tiles), desc="Extracting"):
            img_data = images[i]
            
            # Handle different data types
            if img_data.dtype == np.uint8:
                img_array = img_data
            elif img_data.dtype == np.float32 or img_data.dtype == np.float64:
                # Assume normalized to [0, 1]
                img_array = (img_data * 255).astype(np.uint8)
            else:
                # Try to convert
                img_array = img_data.astype(np.uint8)
            
            # Create PIL Image
            if img_array.ndim == 2:
                # Grayscale
                img = Image.fromarray(img_array, mode='L')
            elif img_array.shape[-1] == 3:
                # RGB
                img = Image.fromarray(img_array, mode='RGB')
            elif img_array.shape[0] == 3:
                # CHW format, convert to HWC
                img_array = np.transpose(img_array, (1, 2, 0))
                img = Image.fromarray(img_array, mode='RGB')
            else:
                print(f"   ⚠️  Unexpected shape for tile {i}: {img_array.shape}")
                continue
            
            # Generate filename
            if coords is not None and i < len(coords):
                x, y = coords[i]
                filename = f"tile_{i:06d}_x{int(x)}_y{int(y)}.png"
            else:
                filename = f"tile_{i:06d}.png"
            
            # Save
            img.save(out_dir / filename)
        
        print(f"\n✅ Extracted {n_tiles} tiles to {out_dir}")
        
        return n_tiles


def main():
    parser = argparse.ArgumentParser(description='Extract tiles from HDF5 to PNG')
    parser.add_argument('--hdf5', type=str, required=True,
                       help='Path to HDF5 file')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for PNG tiles')
    parser.add_argument('--max_tiles', type=int, default=None,
                       help='Maximum number of tiles to extract (default: all)')
    
    args = parser.parse_args()
    
    extract_tiles(args.hdf5, args.out_dir, args.max_tiles)


if __name__ == '__main__':
    main()

