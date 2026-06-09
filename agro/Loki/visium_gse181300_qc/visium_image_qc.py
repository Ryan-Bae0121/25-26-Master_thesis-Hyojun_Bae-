#!/usr/bin/env python3
"""
Visium Image Quality Control Script
Performs comprehensive QC on a single Visium sample's H&E image and spatial data.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, disk
import warnings
warnings.filterwarnings('ignore')


class VisiumImageQC:
    def __init__(self, sample_dir, out_dir, positions_csv='auto', 
                 use_lowres_if_missing=True, tile_size=256, seed=42):
        self.sample_dir = Path(sample_dir)
        self.out_dir = Path(out_dir)
        self.positions_csv = positions_csv
        self.use_lowres_if_missing = use_lowres_if_missing
        self.tile_size = tile_size
        self.seed = seed
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
        
        self.metrics = {}
        self.warnings = {}
        
    def log(self, msg):
        print(f"[VisiumQC] {msg}")
        
    def load_image(self):
        """Load tissue image (hires or lowres fallback)"""
        spatial_dir = self.sample_dir / "spatial"
        
        # Try hires first
        hires_path = spatial_dir / "tissue_hires_image.png"
        lowres_path = spatial_dir / "tissue_lowres_image.png"
        
        if hires_path.exists():
            self.log(f"✅ Loading hires image: {hires_path}")
            self.image = cv2.imread(str(hires_path))
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_type = "hires"
            self.image_path = hires_path
        elif lowres_path.exists() and self.use_lowres_if_missing:
            self.log(f"⚠️  Hires not found, using lowres: {lowres_path}")
            self.image = cv2.imread(str(lowres_path))
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_type = "lowres"
            self.image_path = lowres_path
        else:
            raise FileNotFoundError(f"No tissue image found in {spatial_dir}")
            
        self.metrics['image_width'] = self.image.shape[1]
        self.metrics['image_height'] = self.image.shape[0]
        self.metrics['image_type'] = self.image_type
        self.metrics['file_size_bytes'] = self.image_path.stat().st_size
        self.metrics['file_size_mb'] = self.metrics['file_size_bytes'] / (1024 * 1024)
        
        self.log(f"   Image size: {self.image.shape[1]}×{self.image.shape[0]} px")
        self.log(f"   File size: {self.metrics['file_size_mb']:.2f} MB")
        
    def load_scalefactors(self):
        """Load scalefactors_json.json"""
        sf_path = self.sample_dir / "spatial" / "scalefactors_json.json"
        
        if not sf_path.exists():
            raise FileNotFoundError(f"scalefactors not found: {sf_path}")
            
        with open(sf_path, 'r') as f:
            self.scalefactors = json.load(f)
            
        # Get appropriate scale factor
        if self.image_type == "hires":
            self.scale_factor = self.scalefactors.get('tissue_hires_scalef', 1.0)
        else:
            self.scale_factor = self.scalefactors.get('tissue_lowres_scalef', 1.0)
            
        self.spot_diameter_fullres = self.scalefactors.get('spot_diameter_fullres', 89.0)
        
        self.log(f"✅ Scale factor: {self.scale_factor:.4f}")
        self.log(f"   Spot diameter (fullres): {self.spot_diameter_fullres:.1f} px")
        
    def load_positions(self):
        """Load tissue positions (auto-detect format)"""
        spatial_dir = self.sample_dir / "spatial"
        
        # Try different position file names
        candidates = [
            spatial_dir / "tissue_positions.csv",
            spatial_dir / "tissue_positions_list.csv"
        ]
        
        pos_path = None
        for cand in candidates:
            if cand.exists():
                pos_path = cand
                break
                
        if pos_path is None:
            raise FileNotFoundError(f"No tissue_positions file found in {spatial_dir}")
            
        self.log(f"✅ Loading positions: {pos_path.name}")
        
        # Try to read with header first
        try:
            df = pd.read_csv(pos_path)
            if 'barcode' in df.columns:
                # New format with header
                self.positions = df
            else:
                # Old format without header
                df = pd.read_csv(pos_path, header=None)
                df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 
                             'pxl_row_in_fullres', 'pxl_col_in_fullres']
                self.positions = df
        except:
            # Old format without header
            df = pd.read_csv(pos_path, header=None)
            df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 
                         'pxl_row_in_fullres', 'pxl_col_in_fullres']
            self.positions = df
            
        # Filter in-tissue spots
        self.in_tissue = self.positions[self.positions['in_tissue'] == 1].copy()
        
        # Convert to image coordinates
        self.in_tissue['img_x'] = self.in_tissue['pxl_col_in_fullres'] * self.scale_factor
        self.in_tissue['img_y'] = self.in_tissue['pxl_row_in_fullres'] * self.scale_factor
        
        self.metrics['total_spots'] = len(self.positions)
        self.metrics['in_tissue_spots'] = len(self.in_tissue)
        
        self.log(f"   Total spots: {self.metrics['total_spots']}")
        self.log(f"   In-tissue spots: {self.metrics['in_tissue_spots']}")
        
    def compute_tissue_mask(self):
        """Compute tissue mask using Otsu thresholding"""
        self.log("🔍 Computing tissue mask...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Otsu threshold
        try:
            thresh = threshold_otsu(gray)
            mask = gray < thresh  # Tissue is darker
        except:
            # Fallback: simple threshold
            mask = gray < 200
            
        # Morphological operations
        mask = binary_opening(mask, disk(5))
        mask = binary_closing(mask, disk(10))
        
        self.tissue_mask = mask.astype(np.uint8)
        
        # Compute tissue coverage
        tissue_pixels = np.sum(self.tissue_mask)
        total_pixels = self.tissue_mask.size
        self.metrics['tissue_coverage'] = tissue_pixels / total_pixels
        
        self.log(f"   Tissue coverage: {self.metrics['tissue_coverage']*100:.2f}%")
        
        # Save tissue mask
        mask_vis = (self.tissue_mask * 255).astype(np.uint8)
        cv2.imwrite(str(self.out_dir / "tissue_mask.png"), mask_vis)
        
    def compute_blur_metrics(self):
        """Compute blur/focus metrics"""
        self.log("🔍 Computing blur metrics...")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Global Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        vlp_global = laplacian.var()
        self.metrics['vlp_global'] = float(vlp_global)
        
        # Global Tenengrad (Sobel-based)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobelx**2 + sobely**2)
        self.metrics['tenengrad_global'] = float(tenengrad)
        
        self.log(f"   Global VLP: {vlp_global:.2f}")
        self.log(f"   Global Tenengrad: {tenengrad:.2f}")
        
        # Per-tile analysis
        h, w = gray.shape
        tile_vlps = []
        
        blur_map = np.zeros((h // self.tile_size + 1, w // self.tile_size + 1))
        
        for i, y in enumerate(range(0, h, self.tile_size)):
            for j, x in enumerate(range(0, w, self.tile_size)):
                tile = gray[y:y+self.tile_size, x:x+self.tile_size]
                if tile.size > 0:
                    lap = cv2.Laplacian(tile, cv2.CV_64F)
                    vlp = lap.var()
                    tile_vlps.append(vlp)
                    blur_map[i, j] = vlp
                    
        tile_vlps = np.array(tile_vlps)
        median_vlp = np.median(tile_vlps)
        low_focus_tiles = np.sum(tile_vlps < median_vlp * 0.5)
        low_focus_frac = low_focus_tiles / len(tile_vlps)
        
        self.metrics['tile_vlp_median'] = float(median_vlp)
        self.metrics['low_focus_tile_count'] = int(low_focus_tiles)
        self.metrics['low_focus_tile_frac'] = float(low_focus_frac)
        
        self.log(f"   Low-focus tile fraction: {low_focus_frac*100:.2f}%")
        
        # Save blur map
        plt.figure(figsize=(10, 8))
        plt.imshow(blur_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Variance of Laplacian')
        plt.title('Blur Map (Per-tile VLP)')
        plt.tight_layout()
        plt.savefig(self.out_dir / "blur_map.png", dpi=150)
        plt.close()
        
        # Histograms
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Brightness histogram
        axes[0].hist(gray.ravel(), bins=50, color='gray', alpha=0.7)
        axes[0].set_xlabel('Pixel Intensity')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Brightness Distribution')
        axes[0].grid(alpha=0.3)
        
        # Laplacian histogram
        axes[1].hist(tile_vlps, bins=30, color='blue', alpha=0.7)
        axes[1].axvline(median_vlp, color='red', linestyle='--', label=f'Median: {median_vlp:.1f}')
        axes[1].axvline(median_vlp * 0.5, color='orange', linestyle='--', label='Low-focus threshold')
        axes[1].set_xlabel('Variance of Laplacian')
        axes[1].set_ylabel('Tile Count')
        axes[1].set_title('Focus Quality Distribution')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.out_dir / "hist_brightness.png", dpi=150)
        plt.close()
        
    def compute_spot_alignment(self):
        """Check spot alignment with tissue"""
        self.log("🔍 Checking spot alignment...")
        
        h, w = self.image.shape[:2]
        
        # Check spots within image bounds
        valid_x = (self.in_tissue['img_x'] >= 0) & (self.in_tissue['img_x'] < w)
        valid_y = (self.in_tissue['img_y'] >= 0) & (self.in_tissue['img_y'] < h)
        valid_spots = valid_x & valid_y
        
        out_of_bounds = (~valid_spots).sum()
        self.metrics['spots_out_of_bounds'] = int(out_of_bounds)
        
        if out_of_bounds > 0:
            self.log(f"   ⚠️  {out_of_bounds} spots out of image bounds")
            
        # Clip to valid range
        valid_tissue = self.in_tissue[valid_spots].copy()
        
        # Check spot-on-tissue
        spots_on_tissue = 0
        for _, row in valid_tissue.iterrows():
            x = int(np.clip(row['img_x'], 0, w-1))
            y = int(np.clip(row['img_y'], 0, h-1))
            if self.tissue_mask[y, x] > 0:
                spots_on_tissue += 1
                
        spot_on_tissue_pct = spots_on_tissue / len(valid_tissue) if len(valid_tissue) > 0 else 0
        self.metrics['spots_on_tissue'] = int(spots_on_tissue)
        self.metrics['spot_on_tissue_pct'] = float(spot_on_tissue_pct)
        
        self.log(f"   Spot-on-tissue: {spot_on_tissue_pct*100:.2f}%")
        
    def generate_overlay(self):
        """Generate overlay image with spots"""
        self.log("🎨 Generating overlay image...")
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.image)
        
        # Draw spots
        spot_radius = (self.spot_diameter_fullres * self.scale_factor) / 2
        
        h, w = self.image.shape[:2]
        valid_x = (self.in_tissue['img_x'] >= 0) & (self.in_tissue['img_x'] < w)
        valid_y = (self.in_tissue['img_y'] >= 0) & (self.in_tissue['img_y'] < h)
        valid_spots = self.in_tissue[valid_x & valid_y]
        
        for _, row in valid_spots.iterrows():
            x = row['img_x']
            y = row['img_y']
            
            # Check if on tissue
            xi = int(np.clip(x, 0, w-1))
            yi = int(np.clip(y, 0, h-1))
            on_tissue = self.tissue_mask[yi, xi] > 0
            
            color = 'lime' if on_tissue else 'red'
            alpha = 0.3 if on_tissue else 0.5
            
            circle = Circle((x, y), spot_radius, color=color, alpha=alpha, linewidth=0.5)
            ax.add_patch(circle)
            
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis('off')
        ax.set_title(f'Spot Overlay (Green=On Tissue, Red=Off Tissue)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.out_dir / "overlay_spots.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"   ✅ Saved overlay_spots.png")
        
    def compute_warnings(self):
        """Compute warning flags"""
        self.log("⚠️  Computing warning flags...")
        
        # WARN_LOW_TISSUE_COVERAGE
        self.warnings['WARN_LOW_TISSUE_COVERAGE'] = bool(self.metrics['tissue_coverage'] < 0.30)
        
        # WARN_MOTION_BLUR_OR_OUT_OF_FOCUS
        self.warnings['WARN_MOTION_BLUR_OR_OUT_OF_FOCUS'] = bool(self.metrics['low_focus_tile_frac'] > 0.40)
        
        # WARN_ALIGNMENT_CHECK
        self.warnings['WARN_ALIGNMENT_CHECK'] = bool(self.metrics['spot_on_tissue_pct'] < 0.80)
        
        # WARN_SMALL_IMAGE
        small_file = self.metrics['file_size_mb'] < 5.0
        small_dim = min(self.metrics['image_width'], self.metrics['image_height']) < 1500
        self.warnings['WARN_SMALL_IMAGE'] = bool(small_file or small_dim)
        
        # Log warnings
        for key, value in self.warnings.items():
            if value:
                self.log(f"   🚨 {key}: TRUE")
                
    def save_results(self):
        """Save QC results"""
        self.log("💾 Saving results...")
        
        # Combine metrics and warnings
        results = {**self.metrics, **self.warnings}
        
        # Save JSON
        with open(self.out_dir / "qc_summary.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save CSV (one row)
        df = pd.DataFrame([results])
        df.to_csv(self.out_dir / "qc_summary.csv", index=False)
        
        self.log(f"✅ Results saved to {self.out_dir}")
        
    def run(self):
        """Run complete QC pipeline"""
        self.log(f"Starting QC for: {self.sample_dir.name}")
        self.log("=" * 60)
        
        try:
            self.load_image()
            self.load_scalefactors()
            self.load_positions()
            self.compute_tissue_mask()
            self.compute_blur_metrics()
            self.compute_spot_alignment()
            self.generate_overlay()
            self.compute_warnings()
            self.save_results()
            
            self.log("=" * 60)
            self.log("✅ QC completed successfully!")
            return True
            
        except Exception as e:
            self.log(f"❌ QC failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Visium Image Quality Control')
    parser.add_argument('--sample_dir', type=str, required=True,
                       help='Path to Visium sample directory (e.g., GSM5494475)')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for QC results')
    parser.add_argument('--positions_csv', type=str, default='auto',
                       help='Path to tissue_positions file (auto-detect by default)')
    parser.add_argument('--use_lowres_if_missing', type=str, default='true',
                       help='Use lowres image if hires is missing (true/false)')
    parser.add_argument('--tile_size', type=int, default=256,
                       help='Tile size for blur analysis')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    use_lowres = args.use_lowres_if_missing.lower() == 'true'
    
    qc = VisiumImageQC(
        sample_dir=args.sample_dir,
        out_dir=args.out_dir,
        positions_csv=args.positions_csv,
        use_lowres_if_missing=use_lowres,
        tile_size=args.tile_size,
        seed=args.seed
    )
    
    success = qc.run()
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
