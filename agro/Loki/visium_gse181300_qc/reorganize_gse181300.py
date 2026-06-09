#!/usr/bin/env python3
"""
Reorganize GSE181300 files into proper Visium directory structure
"""

import gzip
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def reorganize_gse181300(root_dir):
    """Reorganize flat GEO files into Visium directory structure"""
    root_dir = Path(root_dir)
    
    print("🔍 Scanning files...")
    
    # Group files by GSM ID
    files_by_gsm = defaultdict(list)
    
    for f in root_dir.glob("GSM*"):
        if f.is_file():
            # Extract GSM ID (e.g., GSM5494475)
            gsm_id = f.name.split('_')[0]
            files_by_gsm[gsm_id].append(f)
    
    print(f"✅ Found {len(files_by_gsm)} unique GSM samples")
    
    for gsm_id in sorted(files_by_gsm.keys()):
        print(f"\n📁 Processing {gsm_id}...")
        
        # Create sample directory
        sample_dir = root_dir / gsm_id
        sample_dir.mkdir(exist_ok=True)
        
        spatial_dir = sample_dir / "spatial"
        spatial_dir.mkdir(exist_ok=True)
        
        files = files_by_gsm[gsm_id]
        
        for src in tqdm(files, desc=f"  {gsm_id}"):
            filename = src.name
            
            # Determine destination
            if 'tissue_hires_image' in filename or \
               'tissue_lowres_image' in filename or \
               'tissue_positions' in filename or \
               'scalefactors_json' in filename:
                # Spatial files
                dest_dir = spatial_dir
                # Simplify filename
                if 'tissue_hires_image' in filename:
                    dest_name = 'tissue_hires_image.png.gz'
                elif 'tissue_lowres_image' in filename:
                    dest_name = 'tissue_lowres_image.png.gz'
                elif 'tissue_positions' in filename:
                    dest_name = 'tissue_positions_list.csv.gz'
                elif 'scalefactors_json' in filename:
                    dest_name = 'scalefactors_json.json.gz'
                else:
                    dest_name = filename
            else:
                # Other files (expression matrices, etc.)
                dest_dir = sample_dir
                dest_name = filename
            
            dest = dest_dir / dest_name
            
            # Move file
            if not dest.exists():
                shutil.move(str(src), str(dest))
        
        print(f"  ✅ Reorganized {gsm_id}")
    
    print("\n" + "="*70)
    print("✅ Reorganization complete!")
    print("="*70)
    
    # Show sample structure
    sample_dirs = sorted(root_dir.glob("GSM*"))
    if sample_dirs:
        print(f"\n📂 Example structure ({sample_dirs[0].name}):")
        for item in sorted(sample_dirs[0].rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(sample_dirs[0])
                print(f"   {rel_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reorganize_gse181300.py <root_dir>")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    reorganize_gse181300(root_dir)
