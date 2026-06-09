#!/usr/bin/env python3
"""
GSE181300 Batch QC Pipeline
Downloads, extracts, and runs quality control on all Visium samples.
"""

import argparse
import os
import subprocess
import tarfile
import gzip
import shutil
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm


class GSE181300Pipeline:
    def __init__(self, root_dir, out_root, download=True):
        self.root_dir = Path(root_dir)
        self.out_root = Path(out_root)
        self.download = download
        
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.out_root.mkdir(parents=True, exist_ok=True)
        
        self.tar_path = self.root_dir / "GSE181300_RAW.tar"
        self.url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE181300&format=file"
        
    def log(self, msg):
        print(f"[GSE181300] {msg}")
        
    def download_tar(self):
        """Download GSE181300_RAW.tar"""
        if self.tar_path.exists():
            size_mb = self.tar_path.stat().st_size / (1024 * 1024)
            self.log(f"✅ TAR file already exists: {self.tar_path} ({size_mb:.1f} MB)")
            
            if size_mb < 300:
                self.log(f"❌ File size too small ({size_mb:.1f} MB < 300 MB), likely corrupt!")
                self.log("   Deleting and re-downloading...")
                self.tar_path.unlink()
            else:
                return
                
        self.log(f"📥 Downloading GSE181300_RAW.tar from NCBI...")
        self.log(f"   URL: {self.url}")
        
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(self.tar_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc='Downloading'
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
                
        size_mb = self.tar_path.stat().st_size / (1024 * 1024)
        self.log(f"✅ Download complete: {size_mb:.1f} MB")
        
        if size_mb < 300:
            raise ValueError(f"Downloaded file too small ({size_mb:.1f} MB < 300 MB), likely corrupt!")
            
    def extract_tar(self):
        """Extract TAR file"""
        # Check if already extracted
        gsm_dirs = list(self.root_dir.glob("GSM*"))
        if gsm_dirs:
            self.log(f"✅ TAR already extracted ({len(gsm_dirs)} GSM directories found)")
            return
            
        self.log(f"📦 Extracting TAR file...")
        
        with tarfile.open(self.tar_path, 'r') as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc='Extracting'):
                tar.extract(member, self.root_dir)
                
        self.log(f"✅ Extraction complete")
        
    def discover_samples(self):
        """Discover GSM sample directories"""
        gsm_dirs = sorted(self.root_dir.glob("GSM*"))
        self.log(f"🔍 Found {len(gsm_dirs)} GSM samples:")
        for d in gsm_dirs:
            self.log(f"   - {d.name}")
        return gsm_dirs
        
    def check_pairing(self, sample_dir):
        """Check if sample has required Visium files"""
        spatial_dir = sample_dir / "spatial"
        
        if not spatial_dir.exists():
            return False, "No spatial/ directory"
            
        # Check for image (hires or lowres)
        hires = spatial_dir / "tissue_hires_image.png"
        hires_gz = spatial_dir / "tissue_hires_image.png.gz"
        lowres = spatial_dir / "tissue_lowres_image.png"
        
        has_image = hires.exists() or hires_gz.exists() or lowres.exists()
        
        # Check for scalefactors
        scalefactors = spatial_dir / "scalefactors_json.json"
        has_scalefactors = scalefactors.exists()
        
        # Check for positions
        positions1 = spatial_dir / "tissue_positions.csv"
        positions2 = spatial_dir / "tissue_positions_list.csv"
        has_positions = positions1.exists() or positions2.exists()
        
        if not has_image:
            return False, "Missing tissue image"
        if not has_scalefactors:
            return False, "Missing scalefactors_json.json"
        if not has_positions:
            return False, "Missing tissue_positions file"
            
        # Gunzip if needed
        if hires_gz.exists() and not hires.exists():
            self.log(f"   🗜️  Gunzipping {hires_gz.name}...")
            with gzip.open(hires_gz, 'rb') as f_in:
                with open(hires, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
        return True, "OK"
        
    def run_qc(self, sample_dir, out_dir):
        """Run QC on a single sample"""
        cmd = [
            'python', 'visium_image_qc.py',
            '--sample_dir', str(sample_dir),
            '--out_dir', str(out_dir),
            '--positions_csv', 'auto',
            '--use_lowres_if_missing', 'true',
            '--tile_size', '256',
            '--seed', '42'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.log(f"❌ QC failed for {sample_dir.name}")
            self.log(f"   Error: {result.stderr}")
            return False
        else:
            self.log(f"✅ QC completed for {sample_dir.name}")
            return True
            
    def aggregate_results(self, samples_info):
        """Aggregate QC results into summary CSV"""
        self.log("📊 Aggregating results...")
        
        all_results = []
        missing_spatial = []
        
        for sample_name, status, reason in samples_info:
            if status == "MISSING_SPATIAL":
                missing_spatial.append({'sample': sample_name, 'reason': reason})
            else:
                qc_csv = self.out_root / sample_name / "qc_summary.csv"
                if qc_csv.exists():
                    df = pd.read_csv(qc_csv)
                    df.insert(0, 'sample', sample_name)
                    all_results.append(df)
                    
        # Save combined results
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            output_path = self.out_root / "gse181300_qc_summary.csv"
            combined.to_csv(output_path, index=False)
            self.log(f"✅ Combined QC summary saved: {output_path}")
            self.log(f"   Total samples processed: {len(all_results)}")
            
            # Print warning summary
            warn_cols = [c for c in combined.columns if c.startswith('WARN_')]
            if warn_cols:
                self.log("\n⚠️  Warning Summary:")
                for col in warn_cols:
                    count = combined[col].sum()
                    if count > 0:
                        samples = combined[combined[col]]['sample'].tolist()
                        self.log(f"   {col}: {count} samples")
                        self.log(f"      → {', '.join(samples)}")
        else:
            self.log("⚠️  No QC results to aggregate")
            
        # Save missing spatial info
        if missing_spatial:
            missing_df = pd.DataFrame(missing_spatial)
            missing_path = self.out_root / "missing_spatial_files.csv"
            missing_df.to_csv(missing_path, index=False)
            self.log(f"\n❌ Samples with missing spatial files: {len(missing_spatial)}")
            self.log(f"   Details saved to: {missing_path}")
            for item in missing_spatial:
                self.log(f"   - {item['sample']}: {item['reason']}")
                
    def generate_readme(self, samples_info):
        """Generate README with summary"""
        self.log("📝 Generating README...")
        
        total = len(samples_info)
        processed = sum(1 for _, status, _ in samples_info if status == "OK")
        missing = sum(1 for _, status, _ in samples_info if status == "MISSING_SPATIAL")
        
        readme = f"""# GSE181300 Visium QC Results

## Summary

- **Total samples**: {total}
- **Successfully processed**: {processed}
- **Missing spatial files**: {missing}

## Files

- `gse181300_qc_summary.csv` - Combined QC metrics for all samples
- `missing_spatial_files.csv` - List of samples with missing files (if any)
- `GSM*/` - Individual sample QC results
  - `qc_summary.json` - Detailed metrics
  - `qc_summary.csv` - One-row summary
  - `overlay_spots.png` - H&E with spot overlay
  - `tissue_mask.png` - Tissue segmentation mask
  - `blur_map.png` - Per-tile focus quality heatmap
  - `hist_brightness.png` - Brightness and Laplacian histograms

## Warning Flags

QC flags are set based on these thresholds:

- **WARN_LOW_TISSUE_COVERAGE**: Tissue coverage < 30%
- **WARN_MOTION_BLUR_OR_OUT_OF_FOCUS**: Low-focus tile fraction > 40%
- **WARN_ALIGNMENT_CHECK**: Spot-on-tissue < 80%
- **WARN_SMALL_IMAGE**: File size < 5 MB or min dimension < 1500 px

## Interpretation Tips

1. **Check `overlay_spots.png`** first:
   - Green circles = spots on tissue (good)
   - Red circles = spots off tissue (alignment issue)

2. **Review `blur_map.png`**:
   - Hot colors = good focus
   - Cold colors = blurry regions

3. **Examine samples with warnings**:
   - Sort `gse181300_qc_summary.csv` by warning columns
   - Prioritize samples with multiple warnings

4. **Tissue coverage**:
   - Low coverage may indicate sectioning issues
   - Check if tissue is centered in the image

5. **Spot alignment**:
   - Poor alignment suggests registration problems
   - May need manual re-alignment

## Generated by

`run_gse181300_qc.py` - GSE181300 Visium QC Pipeline
"""
        
        readme_path = self.out_root / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
            
        self.log(f"✅ README saved: {readme_path}")
        
    def run(self):
        """Run complete pipeline"""
        self.log("=" * 70)
        self.log("GSE181300 Visium QC Pipeline")
        self.log("=" * 70)
        
        # Step 1: Download
        if self.download:
            self.download_tar()
        else:
            if not self.tar_path.exists():
                self.log(f"❌ TAR file not found: {self.tar_path}")
                self.log("   Run with --download true to download it")
                return
                
        # Step 2: Extract
        self.extract_tar()
        
        # Step 3: Discover samples
        samples = self.discover_samples()
        
        if not samples:
            self.log("❌ No GSM samples found!")
            return
            
        # Step 4: Check pairing and run QC
        self.log("\n" + "=" * 70)
        self.log("Checking sample pairing and running QC...")
        self.log("=" * 70)
        
        samples_info = []
        
        for sample_dir in samples:
            sample_name = sample_dir.name
            self.log(f"\n📁 Processing {sample_name}...")
            
            # Check pairing
            has_pairing, reason = self.check_pairing(sample_dir)
            
            if not has_pairing:
                self.log(f"   ⚠️  MISSING_SPATIAL: {reason}")
                samples_info.append((sample_name, "MISSING_SPATIAL", reason))
                continue
                
            self.log(f"   ✅ Spatial files OK")
            
            # Run QC
            out_dir = self.out_root / sample_name
            success = self.run_qc(sample_dir, out_dir)
            
            if success:
                samples_info.append((sample_name, "OK", ""))
            else:
                samples_info.append((sample_name, "FAILED", "QC script error"))
                
        # Step 5: Aggregate results
        self.log("\n" + "=" * 70)
        self.aggregate_results(samples_info)
        
        # Step 6: Generate README
        self.generate_readme(samples_info)
        
        self.log("\n" + "=" * 70)
        self.log("✅ Pipeline completed!")
        self.log("=" * 70)
        self.log(f"\n📂 Results directory: {self.out_root}")
        self.log(f"   - gse181300_qc_summary.csv")
        self.log(f"   - README.md")
        self.log(f"   - GSM*/overlay_spots.png (check alignment)")
        self.log(f"   - GSM*/blur_map.png (check focus)")


def main():
    parser = argparse.ArgumentParser(
        description='GSE181300 Visium QC Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_gse181300_qc.py --root_dir ~/data/GSE181300 --out_root ~/data/GSE181300_QC --download true
        """
    )
    
    parser.add_argument('--root_dir', type=str, required=True,
                       help='Root directory for GSE181300 data')
    parser.add_argument('--out_root', type=str, required=True,
                       help='Output root directory for QC results')
    parser.add_argument('--download', type=str, default='true',
                       help='Download TAR file if missing (true/false)')
    
    args = parser.parse_args()
    
    download = args.download.lower() == 'true'
    
    pipeline = GSE181300Pipeline(
        root_dir=args.root_dir,
        out_root=args.out_root,
        download=download
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()
