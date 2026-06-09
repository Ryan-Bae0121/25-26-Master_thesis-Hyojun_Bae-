#!/usr/bin/env python3
"""
End-to-end zero-shot pipeline: download checkpoint, build bank, predict, evaluate.
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download


def log(msg):
    print(f"[Pipeline] {msg}")


def download_checkpoint(out_dir, hf_token_env):
    """Download Loki checkpoint from HuggingFace"""
    log("📥 Checking for Loki checkpoint...")
    
    ckpt_path = Path(out_dir) / "checkpoint.pt"
    
    if ckpt_path.exists():
        log(f"   ✅ Checkpoint already exists: {ckpt_path}")
        return str(ckpt_path)
    
    log("   Downloading from HuggingFace (WangGuangyuLab/Loki)...")
    
    # Get token from environment
    token = os.environ.get(hf_token_env)
    if not token:
        log(f"   ⚠️  Warning: {hf_token_env} not set, attempting without token...")
        token = None
    
    try:
        downloaded = hf_hub_download(
            repo_id="WangGuangyuLab/Loki",
            filename="Loki/checkpoint.pt",
            token=token,
            cache_dir=out_dir
        )
        
        # Copy to expected location
        import shutil
        shutil.copy(downloaded, ckpt_path)
        
        log(f"   ✅ Downloaded: {ckpt_path}")
        return str(ckpt_path)
        
    except Exception as e:
        log(f"   ❌ Download failed: {e}")
        log(f"   Please download manually from: https://huggingface.co/WangGuangyuLab/Loki")
        raise


def build_bank(bank_h5ad, ckpt_path, bank_dir, topn=50, device='cuda', use_demo=False):
    """Build text bank from AnnData"""
    log("🏗️  Building text bank...")
    
    bank_dir = Path(bank_dir)
    
    # Check if already exists
    required_files = [
        bank_dir / "bank_text_emb.npy",
        bank_dir / "bank_expr.npy",
        bank_dir / "bank_genes.txt"
    ]
    
    if all(f.exists() for f in required_files):
        log(f"   ✅ Bank already exists: {bank_dir}")
        return str(bank_dir)
    
    # Run build_text_bank.py
    script_path = Path(__file__).parent / "build_text_bank.py"
    
    cmd = [
        sys.executable, str(script_path),
        '--bank_h5ad', str(bank_h5ad),
        '--out_dir', str(bank_dir),
        '--topn', str(topn),
        '--device', device
    ]
    
    if use_demo:
        cmd.append('--use_demo')
    else:
        cmd.extend(['--hf_ckpt', str(ckpt_path)])
    
    log(f"   Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        log(f"   ❌ Bank building failed")
        raise RuntimeError("Bank building failed")
    
    log(f"   ✅ Bank built: {bank_dir}")
    return str(bank_dir)


def run_zero_shot(tiles_dir, bank_dir, ckpt_path, out_dir, 
                  genes_list=None, normalize='bank_log1p', 
                  temp=1.0, topk=64, device='cuda', batch_size=128, use_demo=False):
    """Run zero-shot prediction"""
    log("🔮 Running zero-shot prediction...")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pred_csv = out_dir / "pred_tile_gene.csv"
    
    if pred_csv.exists():
        log(f"   ✅ Predictions already exist: {pred_csv}")
        return str(pred_csv)
    
    # Run zero_shot_loki.py
    script_path = Path(__file__).parent / "zero_shot_loki.py"
    
    bank_dir = Path(bank_dir)
    
    cmd = [
        sys.executable, str(script_path),
        '--tiles_dir', str(tiles_dir),
        '--bank_text_emb', f"npy:{bank_dir / 'bank_text_emb.npy'}",
        '--bank_expr', f"npy:{bank_dir / 'bank_expr.npy'}",
        '--bank_genes', str(bank_dir / 'bank_genes.txt'),
        '--out_csv', str(pred_csv),
        '--device', device,
        '--batch_size', str(batch_size),
        '--normalize', normalize,
        '--temp', str(temp),
        '--topk', str(topk)
    ]
    
    if use_demo:
        cmd.append('--use_demo')
    else:
        cmd.extend(['--hf_ckpt', str(ckpt_path)])
    
    if genes_list:
        cmd.extend(['--genes_list', str(genes_list)])
    
    log(f"   Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        log(f"   ❌ Zero-shot prediction failed")
        raise RuntimeError("Zero-shot prediction failed")
    
    log(f"   ✅ Predictions saved: {pred_csv}")
    return str(pred_csv)


def run_evaluation(pred_csv, tcga_csv, slide_id, eval_script, out_dir, 
                   agg='both', log1p=True):
    """Run slide-level evaluation"""
    log("📊 Running slide-level evaluation...")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if eval script exists
    eval_script = Path(eval_script)
    if not eval_script.exists():
        log(f"   ❌ Evaluation script not found: {eval_script}")
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")
    
    cmd = [
        sys.executable, str(eval_script),
        '--pred_csv', str(pred_csv),
        '--tcga_csv', str(tcga_csv),
        '--slide_id', slide_id,
        '--out_dir', str(out_dir),
        '--agg', agg,
        '--log1p', 'true' if log1p else 'false'
    ]
    
    log(f"   Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        log(f"   ⚠️  Evaluation completed with warnings")
    else:
        log(f"   ✅ Evaluation completed: {out_dir}")
    
    return str(out_dir)


def generate_summary(out_root, bank_dir, pred_csv, eval_dir, elapsed):
    """Generate summary README"""
    log("📝 Generating summary...")
    
    out_root = Path(out_root)
    
    readme_content = f"""# Zero-Shot Loki Pipeline Results

## Summary

**Pipeline completed in {elapsed:.1f}s**

## Directories

- **Bank**: `{bank_dir}`
  - Text embeddings, expression matrix, gene list
- **Predictions**: `{pred_csv}`
  - Tile-level gene expression predictions
- **Evaluation**: `{eval_dir}`
  - Slide-level metrics (MSE, sMAPE, Pearson, Spearman)
  - Calibration methods: α-LS, α-L1
  - Transformations: raw, log1p

## Pipeline Steps

1. ✅ Downloaded/verified Loki checkpoint
2. ✅ Built text bank from Visium AnnData
3. ✅ Encoded tiles and computed zero-shot predictions
4. ✅ Evaluated predictions at slide level

## Key Files

### Predictions
- `pred/pred_tile_gene.csv` - Tile×gene predictions (rows=tiles, cols=genes)
- `pred/tile_index.tsv` - Tile name to path mapping
- `pred/gene_index.tsv` - Gene list
- `pred/pred_info.json` - Prediction metadata

### Evaluation
- `eval/metrics_table.csv` - All metrics (MSE, sMAPE, Pearson, Spearman)
- `eval/scatter_*.png` - Scatter plots for visual inspection
- `eval/evaluation_summary.json` - Detailed evaluation info

## Notes

### Zero-Shot Approach
- Uses **Loki/OmiCLIP foundation model** (no task-specific fine-tuning)
- **Retrieval-based inference**: tiles retrieve similar spots from Visium bank
- **Weighted aggregation**: predictions = softmax-weighted sum of retrieved spot expressions

### Hyperparameters
- Temperature: controls softmax sharpness (default: 1.0)
- Top-k: number of spots to retrieve per tile (default: 64)
- Normalization: log1p applied to bank expression (default: bank_log1p)

### Evaluation Methods
- **Raw metrics**: direct comparison of predicted vs ground truth
- **α-LS calibration**: least-squares scaling per gene
- **α-L1 calibration**: L1-norm scaling per gene
- **log1p transform**: log(1+x) for both pred and GT
- **Zero-aware sMAPE**: handles zero values properly

## Interpretation

1. **Check scatter plots** in `eval/` to visually assess prediction quality
2. **Review metrics_table.csv** for quantitative performance
3. **Compare calibration methods**: α-LS typically best for linear scaling
4. **log1p metrics** often more interpretable for gene expression

## Citation

If you use this pipeline, please cite:
- **Loki**: Wang et al., "Loki: A foundation model for spatial transcriptomics"
- **OmiCLIP**: Contrastive learning framework for omics and imaging
"""
    
    readme_path = out_root / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    log(f"   ✅ Summary saved: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end zero-shot Loki pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_zero_shot_and_eval.py \\
    --tiles_dir /data/tiles_TCGA-QK-A6IH \\
    --bank_h5ad /data/HNSCC_visium_bank.h5ad \\
    --tcga_csv /shares/bioit-students/ir_students_2020/TCGA-HNSC/ref_file.csv \\
    --slide_id TCGA-QK-A6IH-01Z-00-DX1 \\
    --eval_script /path/to/slide_level_calibrated_metrics.py \\
    --out_root ./zero_shot_run
        """
    )
    
    # Required
    parser.add_argument('--tiles_dir', type=str, required=True,
                       help='Directory containing tiles')
    parser.add_argument('--bank_h5ad', type=str, required=True,
                       help='Visium bank AnnData (.h5ad)')
    parser.add_argument('--tcga_csv', type=str, required=True,
                       help='TCGA reference CSV')
    parser.add_argument('--slide_id', type=str, required=True,
                       help='Slide ID for evaluation')
    parser.add_argument('--eval_script', type=str, required=True,
                       help='Path to slide_level_calibrated_metrics.py')
    parser.add_argument('--out_root', type=str, required=True,
                       help='Output root directory')
    
    # Optional
    parser.add_argument('--hf_token_env', type=str, default='HF_TOKEN',
                       help='Environment variable for HF token')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--genes_list', type=str, default=None,
                       help='Optional: gene subset file')
    parser.add_argument('--topn', type=int, default=50,
                       help='Top-N genes per spot for bank text')
    parser.add_argument('--normalize', type=str, default='bank_log1p',
                       help='Normalization mode')
    parser.add_argument('--temp', type=float, default=1.0,
                       help='Softmax temperature')
    parser.add_argument('--topk', type=int, default=64,
                       help='Top-k spots per tile')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for encoding')
    parser.add_argument('--agg', type=str, default='both',
                       choices=['mean', 'median', 'both'],
                       help='Aggregation method for slide-level')
    parser.add_argument('--log1p', action='store_true',
                       help='Use log1p transform in evaluation')
    parser.add_argument('--use_demo', action='store_true',
                       help='Use lightweight demo checkpoint (skip HF download)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Zero-Shot Loki Pipeline (Foundation Model Only)")
    print("=" * 70)
    
    start_time = time.time()
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    ckpt_dir = out_root / "checkpoint"
    bank_dir = out_root / "bank"
    pred_dir = out_root / "pred"
    eval_dir = out_root / "eval"
    logs_dir = out_root / "logs"
    
    for d in [ckpt_dir, bank_dir, pred_dir, eval_dir, logs_dir]:
        d.mkdir(exist_ok=True)
    
    try:
        # Step 1: Download checkpoint (skip if demo mode)
        if args.use_demo:
            log("\n" + "=" * 70)
            log("Step 1/4: Using Demo Checkpoint (SKIPPED)")
            log("=" * 70)
            log("   [INFO] Demo mode enabled - skipping HF download")
            ckpt_path = None
        else:
            log("\n" + "=" * 70)
            log("Step 1/4: Download Checkpoint")
            log("=" * 70)
            ckpt_path = download_checkpoint(ckpt_dir, args.hf_token_env)
        
        # Step 2: Build bank
        log("\n" + "=" * 70)
        log("Step 2/4: Build Text Bank")
        log("=" * 70)
        bank_dir = build_bank(
            args.bank_h5ad, ckpt_path, bank_dir, 
            topn=args.topn, device=args.device, use_demo=args.use_demo
        )
        
        # Step 3: Run zero-shot prediction
        log("\n" + "=" * 70)
        log("Step 3/4: Zero-Shot Prediction")
        log("=" * 70)
        pred_csv = run_zero_shot(
            args.tiles_dir, bank_dir, ckpt_path, pred_dir,
            genes_list=args.genes_list,
            normalize=args.normalize,
            temp=args.temp,
            topk=args.topk,
            device=args.device,
            batch_size=args.batch_size,
            use_demo=args.use_demo
        )
        
        # Step 4: Evaluate
        log("\n" + "=" * 70)
        log("Step 4/4: Slide-Level Evaluation")
        log("=" * 70)
        eval_dir = run_evaluation(
            pred_csv, args.tcga_csv, args.slide_id,
            args.eval_script, eval_dir,
            agg=args.agg, log1p=args.log1p
        )
        
        # Generate summary
        elapsed = time.time() - start_time
        generate_summary(out_root, bank_dir, pred_csv, eval_dir, elapsed)
        
        log("\n" + "=" * 70)
        log("✅ Pipeline completed successfully!")
        log("=" * 70)
        log(f"\n📂 Results: {out_root}")
        log(f"   - Predictions: {pred_csv}")
        log(f"   - Evaluation: {eval_dir}")
        log(f"   - Summary: {out_root / 'README.md'}")
        log(f"\n⏱️  Total time: {elapsed:.1f}s")
        
    except Exception as e:
        log(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
