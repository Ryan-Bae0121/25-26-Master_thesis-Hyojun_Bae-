#!/usr/bin/env python3
"""
Run open_clip fine-tuning for each fold in 10-fold CV.

Usage:
    python run_finetune_10fold.py \
        --fold_dir ./folds_10fold \
        --openclip_root /project_antwerp/hbae/open_clip2 \
        --pretrained_ckpt /project_antwerp/assets/loki_ckpts/checkpoint.pt \
        --model coca_ViT-L-14 \
        --out_root ./finetune_10fold_runs \
        --epochs 5 \
        --batch_size 64 \
        --lr 5e-6 \
        --continue_on_error

This script:
1. Loads fold CSVs from fold_dir
2. For each fold, runs open_clip_train.main via subprocess
3. Saves logs to out_root/fold_*/train.log
4. Tracks success/failure for each fold
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run open_clip fine-tuning for each fold in 10-fold CV",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fold_dir", type=str, required=True,
                       help="Directory containing fold_*_train.csv files")
    parser.add_argument("--openclip_root", type=str, required=True,
                       help="Root directory of open_clip repository")
    parser.add_argument("--pretrained_ckpt", type=str, required=True,
                       help="Path to pretrained checkpoint")
    parser.add_argument("--model", type=str, default="coca_ViT-L-14",
                       help="Model name (default: coca_ViT-L-14)")
    parser.add_argument("--out_root", type=str, default="./finetune_10fold_runs",
                       help="Root output directory (default: ./finetune_10fold_runs)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=5e-6,
                       help="Learning rate (default: 5e-6)")
    parser.add_argument("--wd", type=float, default=0.1,
                       help="Weight decay (default: 0.1)")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Warmup steps (default: 10)")
    parser.add_argument("--workers", type=int, default=16,
                       help="Number of data loader workers (default: 16)")
    parser.add_argument("--continue_on_error", action="store_true",
                       help="Continue to next fold if current fold fails")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="WandB project name (default: auto-generated)")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging (use --report-to none)")
    
    args = parser.parse_args()
    
    fold_dir = Path(args.fold_dir)
    openclip_root = Path(args.openclip_root)
    pretrained_ckpt = Path(args.pretrained_ckpt)
    out_root = Path(args.out_root)
    
    # Validation
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    if not openclip_root.exists():
        raise FileNotFoundError(f"OpenCLIP root not found: {openclip_root}")
    if not pretrained_ckpt.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt}")
    
    # Find fold train CSVs
    fold_train_csvs = sorted(fold_dir.glob("fold_*_train.csv"))
    if len(fold_train_csvs) == 0:
        raise RuntimeError(f"No fold_*_train.csv files found in {fold_dir}")
    
    n_folds = len(fold_train_csvs)
    print("=" * 80)
    print("10-Fold CV Fine-tuning Runner")
    print("=" * 80)
    print(f"Fold directory: {fold_dir}")
    print(f"OpenCLIP root: {openclip_root}")
    print(f"Pretrained checkpoint: {pretrained_ckpt}")
    print(f"Output root: {out_root}")
    print(f"Number of folds: {n_folds}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Continue on error: {args.continue_on_error}")
    print("=" * 80)
    print()
    
    # Check WandB
    wandb_key = os.environ.get("WANDB_API_KEY")
    if args.no_wandb:
        report_to = "none"
        print("[INFO] WandB disabled (--no_wandb)")
    elif wandb_key:
        report_to = "wandb"
        print(f"[INFO] WandB enabled (API key found)")
    else:
        print("[WARNING] WANDB_API_KEY not found. Disabling WandB.")
        report_to = "none"
    
    # Create output root
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # Run fine-tuning for each fold
    # ========================================================================
    for fold_idx, train_csv in enumerate(fold_train_csvs, start=1):
        fold_name = f"fold_{fold_idx:02d}"
        print("=" * 80)
        print(f"[{fold_idx}/{n_folds}] {fold_name}")
        print("=" * 80)
        
        fold_out_dir = out_root / fold_name
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = fold_out_dir / "train.log"
        
        # Experiment name
        if args.wandb_project:
            exp_name = f"{args.wandb_project}_{fold_name}_{timestamp}"
        else:
            exp_name = f"finetune_{fold_name}_{timestamp}"
        
        print(f"  Train CSV: {train_csv}")
        print(f"  Output directory: {fold_out_dir}")
        print(f"  Experiment name: {exp_name}")
        print(f"  Log file: {log_file}")
        print()
        
        # Build command
        cmd = [
            sys.executable, "-u", "-m", "open_clip_train.main",
            "--model", args.model,
            "--pretrained", str(pretrained_ckpt),
            "--train-data", str(train_csv),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--wd", str(args.wd),
            "--warmup", str(args.warmup),
            "--workers", str(args.workers),
            "--csv-separator", ",",  # CSV separator: comma (not tab)
            "--csv-img-key", "filepath",  # CSV에서 이미지 경로 컬럼명 (open_clip 기본값)
            "--csv-caption-key", "title",  # CSV에서 텍스트(라벨) 컬럼명 (open_clip 기본값)
            "--lock-text-freeze-layer-norm",
            "--lock-image-freeze-bn-stats",
            "--coca-caption-loss-weight", "0",
            "--coca-contrastive-loss-weight", "1",
            "--save-frequency", "1",
            "--val-frequency", "10",
            "--report-to", report_to,
            "--debug",
            # Removed --aug-cfg due to parsing issues. Use defaults or configure separately if needed.
            "--save-most-recent",
            "--logs", str(fold_out_dir),
            "--name", exp_name,
        ]
        
        # Set environment
        env = os.environ.copy()
        # PYTHONPATH: open_clip src first, then existing paths
        existing_pythonpath = env.get('PYTHONPATH', '')
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{openclip_root}/src:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = f"{openclip_root}/src"
        
        # Also ensure we're using the correct Python site-packages
        # This helps when open_clip modules try to import dependencies
        import site
        site_packages = site.getsitepackages()
        if site_packages:
            # Add site-packages to PYTHONPATH if not already there
            for sp in site_packages:
                if sp not in env.get('PYTHONPATH', ''):
                    env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{sp}"
        
        # Check for common missing dependencies before running
        print(f"  Checking dependencies...")
        print(f"    Python: {sys.executable}")
        
        # Try to check if braceexpand is available in the subprocess environment
        # Use the same Python that will be used for training
        check_cmd = [sys.executable, "-c", "import braceexpand; print('OK')"]
        check_result = subprocess.run(
            check_cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=openclip_root  # Use same cwd as training
        )
        
        if check_result.returncode != 0:
            print(f"    ⚠ WARNING: braceexpand not found in subprocess environment")
            print(f"    Attempting to install in open_clip directory...")
            
            # Try to install braceexpand using the same Python
            install_cmd = [sys.executable, "-m", "pip", "install", "braceexpand", "--quiet"]
            install_result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=openclip_root
            )
            
            if install_result.returncode == 0:
                print(f"    ✓ braceexpand installed successfully")
                # Verify installation
                verify_result = subprocess.run(
                    check_cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=openclip_root
                )
                if verify_result.returncode != 0:
                    print(f"    ❌ Installation verification failed!")
                    print(f"    Please install manually:")
                    print(f"      cd {openclip_root}")
                    print(f"      {sys.executable} -m pip install braceexpand")
            else:
                print(f"    ❌ Failed to install braceexpand:")
                if install_result.stderr:
                    print(f"       {install_result.stderr}")
                if install_result.stdout:
                    print(f"       {install_result.stdout}")
                print(f"    Please install manually:")
                print(f"      cd {openclip_root}")
                print(f"      {sys.executable} -m pip install braceexpand")
        else:
            print(f"    ✓ braceexpand found")
        
        print(f"  Running command:")
        print(f"    {' '.join(cmd)}")
        print()
        
        # Run subprocess
        try:
            with open(log_file, 'w') as f_log:
                result = subprocess.run(
                    cmd,
                    cwd=openclip_root,
                    env=env,
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False  # Don't raise on non-zero exit
                )
            
            success = result.returncode == 0
            
            if success:
                print(f"  ✅ {fold_name} completed successfully")
            else:
                print(f"  ❌ {fold_name} failed with exit code {result.returncode}")
                print(f"     Check log: {log_file}")
                
                if not args.continue_on_error:
                    print("\n[ERROR] Stopping due to failure (use --continue_on_error to continue)")
                    sys.exit(1)
            
            results.append({
                "fold": fold_name,
                "train_csv": str(train_csv),
                "out_dir": str(fold_out_dir),
                "exp_name": exp_name,
                "log_file": str(log_file),
                "success": success,
                "exit_code": result.returncode,
            })
            
        except Exception as e:
            print(f"  ❌ {fold_name} crashed: {e}")
            if not args.continue_on_error:
                raise
            
            results.append({
                "fold": fold_name,
                "train_csv": str(train_csv),
                "out_dir": str(fold_out_dir),
                "exp_name": exp_name,
                "log_file": str(log_file),
                "success": False,
                "exit_code": -1,
                "error": str(e),
            })
        
        print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    
    print(f"Total folds: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed folds:")
        for r in results:
            if not r.get("success", False):
                print(f"  - {r['fold']}: exit_code={r.get('exit_code', 'N/A')}")
    
    # Save results
    import json
    results_json = out_root / "run_results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_json}")
    
    print("=" * 80)
    
    if failed > 0 and not args.continue_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()

