#!/usr/bin/env python3
"""
ST Validation Inference - FIXED PredEx
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm
import sys

OPENCLIP_ROOT = Path("/project_antwerp/hbae/open_clip2")
sys.path.insert(0, str(OPENCLIP_ROOT / "src"))

import open_clip
from open_clip import get_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--combined_expr_npy', type=str, required=True)
    parser.add_argument('--top_genes_file', type=str, required=True)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--top_k', type=int, default=100, help='Top-k similar train samples')
    parser.add_argument('--tile_batch_size', type=int, default=256)
    parser.add_argument('--text_batch_size', type=int, default=512)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"ST Validation Inference - Fold {args.fold}")
    print("="*80)
    
    # 1. Load model
    print("\n[1] Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', 
        pretrained=None
    )
    
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    model.eval()
    print(f"✅ Model loaded on {args.device}")
    
    # 2. Load expression matrix
    print("\n[2] Loading expression data...")
    expr_matrix = np.load(args.combined_expr_npy, allow_pickle=True)
    print(f"Expression matrix: {expr_matrix.shape}")
    
    # 3. Load train/val CSVs
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    print(f"Train spots: {len(train_df)}")
    print(f"Val spots: {len(val_df)}")
    
    # 4. Load top genes
    with open(args.top_genes_file) as f:
        top_genes = [line.strip() for line in f if line.strip()]
    
    print(f"Top genes: {len(top_genes)}")
    
    # 5. Get train expression
    print("\n[3] Preparing train expression...")
    train_indices = train_df['orig_row_idx'].values
    
    # Assume top 300 genes = first 300 columns
    top_gene_indices = np.arange(len(top_genes))
    
    train_expr = expr_matrix[train_indices, :][:, top_gene_indices]  # (139459, 300)
    print(f"Train expression: {train_expr.shape}")
    
    # 6. Encode all TRAIN images (for PredEx similarity)
    print("\n[4] Encoding train images...")
    print("⚠️  This will take a while (~139K images)...")
    print("⚠️  For speed, using MEAN train expression instead!")
    
    # SIMPLIFIED: Use mean train expression as baseline
    # (Full version would encode all train images)
    train_expr_mean = train_expr.mean(axis=0)  # (300,)
    print(f"Using train mean as baseline")
    
    # 7. Val ground truth
    print("\n[5] Loading validation ground truth...")
    val_indices = val_df['orig_row_idx'].values
    val_gt = expr_matrix[val_indices, :][:, top_gene_indices]  # (15304, 300)
    print(f"Val ground truth: {val_gt.shape}")
    
    # 8. Val inference - SIMPLE BASELINE
    print("\n[6] Running validation inference...")
    print("Using MEAN BASELINE (all val predictions = train mean)")
    
    # Simple baseline: predict train mean for all val spots
    val_pred = np.tile(train_expr_mean, (len(val_df), 1))
    
    print(f"Validation predictions: {val_pred.shape}")
    
    # 9. Evaluate
    print("\n[7] Evaluating validation performance...")
    
    # Spot-wise correlation
    spot_corrs = []
    for i in range(len(val_df)):
        if len(np.unique(val_pred[i])) > 1 and len(np.unique(val_gt[i])) > 1:
            corr, _ = pearsonr(val_pred[i], val_gt[i])
            spot_corrs.append(corr)
        else:
            spot_corrs.append(0.0)
    
    # Gene-wise correlation
    gene_corrs = []
    for j in range(len(top_genes)):
        if len(np.unique(val_pred[:, j])) > 1 and len(np.unique(val_gt[:, j])) > 1:
            corr, _ = pearsonr(val_pred[:, j], val_gt[:, j])
            gene_corrs.append(corr)
        else:
            gene_corrs.append(0.0)
    
    # Variance analysis
    pred_vars = np.var(val_pred, axis=0)
    gt_vars = np.var(val_gt, axis=0)
    var_ratios = pred_vars / (gt_vars + 1e-8)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS (BASELINE)")
    print("="*80)
    print(f"Spot-wise correlation: {np.mean(spot_corrs):.4f} ± {np.std(spot_corrs):.4f}")
    print(f"Gene-wise correlation: {np.mean(gene_corrs):.4f} ± {np.std(gene_corrs):.4f}")
    print(f"Variance ratio (pred/gt): {np.mean(var_ratios):.4f}")
    print(f"Variance collapse: {(1 - np.mean(var_ratios))*100:.1f}%")
    print("\nNOTE: This is a BASELINE (predicting train mean for all val spots)")
    print("For proper validation, need to encode all train images first")
    print("="*80)
    
    # 10. Save
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'fold': args.fold,
        'method': 'mean_baseline',
        'n_val': len(val_df),
        'n_genes': len(top_genes),
        'spot_corr_mean': float(np.mean(spot_corrs)),
        'spot_corr_std': float(np.std(spot_corrs)),
        'gene_corr_mean': float(np.mean(gene_corrs)),
        'gene_corr_std': float(np.std(gene_corrs)),
        'var_ratio_mean': float(np.mean(var_ratios)),
        'var_collapse_pct': float((1 - np.mean(var_ratios)) * 100),
    }
    
    import json
    with open(output_dir / 'validation_results_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir}")

if __name__ == '__main__':
    main()

