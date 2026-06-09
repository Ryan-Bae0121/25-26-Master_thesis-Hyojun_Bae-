#!/usr/bin/env python3
"""
ST Validation Inference - FULL TRAIN SET
Encode all 139K train images (will take ~2-3 hours)
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
import json
from datetime import datetime

OPENCLIP_ROOT = Path("/project_antwerp/hbae/open_clip2")
sys.path.insert(0, str(OPENCLIP_ROOT / "src"))

import open_clip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--combined_expr_npy', type=str, required=True)
    parser.add_argument('--top_genes_file', type=str, required=True)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_sample_size', type=int, default=None, help='Sample val for speed (None = all)')
    parser.add_argument('--save_embeddings', action='store_true', help='Save train embeddings to disk')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("="*80)
    print(f"ST Validation Inference - FULL TRAIN SET")
    print(f"Fold {args.fold}")
    print("="*80)
    print(f"Start time: {start_time}")
    print()
    
    # Load model
    print("[1] Loading model...")
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
    
    # Load data
    print("\n[2] Loading data...")
    expr_matrix = np.load(args.combined_expr_npy, allow_pickle=True)
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    with open(args.top_genes_file) as f:
        top_genes = [line.strip() for line in f if line.strip()]
    
    print(f"Expression matrix: {expr_matrix.shape}")
    print(f"Train spots: {len(train_df):,}")
    print(f"Val spots: {len(val_df):,}")
    print(f"Top genes: {len(top_genes)}")
    
    # Val sampling (optional)
    if args.val_sample_size:
        print(f"\n[3] Sampling {args.val_sample_size:,} val spots...")
        np.random.seed(42)
        val_sample_idx = np.random.choice(len(val_df), args.val_sample_size, replace=False)
        val_df = val_df.iloc[val_sample_idx].reset_index(drop=True)
    else:
        print(f"\n[3] Using ALL {len(val_df):,} validation spots")
    
    # Get expression
    top_gene_indices = np.arange(len(top_genes))
    
    train_indices = train_df['orig_row_idx'].values
    train_expr = expr_matrix[train_indices, :][:, top_gene_indices]
    
    val_indices = val_df['orig_row_idx'].values
    val_gt = expr_matrix[val_indices, :][:, top_gene_indices]
    
    print(f"Train expression: {train_expr.shape}")
    print(f"Val ground truth: {val_gt.shape}")
    
    # Encode ALL train images
    print(f"\n[4] Encoding ALL {len(train_df):,} train images...")
    print("⚠️  This will take 2-3 hours!")
    print()
    
    train_embeds = []
    train_success_indices = []
    
    encode_start = datetime.now()
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Encoding train"):
        img_path = row['filepath']
        
        if not Path(img_path).exists():
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                embed = model.encode_image(img_tensor)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                train_embeds.append(embed.cpu().numpy())
                train_success_indices.append(idx)
        
        except Exception as e:
            if idx < 10:  # Only print first 10 errors
                print(f"⚠️  Error at {idx}: {e}")
            continue
        
        # Progress update every 10K
        if (idx + 1) % 10000 == 0:
            elapsed = (datetime.now() - encode_start).total_seconds()
            rate = (idx + 1) / elapsed
            remaining = (len(train_df) - idx - 1) / rate
            print(f"  Progress: {idx+1:,}/{len(train_df):,} | "
                  f"Rate: {rate:.1f} img/s | "
                  f"ETA: {remaining/60:.1f} min")
    
    train_embeds = np.vstack(train_embeds)  # (N, 768)
    train_expr = train_expr[train_success_indices]
    
    encode_end = datetime.now()
    encode_time = (encode_end - encode_start).total_seconds()
    
    print(f"\n✅ Train encoding complete!")
    print(f"   Successful: {len(train_embeds):,} / {len(train_df):,}")
    print(f"   Time: {encode_time/60:.1f} minutes")
    print(f"   Rate: {len(train_embeds)/encode_time:.1f} images/sec")
    print(f"   Train embeddings shape: {train_embeds.shape}")
    print(f"   Train expression shape: {train_expr.shape}")
    
    # Save embeddings (optional)
    if args.save_embeddings:
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[5] Saving train embeddings...")
        np.save(output_dir / 'train_embeddings.npy', train_embeds)
        np.save(output_dir / 'train_expression.npy', train_expr)
        np.save(output_dir / 'train_success_indices.npy', train_success_indices)
        print(f"✅ Saved to {output_dir}")
    
    # Val inference
    print(f"\n[6] Running validation inference on {len(val_df):,} spots...")
    
    val_predictions = []
    inference_start = datetime.now()
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val inference"):
        img_path = row['filepath']
        
        if not Path(img_path).exists():
            val_predictions.append(train_expr.mean(axis=0))
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                val_embed = model.encode_image(img_tensor)
                val_embed = val_embed / val_embed.norm(dim=-1, keepdim=True)
                val_embed_np = val_embed.cpu().numpy()  # (1, 768)
            
            # Compute similarity
            sim = val_embed_np @ train_embeds.T  # (1, N_train)
            
            # Top-k
            top_k = min(args.top_k, len(train_embeds))
            top_k_idx = np.argsort(sim[0])[-top_k:]
            
            # Softmax with numerical stability
            sim_topk = sim[0, top_k_idx] / args.tau
            sim_topk = sim_topk - sim_topk.max()
            weights = np.exp(sim_topk)
            weights = weights / (weights.sum() + 1e-10)
            
            if np.any(np.isnan(weights)):
                weights = np.ones(top_k) / top_k
            
            pred = (train_expr[top_k_idx].T @ weights).T  # (300,)
            
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                pred = train_expr.mean(axis=0)
            
            val_predictions.append(pred)
            
        except Exception as e:
            val_predictions.append(train_expr.mean(axis=0))
    
    val_pred = np.array(val_predictions)
    
    inference_end = datetime.now()
    inference_time = (inference_end - inference_start).total_seconds()
    
    print(f"\n✅ Validation inference complete!")
    print(f"   Time: {inference_time:.1f} seconds")
    print(f"   Val predictions shape: {val_pred.shape}")
    
    # Check for issues
    if np.any(np.isnan(val_pred)) or np.any(np.isinf(val_pred)):
        print("⚠️  WARNING: Predictions contain inf/nan, cleaning...")
        val_pred = np.nan_to_num(val_pred, nan=train_expr.mean(axis=0).mean())
    
    # Evaluate
    print("\n[7] Evaluating...")
    
    spot_corrs = []
    for i in range(len(val_pred)):
        try:
            if len(np.unique(val_pred[i])) > 1 and len(np.unique(val_gt[i])) > 1:
                if not np.any(np.isnan(val_pred[i])) and not np.any(np.isinf(val_pred[i])):
                    corr, _ = pearsonr(val_pred[i], val_gt[i])
                    if not np.isnan(corr):
                        spot_corrs.append(corr)
        except:
            continue
    
    gene_corrs = []
    for j in range(len(top_genes)):
        try:
            if len(np.unique(val_pred[:, j])) > 1 and len(np.unique(val_gt[:, j])) > 1:
                if not np.any(np.isnan(val_pred[:, j])) and not np.any(np.isinf(val_pred[:, j])):
                    corr, _ = pearsonr(val_pred[:, j], val_gt[:, j])
                    if not np.isnan(corr):
                        gene_corrs.append(corr)
        except:
            continue
    
    pred_vars = np.var(val_pred, axis=0)
    gt_vars = np.var(val_gt, axis=0)
    var_ratios = pred_vars / (gt_vars + 1e-8)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS - FULL TRAIN SET")
    print("="*80)
    print(f"Train samples: {len(train_embeds):,}")
    print(f"Val samples: {len(val_pred):,}")
    print(f"Top-k: {args.top_k}")
    print(f"Tau: {args.tau}")
    print()
    print(f"Spot-wise correlation: {np.mean(spot_corrs):.4f} ± {np.std(spot_corrs):.4f} (n={len(spot_corrs):,})")
    print(f"Gene-wise correlation: {np.mean(gene_corrs):.4f} ± {np.std(gene_corrs):.4f} (n={len(gene_corrs)})")
    print(f"Variance ratio: {np.mean(var_ratios):.4f}")
    print(f"Variance collapse: {(1 - np.mean(var_ratios))*100:.1f}%")
    print()
    print(f"Total time: {total_time/60:.1f} minutes")
    print("="*80)
    
    # Save results
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'fold': args.fold,
        'method': 'full_train_predex',
        'n_train_samples': len(train_embeds),
        'n_val_samples': len(val_pred),
        'top_k': args.top_k,
        'tau': args.tau,
        'spot_corr_mean': float(np.mean(spot_corrs)) if spot_corrs else 0.0,
        'spot_corr_std': float(np.std(spot_corrs)) if spot_corrs else 0.0,
        'n_spot_corrs': len(spot_corrs),
        'gene_corr_mean': float(np.mean(gene_corrs)) if gene_corrs else 0.0,
        'gene_corr_std': float(np.std(gene_corrs)) if gene_corrs else 0.0,
        'n_gene_corrs': len(gene_corrs),
        'var_ratio_mean': float(np.mean(var_ratios)),
        'var_collapse_pct': float((1 - np.mean(var_ratios)) * 100),
        'train_encode_time_min': encode_time / 60,
        'val_inference_time_sec': inference_time,
        'total_time_min': total_time / 60,
        'start_time': str(start_time),
        'end_time': str(end_time),
    }
    
    with open(output_dir / 'validation_results_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.save(output_dir / 'val_predictions_full.npy', val_pred)
    np.save(output_dir / 'val_ground_truth_full.npy', val_gt)
    
    # Gene analysis
    pd.DataFrame({
        'gene': top_genes,
        'correlation': gene_corrs if len(gene_corrs) == len(top_genes) else [np.nan]*len(top_genes),
        'pred_var': pred_vars,
        'gt_var': gt_vars,
        'var_ratio': var_ratios,
    }).to_csv(output_dir / 'gene_analysis_full.csv', index=False)
    
    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   - validation_results_full.json")
    print(f"   - val_predictions_full.npy")
    print(f"   - val_ground_truth_full.npy")
    print(f"   - gene_analysis_full.csv")
    
    if args.save_embeddings:
        print(f"   - train_embeddings.npy")
        print(f"   - train_expression.npy")

if __name__ == '__main__':
    main()

