#!/usr/bin/env python3
"""
ST Validation Inference - FIXED overflow issue
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--combined_expr_npy', type=str, required=True)
    parser.add_argument('--top_genes_file', type=str, required=True)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--n_train_samples', type=int, default=5000)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_sample_size', type=int, default=1000)
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"ST Validation Inference (SAMPLED) - Fold {args.fold}")
    print("="*80)
    
    # Load model
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
    
    # Load data
    print("\n[2] Loading data...")
    expr_matrix = np.load(args.combined_expr_npy, allow_pickle=True)
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    with open(args.top_genes_file) as f:
        top_genes = [line.strip() for line in f if line.strip()]
    
    print(f"Expression matrix: {expr_matrix.shape}")
    print(f"Train spots: {len(train_df)}")
    print(f"Val spots: {len(val_df)}")
    print(f"Top genes: {len(top_genes)}")
    
    # Sample train
    print(f"\n[3] Sampling {args.n_train_samples} train spots...")
    np.random.seed(42)
    train_sample_idx = np.random.choice(len(train_df), args.n_train_samples, replace=False)
    train_df_sample = train_df.iloc[train_sample_idx].reset_index(drop=True)
    
    # Sample val
    print(f"[4] Sampling {args.val_sample_size} val spots...")
    val_sample_idx = np.random.choice(len(val_df), args.val_sample_size, replace=False)
    val_df_sample = val_df.iloc[val_sample_idx].reset_index(drop=True)
    
    # Get expression
    top_gene_indices = np.arange(len(top_genes))
    
    train_indices_sample = train_df_sample['orig_row_idx'].values
    train_expr_sample = expr_matrix[train_indices_sample, :][:, top_gene_indices]
    
    val_indices_sample = val_df_sample['orig_row_idx'].values
    val_gt_sample = expr_matrix[val_indices_sample, :][:, top_gene_indices]
    
    print(f"Train expression (sampled): {train_expr_sample.shape}")
    print(f"Val ground truth (sampled): {val_gt_sample.shape}")
    
    # Encode train images
    print(f"\n[5] Encoding {args.n_train_samples} train images...")
    train_embeds = []
    
    for idx, row in tqdm(train_df_sample.iterrows(), total=len(train_df_sample), desc="Train"):
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
        except Exception as e:
            continue
    
    train_embeds = np.vstack(train_embeds)  # (N, 768)
    print(f"Train embeddings: {train_embeds.shape}")
    
    # Match expression
    train_expr_sample = train_expr_sample[:len(train_embeds)]
    
    # Val inference
    print(f"\n[6] Running val inference on {args.val_sample_size} spots...")
    val_predictions = []
    
    for idx, row in tqdm(val_df_sample.iterrows(), total=len(val_df_sample), desc="Val"):
        img_path = row['filepath']
        
        if not Path(img_path).exists():
            val_predictions.append(train_expr_sample.mean(axis=0))
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
            
            # FIXED: Use softmax with numerical stability
            sim_topk = sim[0, top_k_idx] / args.tau
            # Subtract max for numerical stability
            sim_topk = sim_topk - sim_topk.max()
            weights = np.exp(sim_topk)
            weights = weights / (weights.sum() + 1e-10)
            
            # Check for NaN
            if np.any(np.isnan(weights)):
                print(f"⚠️  NaN weights detected, using uniform")
                weights = np.ones(top_k) / top_k
            
            pred = (train_expr_sample[top_k_idx].T @ weights).T  # (300,)
            
            # Check for inf/nan
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                print(f"⚠️  Invalid prediction, using mean")
                pred = train_expr_sample.mean(axis=0)
            
            val_predictions.append(pred)
            
        except Exception as e:
            val_predictions.append(train_expr_sample.mean(axis=0))
    
    val_pred = np.array(val_predictions)
    print(f"Val predictions: {val_pred.shape}")
    
    # Check for inf/nan in predictions
    if np.any(np.isnan(val_pred)) or np.any(np.isinf(val_pred)):
        print("⚠️  WARNING: Predictions contain inf/nan!")
        print(f"   NaN count: {np.isnan(val_pred).sum()}")
        print(f"   Inf count: {np.isinf(val_pred).sum()}")
        # Replace with mean
        val_pred = np.nan_to_num(val_pred, nan=train_expr_sample.mean(axis=0).mean())
    
    # Evaluate
    print("\n[7] Evaluating...")
    
    spot_corrs = []
    for i in range(len(val_pred)):
        try:
            if len(np.unique(val_pred[i])) > 1 and len(np.unique(val_gt_sample[i])) > 1:
                if not np.any(np.isnan(val_pred[i])) and not np.any(np.isinf(val_pred[i])):
                    corr, _ = pearsonr(val_pred[i], val_gt_sample[i])
                    if not np.isnan(corr):
                        spot_corrs.append(corr)
        except:
            continue
    
    gene_corrs = []
    for j in range(len(top_genes)):
        try:
            if len(np.unique(val_pred[:, j])) > 1 and len(np.unique(val_gt_sample[:, j])) > 1:
                if not np.any(np.isnan(val_pred[:, j])) and not np.any(np.isinf(val_pred[:, j])):
                    corr, _ = pearsonr(val_pred[:, j], val_gt_sample[:, j])
                    if not np.isnan(corr):
                        gene_corrs.append(corr)
        except:
            continue
    
    pred_vars = np.var(val_pred, axis=0)
    gt_vars = np.var(val_gt_sample, axis=0)
    var_ratios = pred_vars / (gt_vars + 1e-8)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS (SAMPLED)")
    print("="*80)
    print(f"Train samples: {len(train_embeds)}")
    print(f"Val samples: {len(val_pred)}")
    print(f"Top-k: {args.top_k}")
    print(f"Tau: {args.tau}")
    print()
    print(f"Spot-wise correlation: {np.mean(spot_corrs):.4f} ± {np.std(spot_corrs):.4f} (n={len(spot_corrs)})")
    print(f"Gene-wise correlation: {np.mean(gene_corrs):.4f} ± {np.std(gene_corrs):.4f} (n={len(gene_corrs)})")
    print(f"Variance ratio: {np.mean(var_ratios):.4f}")
    print(f"Variance collapse: {(1 - np.mean(var_ratios))*100:.1f}%")
    print("="*80)
    
    # Save
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'fold': args.fold,
        'method': 'sampled_predex',
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
    }
    
    import json
    with open(output_dir / 'validation_results_sampled.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved to: {output_dir}")

if __name__ == '__main__':
    main()

