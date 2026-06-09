#!/usr/bin/env python3
"""
ST Validation - FULL TRAIN with LOW MEMORY
Save embeddings to disk in batches
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

def encode_train_batches(model, preprocess, train_df, device, batch_size=10000, output_dir=None):
    """Encode train images in batches and save to disk"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_batches = (len(train_df) + batch_size - 1) // batch_size
    
    print(f"Encoding {len(train_df):,} images in {n_batches} batches of {batch_size:,}")
    
    all_success_indices = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_df))
        
        print(f"\nBatch {batch_idx+1}/{n_batches}: [{start_idx:,} - {end_idx:,}]")
        
        batch_embeds = []
        batch_success_indices = []
        
        for idx in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx+1}"):
            row = train_df.iloc[idx]
            img_path = row['filepath']
            
            if not Path(img_path).exists():
                continue
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    embed = model.encode_image(img_tensor)
                    embed = embed / embed.norm(dim=-1, keepdim=True)
                    batch_embeds.append(embed.cpu().numpy())
                    batch_success_indices.append(idx)
            except:
                continue
        
        # Save batch to disk
        if batch_embeds:
            batch_embeds = np.vstack(batch_embeds)
            batch_file = output_dir / f'train_embeds_batch_{batch_idx:03d}.npy'
            np.save(batch_file, batch_embeds)
            
            indices_file = output_dir / f'train_indices_batch_{batch_idx:03d}.npy'
            np.save(indices_file, batch_success_indices)
            
            all_success_indices.extend(batch_success_indices)
            
            print(f"  Saved {len(batch_embeds):,} embeddings to {batch_file.name}")
            
            # Clear memory
            del batch_embeds
            torch.cuda.empty_cache()
    
    # Save combined index
    np.save(output_dir / 'train_indices_all.npy', all_success_indices)
    
    return all_success_indices, n_batches

def load_train_embeddings(output_dir, n_batches):
    """Load train embeddings from disk batches"""
    
    output_dir = Path(output_dir)
    
    print(f"Loading {n_batches} batches of embeddings...")
    
    all_embeds = []
    for batch_idx in range(n_batches):
        batch_file = output_dir / f'train_embeds_batch_{batch_idx:03d}.npy'
        if batch_file.exists():
            batch = np.load(batch_file)
            all_embeds.append(batch)
            print(f"  Loaded batch {batch_idx}: {batch.shape}")
    
    train_embeds = np.vstack(all_embeds)
    print(f"Total embeddings: {train_embeds.shape}")
    
    return train_embeds

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
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--val_sample_size', type=int, default=1000, help='Sample val for speed')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ST Validation - FULL TRAIN (LOW MEMORY)")
    print("="*80)
    print(f"Start: {start_time}")
    
    # Load model
    print("\n[1] Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', pretrained=None
    )
    
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    model.eval()
    print(f"✅ Model loaded")
    
    # Load data
    print("\n[2] Loading data...")
    expr_matrix = np.load(args.combined_expr_npy, allow_pickle=True)
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    with open(args.top_genes_file) as f:
        top_genes = [line.strip() for line in f if line.strip()]
    
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Genes: {len(top_genes)}")
    
    # Sample val
    if args.val_sample_size and args.val_sample_size < len(val_df):
        print(f"\n[3] Sampling {args.val_sample_size:,} val spots...")
        np.random.seed(42)
        val_sample_idx = np.random.choice(len(val_df), args.val_sample_size, replace=False)
        val_df = val_df.iloc[val_sample_idx].reset_index(drop=True)
    
    # Get expression
    top_gene_indices = np.arange(len(top_genes))
    train_indices = train_df['orig_row_idx'].values
    train_expr = expr_matrix[train_indices, :][:, top_gene_indices]
    
    val_indices = val_df['orig_row_idx'].values
    val_gt = expr_matrix[val_indices, :][:, top_gene_indices]
    
    # Encode train in batches
    print(f"\n[4] Encoding train images in batches...")
    success_indices, n_batches = encode_train_batches(
        model, preprocess, train_df, args.device, 
        args.batch_size, output_dir / 'train_batches'
    )
    
    # Load all embeddings
    print(f"\n[5] Loading all embeddings...")
    train_embeds = load_train_embeddings(output_dir / 'train_batches', n_batches)
    train_expr = train_expr[success_indices]
    
    print(f"Final: {train_embeds.shape}, {train_expr.shape}")
    
    # Val inference
    print(f"\n[6] Val inference on {len(val_df):,} spots...")
    val_predictions = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val"):
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
                val_embed_np = val_embed.cpu().numpy()
            
            sim = val_embed_np @ train_embeds.T
            top_k = min(args.top_k, len(train_embeds))
            top_k_idx = np.argsort(sim[0])[-top_k:]
            
            sim_topk = sim[0, top_k_idx] / args.tau
            sim_topk = sim_topk - sim_topk.max()
            weights = np.exp(sim_topk)
            weights = weights / (weights.sum() + 1e-10)
            
            if np.any(np.isnan(weights)):
                weights = np.ones(top_k) / top_k
            
            pred = (train_expr[top_k_idx].T @ weights).T
            
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                pred = train_expr.mean(axis=0)
            
            val_predictions.append(pred)
        except:
            val_predictions.append(train_expr.mean(axis=0))
    
    val_pred = np.array(val_predictions)
    
    # Evaluate
    print("\n[7] Evaluating...")
    
    spot_corrs = []
    for i in range(len(val_pred)):
        try:
            if len(np.unique(val_pred[i])) > 1 and len(np.unique(val_gt[i])) > 1:
                corr, _ = pearsonr(val_pred[i], val_gt[i])
                if not np.isnan(corr):
                    spot_corrs.append(corr)
        except:
            continue
    
    gene_corrs = []
    for j in range(len(top_genes)):
        try:
            if len(np.unique(val_pred[:, j])) > 1 and len(np.unique(val_gt[:, j])) > 1:
                corr, _ = pearsonr(val_pred[:, j], val_gt[:, j])
                if not np.isnan(corr):
                    gene_corrs.append(corr)
        except:
            continue
    
    pred_vars = np.var(val_pred, axis=0)
    gt_vars = np.var(val_gt, axis=0)
    var_ratios = pred_vars / (gt_vars + 1e-8)
    
    end_time = datetime.now()
    
    print("\n" + "="*80)
    print("RESULTS - FULL TRAIN")
    print("="*80)
    print(f"Train: {len(train_embeds):,}, Val: {len(val_pred):,}")
    print(f"Spot corr: {np.mean(spot_corrs):.4f} ± {np.std(spot_corrs):.4f} (n={len(spot_corrs)})")
    print(f"Gene corr: {np.mean(gene_corrs):.4f} ± {np.std(gene_corrs):.4f} (n={len(gene_corrs)})")
    print(f"Var ratio: {np.mean(var_ratios):.4f}")
    print(f"Time: {(end_time-start_time).total_seconds()/60:.1f} min")
    print("="*80)
    
    # Save
    results = {
        'fold': args.fold,
        'n_train': len(train_embeds),
        'n_val': len(val_pred),
        'spot_corr_mean': float(np.mean(spot_corrs)),
        'gene_corr_mean': float(np.mean(gene_corrs)),
        'var_ratio': float(np.mean(var_ratios)),
    }
    
    with open(output_dir / 'results_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    np.save(output_dir / 'val_pred.npy', val_pred)
    np.save(output_dir / 'val_gt.npy', val_gt)
    
    print(f"\n✅ Saved to {output_dir}")

if __name__ == '__main__':
    main()

