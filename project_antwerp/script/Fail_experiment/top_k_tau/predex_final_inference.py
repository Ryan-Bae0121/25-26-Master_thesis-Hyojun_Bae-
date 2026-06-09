#!/usr/bin/env python3
"""
Final ST Validation Inference
- tau=0.1, top_k=20000
- Batch encoding for memory efficiency
- Reuses saved embeddings if available
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

def encode_in_batches(model, preprocess, train_df, device, batch_size, out_dir):
    """Encode train images in batches, save to disk"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_batches = (len(train_df) + batch_size - 1) // batch_size
    all_success_indices = []

    for batch_idx in range(n_batches):
        batch_file = out_dir / f'train_embeds_batch_{batch_idx:03d}.npy'

        # Skip if already exists
        if batch_file.exists():
            indices_file = out_dir / f'train_indices_batch_{batch_idx:03d}.npy'
            if indices_file.exists():
                batch_indices = np.load(indices_file).tolist()
                all_success_indices.extend(batch_indices)
                print(f"  Batch {batch_idx+1}/{n_batches}: already exists, skipping")
                continue

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_df))

        batch_embeds = []
        batch_indices = []

        for idx in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx+1}/{n_batches}"):
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
                    batch_indices.append(idx)
            except:
                continue

        if batch_embeds:
            batch_embeds = np.vstack(batch_embeds)
            np.save(batch_file, batch_embeds)
            np.save(out_dir / f'train_indices_batch_{batch_idx:03d}.npy', batch_indices)
            all_success_indices.extend(batch_indices)
            print(f"  Batch {batch_idx+1}/{n_batches}: saved {len(batch_embeds):,} embeddings")

            del batch_embeds
            torch.cuda.empty_cache()

    np.save(out_dir / 'train_indices_all.npy', all_success_indices)
    return all_success_indices

def load_embeddings(embed_dir):
    """Load all saved embeddings"""
    embed_dir = Path(embed_dir)
    batch_files = sorted(embed_dir.glob('train_embeds_batch_*.npy'))
    all_embeds = [np.load(bf) for bf in batch_files]
    return np.vstack(all_embeds)

def evaluate(val_pred, val_gt, top_genes):
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
    gt_vars   = np.var(val_gt, axis=0)
    var_ratios = pred_vars / (gt_vars + 1e-8)

    return {
        'spot_corr_mean': float(np.mean(spot_corrs)) if spot_corrs else 0.0,
        'spot_corr_std':  float(np.std(spot_corrs))  if spot_corrs else 0.0,
        'gene_corr_mean': float(np.mean(gene_corrs)) if gene_corrs else 0.0,
        'gene_corr_std':  float(np.std(gene_corrs))  if gene_corrs else 0.0,
        'var_ratio_mean': float(np.mean(var_ratios)),
        'n_spots': len(spot_corrs),
        'n_genes': len(gene_corrs),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--combined_expr_npy', type=str, required=True)
    parser.add_argument('--top_genes_file', type=str, required=True)
    parser.add_argument('--saved_embeds_dir', type=str, default=None)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--val_sample_size', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    start_time = datetime.now()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embed_dir = out_dir / 'train_batches'

    print("="*70)
    print(f"ST Validation - Fold {args.fold:02d}")
    print(f"tau={args.tau}, top_k={args.top_k:,}, batch={args.batch_size:,}")
    print("="*70)

    # Load model
    print("\n[1] Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms('coca_ViT-L-14', pretrained=None)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    model.eval()
    print("✅ Model loaded")

    # Load data
    print("\n[2] Loading data...")
    expr_matrix = np.load(args.combined_expr_npy, allow_pickle=True)
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    with open(args.top_genes_file) as f:
        top_genes = [line.strip() for line in f if line.strip()]

    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Genes: {len(top_genes)}")

    # Sample val
    np.random.seed(42)
    val_sample_idx = np.random.choice(len(val_df), args.val_sample_size, replace=False)
    val_df_sample  = val_df.iloc[val_sample_idx].reset_index(drop=True)

    with open("/project_antwerp/hbae/data/all_shared_genes.txt") as fg:
        all_genes = [line.strip() for line in fg if line.strip()]
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    top_gene_indices = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
    train_indices    = train_df['orig_row_idx'].values
    train_expr       = expr_matrix[train_indices, :][:, top_gene_indices]
    val_indices      = val_df_sample['orig_row_idx'].values
    val_gt           = expr_matrix[val_indices, :][:, top_gene_indices]

    # Encode train (or reuse)
    saved_dir = args.saved_embeds_dir if args.saved_embeds_dir else str(embed_dir)

    if args.saved_embeds_dir and Path(args.saved_embeds_dir).exists():
        print(f"\n[3] Loading saved embeddings from {saved_dir}...")
        train_embeds = load_embeddings(saved_dir)
        success_indices = np.load(Path(saved_dir) / 'train_indices_all.npy').tolist()
    else:
        print(f"\n[3] Encoding {len(train_df):,} train images in batches...")
        success_indices = encode_in_batches(
            model, preprocess, train_df, args.device, args.batch_size, embed_dir
        )
        train_embeds = load_embeddings(embed_dir)

    train_expr = train_expr[success_indices]
    print(f"Train embeddings: {train_embeds.shape}")
    print(f"Train expression: {train_expr.shape}")

    # Encode val images
    print(f"\n[4] Encoding {args.val_sample_size:,} val images...")
    val_embeds = []
    for idx, row in tqdm(val_df_sample.iterrows(), total=len(val_df_sample)):
        img_path = row['filepath']
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(args.device)
            with torch.no_grad():
                embed = model.encode_image(img_tensor)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                val_embeds.append(embed.cpu().numpy()[0])
        except:
            val_embeds.append(np.zeros(768))
    val_embeds = np.array(val_embeds)

    # Compute similarity
    print("\n[5] Computing similarity matrix...")
    sim_matrix = val_embeds @ train_embeds.T  # (val_sample, N_train)
    print(f"Similarity matrix: {sim_matrix.shape}")

    # Inference
    print(f"\n[6] Running inference (top_k={args.top_k:,})...")
    val_predictions = []
    top_k = min(args.top_k, len(train_embeds))

    for i in tqdm(range(args.val_sample_size)):
        sim = sim_matrix[i]
        top_k_idx = np.argsort(sim)[-top_k:]

        sim_topk  = sim[top_k_idx] / args.tau
        sim_topk  = sim_topk - sim_topk.max()
        weights   = np.exp(sim_topk)
        weights   = weights / (weights.sum() + 1e-10)

        if np.any(np.isnan(weights)):
            weights = np.ones(top_k) / top_k

        pred = (train_expr[top_k_idx].T @ weights).T

        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            pred = train_expr.mean(axis=0)

        val_predictions.append(pred)

    val_pred = np.array(val_predictions)

    # Evaluate
    print("\n[7] Evaluating...")
    metrics = evaluate(val_pred, val_gt, top_genes)
    metrics['fold'] = args.fold
    metrics['tau']  = args.tau
    metrics['top_k'] = args.top_k
    metrics['total_time_min'] = (datetime.now() - start_time).total_seconds() / 60

    print("\n" + "="*70)
    print(f"RESULTS - Fold {args.fold:02d}")
    print("="*70)
    print(f"Spot corr: {metrics['spot_corr_mean']:.4f} ± {metrics['spot_corr_std']:.4f}")
    print(f"Gene corr: {metrics['gene_corr_mean']:.4f} ± {metrics['gene_corr_std']:.4f}")
    print(f"Var ratio: {metrics['var_ratio_mean']:.4f}")
    print(f"Time:      {metrics['total_time_min']:.1f} min")
    print("="*70)

    # Save
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    np.save(out_dir / 'val_pred.npy', val_pred)
    np.save(out_dir / 'val_gt.npy', val_gt)
    print(f"\n✅ Saved to: {out_dir}")

if __name__ == '__main__':
    main()
