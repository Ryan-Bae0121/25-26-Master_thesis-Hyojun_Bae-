#!/usr/bin/env python3
"""
Tau Experiment - Reuse saved train embeddings
No need to re-encode! Just change tau and re-run inference
python3 predex_tau_experiment.py \
    --fold 1 \
    --ckpt_path "/project_antwerp/hbae/Loki_output/finetune_10fold_runs/fold_01/finetune_fold_01_20260103_003308/checkpoints/epoch_latest.pt" \
    --train_csv "/project_antwerp/hbae/Loki_output/folds_10fold/fold_01_train_fixed.csv" \
    --val_csv "/project_antwerp/hbae/Loki_output/folds_10fold/fold_01_val_fixed.csv" \
    --combined_expr_npy "/project_antwerp/hbae/data/combined_expression_matrix.npy" \
    --top_genes_file "/project_antwerp/hbae/script/top300_genes.txt" \
    --saved_embeds_dir "/project_antwerp/hbae/Loki_output/st_val_results/fold_01/train_batches" \
    --tau_list 0.01 0.05 0.1 0.2 0.5 \
    --top_k 100 \
    --val_sample_size 1000 \
    --out_dir "/project_antwerp/hbae/Loki_output/tau_experiment/fold_01" \
    --device cuda
```

---

## 💡 **이 스크립트의 핵심:**
```
기존 방법:              새 방법:
tau=0.01 → 60분        tau=0.01 → 3분
tau=0.1  → 60분   →    tau=0.1  → 1분
tau=0.5  → 60분        tau=0.5  → 1분

총 300분               총 ~15분!
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

def load_saved_embeddings(embed_dir):
    """Load previously saved train embeddings"""
    embed_dir = Path(embed_dir)
    
    batch_files = sorted(embed_dir.glob('train_embeds_batch_*.npy'))
    
    if not batch_files:
        print(f"❌ No embeddings found in {embed_dir}")
        return None, None
    
    print(f"Found {len(batch_files)} batch files")
    
    all_embeds = []
    for bf in batch_files:
        batch = np.load(bf)
        all_embeds.append(batch)
        print(f"  Loaded {bf.name}: {batch.shape}")
    
    train_embeds = np.vstack(all_embeds)
    print(f"Total train embeddings: {train_embeds.shape}")
    
    # Load indices
    indices_file = embed_dir / 'train_indices_all.npy'
    if indices_file.exists():
        indices = np.load(indices_file)
    else:
        indices = np.arange(len(train_embeds))
    
    return train_embeds, indices

def run_inference(model, preprocess, val_df, train_embeds, train_expr, tau, top_k, device):
    """Run PredEx inference with given tau"""
    
    val_predictions = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Val (tau={tau})"):
        img_path = row['filepath']
        
        if not Path(img_path).exists():
            val_predictions.append(train_expr.mean(axis=0))
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                val_embed = model.encode_image(img_tensor)
                val_embed = val_embed / val_embed.norm(dim=-1, keepdim=True)
                val_embed_np = val_embed.cpu().numpy()
            
            # Similarity with all train embeddings
            sim = val_embed_np @ train_embeds.T  # (1, N_train)
            
            # Top-k
            top_k_actual = min(top_k, len(train_embeds))
            top_k_idx = np.argsort(sim[0])[-top_k_actual:]
            
            # Softmax with numerical stability
            sim_topk = sim[0, top_k_idx] / tau
            sim_topk = sim_topk - sim_topk.max()
            weights = np.exp(sim_topk)
            weights = weights / (weights.sum() + 1e-10)
            
            if np.any(np.isnan(weights)):
                weights = np.ones(top_k_actual) / top_k_actual
            
            pred = (train_expr[top_k_idx].T @ weights).T
            
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                pred = train_expr.mean(axis=0)
            
            val_predictions.append(pred)
            
        except Exception as e:
            val_predictions.append(train_expr.mean(axis=0))
    
    return np.array(val_predictions)

def evaluate(val_pred, val_gt, top_genes):
    """Evaluate predictions"""
    
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
    
    return {
        'spot_corr_mean': float(np.mean(spot_corrs)) if spot_corrs else 0.0,
        'spot_corr_std': float(np.std(spot_corrs)) if spot_corrs else 0.0,
        'gene_corr_mean': float(np.mean(gene_corrs)) if gene_corrs else 0.0,
        'gene_corr_std': float(np.std(gene_corrs)) if gene_corrs else 0.0,
        'var_ratio_mean': float(np.mean(var_ratios)),
        'n_spots': len(spot_corrs),
        'n_genes': len(gene_corrs),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--combined_expr_npy', type=str, required=True)
    parser.add_argument('--top_genes_file', type=str, required=True)
    parser.add_argument('--saved_embeds_dir', type=str, required=True,
                        help='Directory with saved train embeddings (train_batches/)')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Train CSV to get expression indices')
    parser.add_argument('--tau_list', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2, 0.5],
                        help='List of tau values to test')
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--val_sample_size', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TAU EXPERIMENT - Reusing Saved Embeddings")
    print("="*80)
    print(f"Tau values to test: {args.tau_list}")
    print(f"Val sample size: {args.val_sample_size}")
    print()
    
    # Load model (for val image encoding only)
    print("[1] Loading model...")
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
    
    # Sample val
    np.random.seed(42)
    val_sample_idx = np.random.choice(len(val_df), args.val_sample_size, replace=False)
    val_df_sample = val_df.iloc[val_sample_idx].reset_index(drop=True)
    
    # Get expression
    with open("/project_antwerp/hbae/data/all_shared_genes.txt") as fg:
        all_genes = [line.strip() for line in fg if line.strip()]
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    top_gene_indices = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
    train_indices = train_df['orig_row_idx'].values
    train_expr = expr_matrix[train_indices, :][:, top_gene_indices]
    
    val_indices = val_df_sample['orig_row_idx'].values
    val_gt = expr_matrix[val_indices, :][:, top_gene_indices]
    
    # Load saved train embeddings
    print(f"\n[3] Loading saved train embeddings...")
    print(f"From: {args.saved_embeds_dir}")
    
    train_embeds, success_indices = load_saved_embeddings(args.saved_embeds_dir)
    
    if train_embeds is None:
        print("❌ Failed to load embeddings!")
        return
    
    # Match expression to successful embeddings
    train_expr = train_expr[success_indices]
    print(f"Train expr matched: {train_expr.shape}")
    
    # Encode val images ONCE (reuse across tau experiments)
    print(f"\n[4] Encoding {args.val_sample_size} val images (once for all tau)...")
    
    val_embeds = []
    val_valid_mask = []
    
    for idx, row in tqdm(val_df_sample.iterrows(), total=len(val_df_sample), desc="Val encoding"):
        img_path = row['filepath']
        
        if not Path(img_path).exists():
            val_embeds.append(np.zeros(768))
            val_valid_mask.append(False)
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                embed = model.encode_image(img_tensor)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                val_embeds.append(embed.cpu().numpy()[0])
                val_valid_mask.append(True)
        except:
            val_embeds.append(np.zeros(768))
            val_valid_mask.append(False)
    
    val_embeds = np.array(val_embeds)  # (1000, 768)
    val_valid_mask = np.array(val_valid_mask)
    
    print(f"Val embeddings: {val_embeds.shape}")
    print(f"Valid: {val_valid_mask.sum()} / {len(val_valid_mask)}")
    
    # Pre-compute all similarities (once!)
    print(f"\n[5] Pre-computing similarities (train x val)...")
    sim_matrix = val_embeds @ train_embeds.T  # (1000, 139459)
    print(f"Similarity matrix: {sim_matrix.shape}")
    
    # Run experiments for each tau
    print(f"\n[6] Running tau experiments...")
    
    all_results = []
    
    for tau in args.tau_list:
        print(f"\n{'='*60}")
        print(f"TAU = {tau}")
        print('='*60)
        
        val_predictions = []
        top_k = min(args.top_k, len(train_embeds))
        
        for i in tqdm(range(len(val_df_sample)), desc=f"tau={tau}"):
            if not val_valid_mask[i]:
                val_predictions.append(train_expr.mean(axis=0))
                continue
            
            sim = sim_matrix[i]  # (139459,)
            
            # Top-k
            top_k_idx = np.argsort(sim)[-top_k:]
            
            # Softmax with stability
            sim_topk = sim[top_k_idx] / tau
            sim_topk = sim_topk - sim_topk.max()
            weights = np.exp(sim_topk)
            weights = weights / (weights.sum() + 1e-10)
            
            if np.any(np.isnan(weights)):
                weights = np.ones(top_k) / top_k
            
            pred = (train_expr[top_k_idx].T @ weights).T
            
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                pred = train_expr.mean(axis=0)
            
            val_predictions.append(pred)
        
        val_pred = np.array(val_predictions)
        
        # Evaluate
        metrics = evaluate(val_pred, val_gt, top_genes)
        metrics['tau'] = tau
        
        print(f"Spot corr: {metrics['spot_corr_mean']:.4f} ± {metrics['spot_corr_std']:.4f}")
        print(f"Gene corr: {metrics['gene_corr_mean']:.4f} ± {metrics['gene_corr_std']:.4f}")
        print(f"Var ratio: {metrics['var_ratio_mean']:.4f}")
        
        all_results.append(metrics)
        
        # Save per-tau predictions
        np.save(output_dir / f'val_pred_tau{tau}.npy', val_pred)
    
    # Summary
    print("\n" + "="*80)
    print("TAU EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'tau':<8} {'Spot Corr':<15} {'Gene Corr':<15} {'Var Ratio':<15}")
    print("-"*53)
    for r in all_results:
        print(f"{r['tau']:<8} "
              f"{r['spot_corr_mean']:.4f}±{r['spot_corr_std']:.4f}  "
              f"{r['gene_corr_mean']:.4f}±{r['gene_corr_std']:.4f}  "
              f"{r['var_ratio_mean']:.4f}")
    print("="*80)
    
    # Best tau
    best_spot = max(all_results, key=lambda x: x['spot_corr_mean'])
    best_gene = max(all_results, key=lambda x: x['gene_corr_mean'])
    best_var = min(all_results, key=lambda x: abs(x['var_ratio_mean'] - 1.0))
    
    print(f"\n✅ Best Spot Corr: tau={best_spot['tau']} ({best_spot['spot_corr_mean']:.4f})")
    print(f"✅ Best Gene Corr: tau={best_gene['tau']} ({best_gene['gene_corr_mean']:.4f})")
    print(f"✅ Best Var Ratio: tau={best_var['tau']} ({best_var['var_ratio_mean']:.4f}, closest to 1.0)")
    print("="*80)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'tau_experiment_results.csv', index=False)
    
    with open(output_dir / 'tau_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    np.save(output_dir / 'val_ground_truth.npy', val_gt)
    
    print(f"\n✅ All results saved to: {output_dir}")

if __name__ == '__main__':
    main()