#!/usr/bin/env python3
"""
Top-k Experiment Part 2 - Find Var ratio ≈ 1.0
Testing larger top_k values
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm
import json

OPENCLIP_ROOT = Path("/project_antwerp/hbae/open_clip2")
import sys
sys.path.insert(0, str(OPENCLIP_ROOT / "src"))
import open_clip

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
    gt_vars = np.var(val_gt, axis=0)
    var_ratios = pred_vars / (gt_vars + 1e-8)

    return {
        'spot_corr_mean': float(np.mean(spot_corrs)) if spot_corrs else 0.0,
        'spot_corr_std': float(np.std(spot_corrs)) if spot_corrs else 0.0,
        'gene_corr_mean': float(np.mean(gene_corrs)) if gene_corrs else 0.0,
        'gene_corr_std': float(np.std(gene_corrs)) if gene_corrs else 0.0,
        'var_ratio_mean': float(np.mean(var_ratios)),
    }

def main():
    # ============================================================
    # Settings - 이전보다 큰 top_k 값들
    # ============================================================
    TAU = 0.1
    TOPK_LIST = [10000, 20000, 30000, 50000, 70000, 100000]
    VAL_SAMPLE = 1000

    CKPT = "/project_antwerp/hbae/Loki_output/finetune_10fold_runs/fold_01/finetune_fold_01_20260103_003308/checkpoints/epoch_latest.pt"
    TRAIN_CSV = "/project_antwerp/hbae/Loki_output/folds_10fold/fold_01_train_fixed.csv"
    VAL_CSV = "/project_antwerp/hbae/Loki_output/folds_10fold/fold_01_val_fixed.csv"
    EXPR_NPY = "/project_antwerp/hbae/data/combined_expression_matrix.npy"
    GENES_FILE = "/project_antwerp/hbae/script/top300_genes.txt"
    EMBEDS_DIR = "/project_antwerp/hbae/Loki_output/st_val_results/fold_01/train_batches"
    OUT_DIR = Path("/project_antwerp/hbae/Loki_output/topk_experiment2/fold_01")
    DEVICE = "cuda"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TOP-K EXPERIMENT PART 2")
    print(f"Tau: {TAU} (fixed)")
    print(f"Top-k values: {TOPK_LIST}")
    print("="*80)

    # Load model
    print("\n[1] Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms('coca_ViT-L-14', pretrained=None)
    checkpoint = torch.load(CKPT, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    print("✅ Model loaded")

    # Load data
    print("\n[2] Loading data...")
    expr_matrix = np.load(EXPR_NPY, allow_pickle=True)
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    with open(GENES_FILE) as f:
        top_genes = [line.strip() for line in f if line.strip()]

    # Sample val (same seed as before!)
    np.random.seed(42)
    val_sample_idx = np.random.choice(len(val_df), VAL_SAMPLE, replace=False)
    val_df_sample = val_df.iloc[val_sample_idx].reset_index(drop=True)

    with open("/project_antwerp/hbae/data/all_shared_genes.txt") as fg:
        all_genes = [line.strip() for line in fg if line.strip()]
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    top_gene_indices = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
    train_indices = train_df['orig_row_idx'].values
    train_expr = expr_matrix[train_indices, :][:, top_gene_indices]
    val_indices = val_df_sample['orig_row_idx'].values
    val_gt = expr_matrix[val_indices, :][:, top_gene_indices]

    # Load saved train embeddings
    print("\n[3] Loading saved train embeddings...")
    batch_files = sorted(Path(EMBEDS_DIR).glob('train_embeds_batch_*.npy'))
    train_embeds = np.vstack([np.load(bf) for bf in batch_files])
    print(f"Train embeddings: {train_embeds.shape}")

    # Encode val images (once)
    print(f"\n[4] Encoding {VAL_SAMPLE} val images...")
    val_embeds = []
    for idx, row in tqdm(val_df_sample.iterrows(), total=len(val_df_sample)):
        img_path = row['filepath']
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embed = model.encode_image(img_tensor)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                val_embeds.append(embed.cpu().numpy()[0])
        except:
            val_embeds.append(np.zeros(768))
    val_embeds = np.array(val_embeds)

    # Compute similarity (once!)
    print("\n[5] Computing similarity matrix (once)...")
    sim_matrix = val_embeds @ train_embeds.T  # (1000, 139459)
    print(f"Similarity matrix: {sim_matrix.shape}")

    # Run experiments
    print("\n[6] Running top_k experiments...")
    all_results = []

    # Include previous best for reference
    prev_results = [
        {'top_k': 100,   'spot_corr_mean': 0.7251, 'spot_corr_std': 0.1256, 'gene_corr_mean': 0.0069, 'gene_corr_std': 0.0492, 'var_ratio_mean': 4.4444},
        {'top_k': 1000,  'spot_corr_mean': 0.7319, 'spot_corr_std': 0.1211, 'gene_corr_mean': 0.0113, 'gene_corr_std': 0.0556, 'var_ratio_mean': 3.6549},
        {'top_k': 10000, 'spot_corr_mean': 0.7382, 'spot_corr_std': 0.1164, 'gene_corr_mean': 0.0209, 'gene_corr_std': 0.0610, 'var_ratio_mean': 1.3689},
    ]

    for top_k in TOPK_LIST:
        print(f"\n{'='*60}")
        print(f"TOP_K = {top_k:,}")
        print('='*60)

        val_predictions = []

        for i in tqdm(range(VAL_SAMPLE), desc=f"top_k={top_k}"):
            sim = sim_matrix[i]

            top_k_actual = min(top_k, len(train_embeds))
            top_k_idx = np.argsort(sim)[-top_k_actual:]

            sim_topk = sim[top_k_idx] / TAU
            sim_topk = sim_topk - sim_topk.max()
            weights = np.exp(sim_topk)
            weights = weights / (weights.sum() + 1e-10)

            if np.any(np.isnan(weights)):
                weights = np.ones(top_k_actual) / top_k_actual

            pred = (train_expr[top_k_idx].T @ weights).T

            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                pred = train_expr.mean(axis=0)

            val_predictions.append(pred)

        val_pred = np.array(val_predictions)
        metrics = evaluate(val_pred, val_gt, top_genes)
        metrics['top_k'] = top_k
        metrics['tau'] = TAU

        print(f"Spot corr: {metrics['spot_corr_mean']:.4f} ± {metrics['spot_corr_std']:.4f}")
        print(f"Gene corr: {metrics['gene_corr_mean']:.4f} ± {metrics['gene_corr_std']:.4f}")
        print(f"Var ratio: {metrics['var_ratio_mean']:.4f}")

        all_results.append(metrics)

    # Full summary (이전 결과 포함)
    print("\n" + "="*80)
    print("FULL SUMMARY (tau=0.1, all top_k)")
    print("="*80)
    print(f"{'top_k':<12} {'Spot Corr':<22} {'Gene Corr':<22} {'Var Ratio':<12} {'상태'}")
    print("-"*80)

    for r in prev_results:
        dist = abs(r['var_ratio_mean'] - 1.0)
        status = "⭐ 최적!" if dist < 0.3 else ("🟢 양호" if dist < 1.0 else ("🟡 과함" if r['var_ratio_mean'] > 1.0 else "🔴 붕괴"))
        print(f"{r['top_k']:<12} "
              f"{r['spot_corr_mean']:.4f}±{r['spot_corr_std']:.4f}        "
              f"{r['gene_corr_mean']:.4f}±{r['gene_corr_std']:.4f}        "
              f"{r['var_ratio_mean']:<12.4f} {status}")

    for r in all_results:
        dist = abs(r['var_ratio_mean'] - 1.0)
        status = "⭐ 최적!" if dist < 0.3 else ("🟢 양호" if dist < 1.0 else ("🟡 과함" if r['var_ratio_mean'] > 1.0 else "🔴 붕괴"))
        print(f"{r['top_k']:<12} "
              f"{r['spot_corr_mean']:.4f}±{r['spot_corr_std']:.4f}        "
              f"{r['gene_corr_mean']:.4f}±{r['gene_corr_std']:.4f}        "
              f"{r['var_ratio_mean']:<12.4f} {status}")

    print("="*80)

    best_var = min(all_results, key=lambda x: abs(x['var_ratio_mean'] - 1.0))
    best_spot = max(all_results, key=lambda x: x['spot_corr_mean'])

    print(f"\n✅ Var ratio 1.0에 가장 가까운: top_k={best_var['top_k']:,} (ratio={best_var['var_ratio_mean']:.4f})")
    print(f"✅ Best Spot Corr: top_k={best_spot['top_k']:,} ({best_spot['spot_corr_mean']:.4f})")
    print("="*80)

    # Save
    pd.DataFrame(all_results).to_csv(OUT_DIR / 'topk_experiment2_results.csv', index=False)
    with open(OUT_DIR / 'topk_experiment2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Saved to: {OUT_DIR}")

if __name__ == '__main__':
    main()
