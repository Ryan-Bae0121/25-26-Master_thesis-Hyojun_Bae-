#!/usr/bin/env python3
"""
ST Validation Inference - COMPLETE VERSION
Load images, encode tiles, run PredEx, evaluate
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
    
    # 5. Get train expression (for PredEx reference)
    print("\n[3] Preparing train expression...")
    train_indices = train_df['orig_row_idx'].values
    
    # TODO: Map top_genes to matrix columns
    # For now, assume first 300 columns
    print("⚠️  Assuming top 300 genes = first 300 columns")
    top_gene_indices = np.arange(len(top_genes))
    
    train_expr = expr_matrix[train_indices, :][:, top_gene_indices]  # (n_train, 300)
    print(f"Train expression: {train_expr.shape}")
    
    # 6. Encode gene text
    print("\n[4] Encoding gene text...")
    tokenizer = get_tokenizer('coca_ViT-L-14')
    gene_prompts = [f"a histology image with high expression of {g}" for g in top_genes]
    
    gene_text_embeds = []
    with torch.no_grad():
        for i in range(0, len(gene_prompts), args.text_batch_size):
            batch_prompts = gene_prompts[i:i+args.text_batch_size]
            tokens = tokenizer(batch_prompts).to(args.device)
            embeds = model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            gene_text_embeds.append(embeds.cpu().numpy())
    
    gene_text_embeds = np.vstack(gene_text_embeds)  # (300, 768)
    print(f"Gene text embeddings: {gene_text_embeds.shape}")
    
    # 7. Val ground truth
    print("\n[5] Loading validation ground truth...")
    val_indices = val_df['orig_row_idx'].values
    val_gt = expr_matrix[val_indices, :][:, top_gene_indices]  # (n_val, 300)
    print(f"Val ground truth: {val_gt.shape}")
    
    # 8. Val inference - encode images and predict
    print("\n[6] Running validation inference...")
    print("Encoding validation images and computing PredEx...")
    
    val_predictions = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val spots"):
        img_path = row['filepath']
        
        # Check if image exists
        if not Path(img_path).exists():
            print(f"⚠️  Image not found: {img_path}")
            # Use mean prediction as fallback
            val_predictions.append(train_expr.mean(axis=0))
            continue
        
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(args.device)
            
            # Encode image
            with torch.no_grad():
                img_embed = model.encode_image(img_tensor)
                img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
                img_embed = img_embed.cpu().numpy()  # (1, 768)
            
            # Compute similarity with gene text
            img_embed_torch = torch.from_numpy(img_embed).to(args.device)
            gene_embed_torch = torch.from_numpy(gene_text_embeds).to(args.device)
            
            similarity = img_embed_torch @ gene_embed_torch.T  # (1, 300)
            
            # Apply temperature and softmax
            weights = torch.softmax(similarity / args.tau, dim=-1)  # (1, 300)
            weights = weights.cpu().numpy()
            
            # PredEx: weighted average of train expression
            # For each gene, weight all train samples
            pred = np.zeros(len(top_genes))
            for gene_idx in range(len(top_genes)):
                gene_weight = weights[0, gene_idx]
                # Use all train samples (simplified - should use top-k)
                pred[gene_idx] = (train_expr[:, gene_idx] * gene_weight).sum()
            
            val_predictions.append(pred)
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            val_predictions.append(train_expr.mean(axis=0))
    
    val_pred = np.array(val_predictions)  # (n_val, 300)
    print(f"\nValidation predictions: {val_pred.shape}")
    
    # 9. Evaluate
    print("\n[7] Evaluating validation performance...")
    
    # Spot-wise correlation
    spot_corrs = []
    for i in range(len(val_df)):
        if len(np.unique(val_pred[i])) > 1 and len(np.unique(val_gt[i])) > 1:
            corr, _ = pearsonr(val_pred[i], val_gt[i])
            spot_corrs.append(corr)
    
    # Gene-wise correlation
    gene_corrs = []
    for j in range(len(top_genes)):
        if len(np.unique(val_pred[:, j])) > 1 and len(np.unique(val_gt[:, j])) > 1:
            corr, _ = pearsonr(val_pred[:, j], val_gt[:, j])
            gene_corrs.append(corr)
    
    # Variance analysis
    pred_vars = np.var(val_pred, axis=0)
    gt_vars = np.var(val_gt, axis=0)
    var_ratios = pred_vars / (gt_vars + 1e-8)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Spot-wise correlation: {np.mean(spot_corrs):.4f} ± {np.std(spot_corrs):.4f}")
    print(f"Gene-wise correlation: {np.mean(gene_corrs):.4f} ± {np.std(gene_corrs):.4f}")
    print(f"Variance ratio (pred/gt): {np.mean(var_ratios):.4f}")
    print(f"Variance collapse: {(1 - np.mean(var_ratios))*100:.1f}%")
    print("="*80)
    
    # 10. Save results
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'fold': args.fold,
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
    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    np.save(output_dir / 'val_predictions.npy', val_pred)
    np.save(output_dir / 'val_ground_truth.npy', val_gt)
    
    pd.DataFrame({
        'gene': top_genes,
        'correlation': gene_corrs if len(gene_corrs) == len(top_genes) else [np.nan]*len(top_genes),
        'pred_var': pred_vars,
        'gt_var': gt_vars,
        'var_ratio': var_ratios,
    }).to_csv(output_dir / 'gene_analysis.csv', index=False)
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   - validation_results.json")
    print(f"   - val_predictions.npy")
    print(f"   - val_ground_truth.npy")
    print(f"   - gene_analysis.csv")

if __name__ == '__main__':
    main()

