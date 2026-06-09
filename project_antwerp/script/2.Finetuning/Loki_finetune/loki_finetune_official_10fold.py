#!/usr/bin/env python3
"""
loki_finetune_official_10fold.py
=================================
Loki 논문 공식 Fine-tuning 방법으로 10-fold CV

Fine-tuning 방법:
- Contrastive Loss (영상-텍스트 대조 학습)
- Image encoder + Text encoder 둘 다 학습
- Batch size: 64
- Epochs: 10 (권장)
- Temperature: 0.07

Usage:
    python loki_finetune_official_10fold.py \
        --folds_dir /path/to/folds_10fold_hvg_predex \
        --output_dir ./loki_finetune_10fold \
        --epochs 10 \
        --batch_size 64 \
        --device cuda:0
        
    python loki_finetune_official_10fold.py \
        --folds_dir /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold \
        --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
        --output_dir project_antwerp/hbae/Loki_output/0228_loki_finetune_10fold \
        --epochs 10 \
        --batch_size 64 \
        --device cuda:0 \
        --start_fold 3 \
        --end_fold 3
        
    python loki_finetune_official_10fold.py \
        --folds_dir /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold \
        --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
        --output_dir project_antwerp/hbae/Loki_output/0228_loki_finetune_10fold \
        --epochs 10 \
        --batch_size 64 \
        --device cuda:0 \
        --start_fold 3 \
        --end_fold 3

    python loki_finetune_official_10fold.py \
        --folds_dir /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_10fold \
        --hvg_file /project_antwerp/hbae/data/0317_training_data_excluding_GSE220978_and_19h1257/HVG_2000_genes.txt \
        --output_dir /project_antwerp/hbae/Loki_output/0317_loki_finetune_10fold_remove_patient \
        --epochs 10 \
        --batch_size 64 \
        --start_fold 1 \
        --end_fold 2 \
        --device cuda:0
        
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm


class STDataset(Dataset):
    """ST 데이터셋 (이미지 + 유전자 문장)"""
    
    def __init__(self, csv_file, preprocess, tokenizer, gt_expr, gt_obs, 
                 hvg_indices, obs_to_idx, gene_to_idx):
        self.df = pd.read_csv(csv_file)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.gt_expr = gt_expr
        self.hvg_indices = hvg_indices
        self.obs_to_idx = obs_to_idx
        
        # Extract obs_key
        def extract_obs_key(filepath):
            parts = filepath.split('/')
            try:
                idx = parts.index('patches')
                sample_id = parts[idx - 1]  # Full sample ID without splitting
                barcode = parts[-1].replace('.png', '')
                return f"{sample_id}_{barcode}_hires"
            except ValueError:
                # Fallback for non-standard paths
                gsm = [p for p in parts if p.startswith('GSM')]
                sample_id = gsm[0] if gsm else parts[-3]
                barcode = parts[-1].replace('.png', '')
                return f"{sample_id}_{barcode}_hires"
        
        self.df['obs_key'] = self.df['img_path'].apply(extract_obs_key)
        
        # Filter valid spots
        self.df = self.df[self.df['obs_key'].isin(obs_to_idx)].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        try:
            img = Image.open(row['img_path']).convert('RGB')
            img_tensor = self.preprocess(img)
        except:
            img_tensor = torch.zeros(3, 224, 224)
        
        # Gene sentence
        gene_sentence = row['label']
        
        # Ground truth expression (for evaluation only)
        spot_idx = self.obs_to_idx[row['obs_key']]
        gt_expr = self.gt_expr[spot_idx, self.hvg_indices]
        
        return {
            'image': img_tensor,
            'text': gene_sentence,
            'gt_expr': torch.from_numpy(gt_expr).float()
        }


def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    Loki 공식 Contrastive Loss
    
    L_con = -1/N * (Σ log(exp(x_i^T y_i / σ) / Σ exp(x_i^T y_j / σ))
                    + Σ log(exp(y_i^T x_i / σ) / Σ exp(y_i^T x_j / σ)))
    """
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Compute similarity matrix
    logits = image_embeds @ text_embeds.T / temperature  # (N, N)
    
    # Labels (diagonal)
    labels = torch.arange(len(image_embeds), device=image_embeds.device)
    
    # Image-to-text loss
    loss_i2t = F.cross_entropy(logits, labels)
    
    # Text-to-image loss
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    # Total loss
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss


def load_omiclip(checkpoint_path, device):
    """OmiCLIP 로드"""
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', pretrained=None)
    
    # Checkpoint 로드
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    return model, preprocess


def train_one_epoch(model, train_loader, optimizer, device, temperature=0.07):
    """1 epoch 학습"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        images = batch['image'].to(device)
        texts = batch['text']
        
        # Tokenize texts
        text_tokens = model.tokenizer(texts).to(device)
        
        # Forward
        image_embeds = model.encode_image(images)
        if isinstance(image_embeds, tuple):
            image_embeds = image_embeds[0]
            
        text_embeds = model.encode_text(text_tokens)
        
        # Contrastive loss
        loss = contrastive_loss(image_embeds, text_embeds, temperature)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate_with_predex(model, train_loader, val_loader, device, temperature=0.07):
    """
    Fine-tuned 모델로 Loki PredEx 방식 평가
    
    1. Train set 인코딩 (참고 DB)
    2. Val set 예측 (가중 평균)
    3. Spot-wise & Gene-wise Pearson 계산
    """
    model.eval()
    
    # ========================================
    # 1. Train set 인코딩 (참고 DB)
    # ========================================
    print("  Encoding train set...")
    train_img_embs = []
    train_text_embs = []
    train_exprs = []
    
    for batch in tqdm(train_loader, desc="  Train encoding"):
        images = batch['image'].to(device)
        texts = batch['text']
        gt_expr = batch['gt_expr']
        
        # Image embedding
        img_emb = model.encode_image(images)
        if isinstance(img_emb, tuple):
            img_emb = img_emb[0]
        img_emb = F.normalize(img_emb, dim=-1)
        
        # Text embedding
        text_tokens = model.tokenizer(texts).to(device)
        text_emb = model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, dim=-1)
        
        train_img_embs.append(img_emb)
        train_text_embs.append(text_emb)
        train_exprs.append(gt_expr.to(device))
    
    train_img_embs = torch.cat(train_img_embs, dim=0).to(device)  # (N, 768)
    train_text_embs = torch.cat(train_text_embs, dim=0).to(device)  # (N, 768)
    train_exprs = torch.cat(train_exprs, dim=0).to(device)  # (N, G)
    
    # ========================================
    # 2. Val set 예측 (Loki PredEx)
    # ========================================
    print("  Predicting val set...")
    predictions = []
    ground_truths = []
    
    for batch in tqdm(val_loader, desc="  Val prediction"):
        images = batch['image'].to(device)
        gt_expr = batch['gt_expr'].numpy()
        
        # Image embedding
        img_emb = model.encode_image(images)
        if isinstance(img_emb, tuple):
            img_emb = img_emb[0]
        img_emb = F.normalize(img_emb, dim=-1)
        
        # Loki PredEx: 가중 평균 (공식 수식)
        for i in range(len(img_emb)):
            test_emb = img_emb[i]  # (768,)
            
            # Similarity (cosine similarity)
            similarities = test_emb @ train_text_embs.T  # (N,)
            
            # Weights: sum normalization (Loki 공식 수식)
            weights = similarities / similarities.sum()  # (N,)
            
            # Weighted average
            pred_expr = (weights[:, None] * train_exprs).sum(dim=0)  # (G,)
            
            predictions.append(pred_expr.cpu().numpy())
            ground_truths.append(gt_expr[i])
    
    predictions = np.array(predictions)  # (M, G)
    ground_truths = np.array(ground_truths)  # (M, G)
    
    # ========================================
    # 3. Evaluation
    # ========================================
    print("  Calculating metrics...")
    
    # Spot-wise correlation
    spot_corrs = []
    for i in range(len(predictions)):
        if ground_truths[i].std() > 1e-8:
            r, _ = pearsonr(predictions[i], ground_truths[i])
            if np.isfinite(r):
                spot_corrs.append(r)
    
    # Gene-wise correlation
    gene_corrs = []
    for g in range(predictions.shape[1]):
        if ground_truths[:, g].std() > 1e-8:
            r, _ = pearsonr(predictions[:, g], ground_truths[:, g])
            if np.isfinite(r):
                gene_corrs.append(r)
    
    return {
        'spot_pearson_mean': float(np.mean(spot_corrs)),
        'spot_pearson_std': float(np.std(spot_corrs)),
        'gene_pearson_mean': float(np.mean(gene_corrs)),
        'gene_pearson_std': float(np.std(gene_corrs))
    }


def run_fold(fold_idx, args, gt_expr, gt_obs, all_genes, hvg_indices, obs_to_idx, gene_to_idx):
    """한 fold 실행"""
    import open_clip
    
    print(f"\n{'='*70}")
    print(f"Fold {fold_idx:02d}")
    print(f"{'='*70}")
    
    device = torch.device(args.device)
    fold_dir = Path(args.output_dir) / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 1. Load pretrained model
    # ========================================
    print("[1] Loading pretrained OmiCLIP...")
    model, preprocess = load_omiclip(args.pretrained, device)
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')
    model.tokenizer = tokenizer  # Add tokenizer to model
    
    # ========================================
    # 2. Load datasets
    # ========================================
    print("[2] Loading datasets...")
    train_csv = Path(args.folds_dir) / f"fold_{fold_idx:02d}_train.csv"
    val_csv = Path(args.folds_dir) / f"fold_{fold_idx:02d}_val.csv"
    
    train_dataset = STDataset(train_csv, preprocess, tokenizer, 
                              gt_expr, gt_obs, hvg_indices, obs_to_idx, gene_to_idx)
    val_dataset = STDataset(val_csv, preprocess, tokenizer,
                           gt_expr, gt_obs, hvg_indices, obs_to_idx, gene_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    print(f"  Train: {len(train_dataset)} spots")
    print(f"  Val:   {len(val_dataset)} spots")
    
    # ========================================
    # 3. Fine-tuning
    # ========================================
    print(f"[3] Fine-tuning ({args.epochs} epochs)...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    best_val_spot = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.temperature)
        print(f"  Train loss: {train_loss:.4f}")
        
        # 매 epoch 저장
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss
        }, fold_dir / f'epoch_{epoch+1:02d}.pt')
        print(f"  ✓ epoch_{epoch+1:02d}.pt saved")
        
        # 평가는 epoch 2, 4, 6, 8, 10에만
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            print(f"  Evaluating...")
            metrics = evaluate_with_predex(model, train_loader, val_loader, device, args.temperature)
            print(f"  Val Spot Pearson: {metrics['spot_pearson_mean']:.4f}")
            print(f"  Val Gene Pearson: {metrics['gene_pearson_mean']:.4f}")
            
            # best model 저장
            if metrics['spot_pearson_mean'] > best_val_spot:
                best_val_spot = metrics['spot_pearson_mean']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }, fold_dir / 'best_model.pt')
                print(f"  ✓ Best model saved! (epoch {epoch+1})")
    
    # ========================================
    # 4. Final evaluation with best model
    # ========================================
    print("\n[4] Final evaluation...")
    best_ckpt = torch.load(fold_dir / 'best_model.pt')
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    final_metrics = evaluate_with_predex(model, train_loader, val_loader, device, args.temperature)
    
    # Save results
    results = {
        'fold': fold_idx,
        'train_spots': len(train_dataset),
        'val_spots': len(val_dataset),
        'epochs': args.epochs,
        'best_epoch': best_ckpt['epoch'],
        'spot_pearson_mean': final_metrics['spot_pearson_mean'],
        'spot_pearson_std': final_metrics['spot_pearson_std'],
        'gene_pearson_mean': final_metrics['gene_pearson_mean'],
        'gene_pearson_std': final_metrics['gene_pearson_std']
    }
    
    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Fold {fold_idx:02d} Results:")
    print(f"  Spot Pearson: {results['spot_pearson_mean']:.4f} ± {results['spot_pearson_std']:.4f}")
    print(f"  Gene Pearson: {results['gene_pearson_mean']:.4f} ± {results['gene_pearson_std']:.4f}")
    print(f"{'='*70}")
    
    return results


def main(args):
    # ========================================
    # Load ground truth
    # ========================================
    print("="*70)
    print("Loki Official Fine-tuning 10-Fold CV")
    print("="*70)
    print(f"Folds directory: {args.folds_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print("="*70)
    
    print("\n[Loading ground truth...]")
    gt_expr = np.load(args.gt_expr)
    gt_obs = np.load(args.gt_obs, allow_pickle=True)
    all_genes = open(args.gene_list).read().strip().split('\n')
    hvg_genes = open(args.hvg_file).read().strip().split('\n')
    
    obs_to_idx = {b: i for i, b in enumerate(gt_obs)}
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    hvg_indices = [gene_to_idx[g] for g in hvg_genes if g in gene_to_idx]
    
    print(f"  Spots: {len(gt_obs)}")
    print(f"  Genes: {len(all_genes)}")
    print(f"  HVG: {len(hvg_indices)}")
    
    # ========================================
    # Run all folds
    # ========================================
    all_results = []
    
    for fold_idx in range(args.start_fold, args.end_fold + 1):
        start_time = time.time()
        
        results = run_fold(fold_idx, args, gt_expr, gt_obs, all_genes, 
                          hvg_indices, obs_to_idx, gene_to_idx)
        all_results.append(results)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Fold {fold_idx:02d} completed in {elapsed/60:.1f} minutes\n")
    
    # ========================================
    # Aggregate results
    # ========================================
    print("\n" + "="*70)
    print("10-Fold CV Results Summary")
    print("="*70)
    
    spot_means = [r['spot_pearson_mean'] for r in all_results]
    gene_means = [r['gene_pearson_mean'] for r in all_results]
    
    print(f"\nSpot-wise Pearson: {np.mean(spot_means):.4f} ± {np.std(spot_means):.4f}")
    print(f"Gene-wise Pearson: {np.mean(gene_means):.4f} ± {np.std(gene_means):.4f}")
    
    # Save summary
    summary = {
        'method': 'Loki Official Fine-tuning',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'temperature': args.temperature,
        'spot_pearson_mean': float(np.mean(spot_means)),
        'spot_pearson_std': float(np.std(spot_means)),
        'gene_pearson_mean': float(np.mean(gene_means)),
        'gene_pearson_std': float(np.std(gene_means)),
        'folds': all_results
    }
    
    with open(Path(args.output_dir) / '10fold_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    
    # Data
    p.add_argument("--folds_dir", required=True, help="10-fold CSV directory")
    p.add_argument("--hvg_file", required=True, help="HVG gene list")
    p.add_argument("--gt_expr", default="/project_antwerp/hbae/data/0317_HVG_NEW_norm/combined_expression_matrix.npy")
    p.add_argument("--gt_obs", default="/project_antwerp/hbae/data/0317_HVG_NEW_norm/combined_obs.npy")
    p.add_argument("--gene_list", default="/project_antwerp/hbae/data/0317_HVG_NEW_norm/31s_all_shared_genes.txt")
    p.add_argument("--pretrained", default="/project_antwerp/assets/loki_ckpts/checkpoint.pt")
    
    # Training
    p.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs (논문 권장: 10)")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size (논문: 64)")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    
    # Folds
    p.add_argument("--start_fold", type=int, default=1)
    p.add_argument("--end_fold", type=int, default=10)
    
    # Output
    p.add_argument("--output_dir", default="./loki_finetune_10fold")
    p.add_argument("--device", default="cuda:0")
    
    args = p.parse_args()
    main(args)