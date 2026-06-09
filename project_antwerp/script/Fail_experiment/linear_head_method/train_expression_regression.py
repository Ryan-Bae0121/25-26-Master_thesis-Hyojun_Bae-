#!/usr/bin/env python3
"""
train_expression_regression.py
==============================
Regression head 방식으로 gene expression 예측 학습

Strategy:
- Freeze pretrained OmiCLIP visual encoder
- Add regression head: embedding → expression values
- Train with MSE loss
- Evaluate with Pearson correlation

Usage:
    python train_expression_regression.py \
        --train_csv /path/to/train.csv \
        --val_csv /path/to/val.csv \
        --gene_list /path/to/HVG_genelist.txt \
        --output_dir ./output \
        --epochs 20 \
        --batch_size 64 \
        --lr 1e-4 \
        --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════
class SpatialExpressionDataset(Dataset):
    def __init__(self, csv_path, gt_expr, obs_to_idx, gene_to_idx, 
                 hvg_genes, preprocess):
        """
        csv: filepath column
        gt_expr: (n_spots, n_genes) expression matrix
        hvg_genes: list of HVG gene names (2089)
        """
        import pandas as pd
        
        self.df = pd.read_csv(csv_path)
        self.df['obs_key'] = self.df['filepath'].apply(self.extract_obs_key)
        
        # GT 매칭 가능한 spot만
        self.df = self.df[self.df['obs_key'].isin(obs_to_idx)].reset_index(drop=True)
        
        self.gt_expr = gt_expr
        self.obs_to_idx = obs_to_idx
        self.preprocess = preprocess
        
        # HVG indices
        self.hvg_indices = [gene_to_idx[g] for g in hvg_genes if g in gene_to_idx]
        self.n_genes = len(self.hvg_indices)
        
        print(f"  Dataset size: {len(self.df):,}")
        print(f"  HVG genes: {self.n_genes}")
    
    def extract_obs_key(self, filepath):
        parts = filepath.split('/')
        # GSM... 또는 숫자 디렉토리 찾기
        gsm_parts = [p for p in parts if p.startswith('GSM') or 
                     (len(p) == 7 and p[0].isdigit())]
        if gsm_parts:
            sample_id = gsm_parts[0].split('_')[0]
        else:
            sample_id = parts[6].split('_')[0]
        
        barcode = parts[-1].replace('.png', '')
        return f"{sample_id}_{barcode}_hires"
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image
        try:
            img = Image.open(row['filepath']).convert('RGB')
            img = self.preprocess(img)
        except:
            img = torch.zeros(3, 224, 224)
        
        # Expression (2089 HVG)
        spot_idx = self.obs_to_idx[row['obs_key']]
        expr = self.gt_expr[spot_idx, self.hvg_indices].astype(np.float32)
        expr = torch.from_numpy(expr)
        
        return img, expr


# ═══════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════
class ExpressionPredictor(nn.Module):
    def __init__(self, base_model, n_genes=2089, freeze_encoder=True):
        super().__init__()
        
        # Visual encoder from pretrained OmiCLIP
        self.visual = base_model.visual
        
        # Freeze encoder
        if freeze_encoder:
            for param in self.visual.parameters():
                param.requires_grad = False
            print("  ✓ Visual encoder frozen")
        
        # Embedding dimension (CoCa ViT-L-14: 768)
        self.embed_dim = 768
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_genes)
        )
        
        print(f"  ✓ Regression head created: {self.embed_dim} → {n_genes}")
    
    def forward(self, images):
        # Encode images
        img_emb = self.visual(images)  # May return tuple
        
        # Handle tuple output (some models return (embedding, other_stuff))
        if isinstance(img_emb, tuple):
            img_emb = img_emb[0]
        
        # Predict expression
        expr_pred = self.regression_head(img_emb)  # (B, n_genes)
        return expr_pred


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, expr_gt in tqdm(loader, desc="Training"):
        images = images.to(device)
        expr_gt = expr_gt.to(device)
        
        optimizer.zero_grad()
        
        expr_pred = model(images)
        loss = F.mse_loss(expr_pred, expr_gt)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    
    all_preds = []
    all_gts = []
    total_loss = 0
    
    for images, expr_gt in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        expr_gt = expr_gt.to(device)
        
        expr_pred = model(images)
        loss = F.mse_loss(expr_pred, expr_gt)
        total_loss += loss.item()
        
        all_preds.append(expr_pred.cpu().numpy())
        all_gts.append(expr_gt.cpu().numpy())
    
    # Concatenate
    all_preds = np.vstack(all_preds)  # (n_spots, n_genes)
    all_gts = np.vstack(all_gts)
    
    # Spot-wise correlation
    spot_corrs = []
    for i in range(len(all_preds)):
        if all_gts[i].std() > 1e-8:
            r, _ = pearsonr(all_preds[i], all_gts[i])
            if np.isfinite(r):
                spot_corrs.append(r)
    
    # Gene-wise correlation
    gene_corrs = []
    for g in range(all_preds.shape[1]):
        if all_gts[:, g].std() > 1e-8:
            r, _ = pearsonr(all_preds[:, g], all_gts[:, g])
            if np.isfinite(r):
                gene_corrs.append(r)
    
    return {
        'loss': total_loss / len(loader),
        'spot_pearson_mean': np.mean(spot_corrs) if spot_corrs else 0,
        'spot_pearson_median': np.median(spot_corrs) if spot_corrs else 0,
        'gene_pearson_mean': np.mean(gene_corrs) if gene_corrs else 0,
        'gene_pearson_median': np.median(gene_corrs) if gene_corrs else 0,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main(args):
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # ─── Load base model ───────────────────────────────────
    print("[1] Loading pretrained OmiCLIP...")
    import open_clip
    
    # Create model without pretrained weights first
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', pretrained=None)
    
    # Load checkpoint manually
    ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict, strict=False)
    
    base_model = base_model.to(device)
    print(f"  ✓ Loaded from {args.pretrained}")
    
    # ─── Load GT ───────────────────────────────────────────
    print("[2] Loading ground truth...")
    gt_expr = np.load(args.gt_expr)
    gt_obs = np.load(args.gt_obs, allow_pickle=True)
    all_genes = open(args.gene_list).read().strip().split('\n')
    
    obs_to_idx = {b: i for i, b in enumerate(gt_obs)}
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    
    print(f"  Expression: {gt_expr.shape}")
    print(f"  Genes: {len(all_genes)}")
    
    # ─── Load HVG ──────────────────────────────────────────
    print("[3] Loading HVG list...")
    hvg_genes = open(args.hvg_file).read().strip().split('\n')
    hvg_genes = [g for g in hvg_genes if g in gene_to_idx]
    print(f"  HVG: {len(hvg_genes)}")
    
    # ─── Create datasets ───────────────────────────────────
    print("[4] Creating datasets...")
    train_ds = SpatialExpressionDataset(
        args.train_csv, gt_expr, obs_to_idx, gene_to_idx, hvg_genes, preprocess)
    val_ds = SpatialExpressionDataset(
        args.val_csv, gt_expr, obs_to_idx, gene_to_idx, hvg_genes, preprocess)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # ─── Create model ──────────────────────────────────────
    print("[5] Creating regression model...")
    model = ExpressionPredictor(base_model, n_genes=len(hvg_genes), 
                                freeze_encoder=args.freeze_encoder)
    model = model.to(device)
    
    # ─── Optimizer ─────────────────────────────────────────
    print("[6] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.regression_head.parameters(),  # Only train head
        lr=args.lr,
        weight_decay=args.wd
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    
    # ─── Training loop ─────────────────────────────────────
    print(f"[7] Training for {args.epochs} epochs...")
    
    best_val_corr = -1
    history = []
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        # Log
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_spot_pearson_mean': val_metrics['spot_pearson_mean'],
            'val_spot_pearson_median': val_metrics['spot_pearson_median'],
            'val_gene_pearson_mean': val_metrics['gene_pearson_mean'],
            'val_gene_pearson_median': val_metrics['gene_pearson_median'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(metrics)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f}")
        print(f"Val Spot Pearson:  mean={val_metrics['spot_pearson_mean']:.4f}, "
              f"median={val_metrics['spot_pearson_median']:.4f}")
        print(f"Val Gene Pearson:  mean={val_metrics['gene_pearson_mean']:.4f}, "
              f"median={val_metrics['gene_pearson_median']:.4f}")
        
        # Save best
        if val_metrics['spot_pearson_mean'] > best_val_corr:
            best_val_corr = val_metrics['spot_pearson_mean']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_spot_pearson': val_metrics['spot_pearson_mean'],
            }, out / 'best_model.pt')
            print(f"✓ Saved best model (spot_pearson={best_val_corr:.4f})")
    
    # ─── Save history ──────────────────────────────────────
    with open(out / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val spot Pearson: {best_val_corr:.4f}")
    print(f"Saved to: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--hvg_file", required=True, help="HVG gene list")
    p.add_argument("--gt_expr", default="/project_antwerp/hbae/data/combined_expression_matrix.npy")
    p.add_argument("--gt_obs", default="/project_antwerp/hbae/data/combined_obs.npy")
    p.add_argument("--gene_list", default="/project_antwerp/hbae/data/all_shared_genes.txt")
    p.add_argument("--pretrained", default="/project_antwerp/assets/loki_ckpts/checkpoint.pt")
    p.add_argument("--output_dir", default="./regression_output")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--freeze_encoder", action='store_true', default=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    
    args = p.parse_args()
    main(args)