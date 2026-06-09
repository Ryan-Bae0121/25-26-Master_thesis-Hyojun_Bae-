#!/usr/bin/env python3
"""
train_linear_head.py
===================
Simple Linear Head for Gene Expression Prediction

Architecture:
    Image Embedding (768) → Linear → Gene Expression (2089)

Comparison:
    - Loki PredEx: Weighted average (no training)
    - Linear Head: Direct mapping (with training)

Usage:
    python train_linear_head.py \
        --train_csv fold_01_train.csv \
        --val_csv fold_01_val.csv \
        --hvg_file HVG_genelist.txt \
        --output_dir ./linear_head_results \
        --epochs 10
"""

import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr


class SimpleLinearHead(nn.Module):
    """Simple linear transformation"""
    def __init__(self, input_dim=768, output_dim=2089):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class LinearHeadWithDropout(nn.Module):
    """Linear head with dropout for regularization"""
    def __init__(self, input_dim=768, output_dim=2089, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class TwoLayerHead(nn.Module):
    """Two-layer head for comparison"""
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=2089):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        embeddings = batch['embedding'].to(device)
        targets = batch['expression'].to(device)
        
        # Forward
        predictions = model(embeddings)
        loss = criterion(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for batch in dataloader:
        embeddings = batch['embedding'].to(device)
        targets = batch['expression']
        
        predictions = model(embeddings).cpu().numpy()
        
        all_predictions.append(predictions)
        all_targets.append(targets.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Spot-wise Pearson
    spot_corrs = []
    for i in range(len(predictions)):
        if targets[i].std() > 1e-8:
            r, _ = pearsonr(predictions[i], targets[i])
            if np.isfinite(r):
                spot_corrs.append(r)
    
    # Gene-wise Pearson
    gene_corrs = []
    for g in range(predictions.shape[1]):
        if targets[:, g].std() > 1e-8:
            r, _ = pearsonr(predictions[:, g], targets[:, g])
            if np.isfinite(r):
                gene_corrs.append(r)
    
    # Variance ratio
    pred_var = predictions.var(axis=0)
    true_var = targets.var(axis=0)
    var_ratios = []
    for i in range(len(pred_var)):
        if true_var[i] > 1e-8:
            var_ratios.append(pred_var[i] / true_var[i])
    
    return {
        'spot_pearson_mean': np.mean(spot_corrs),
        'spot_pearson_std': np.std(spot_corrs),
        'gene_pearson_mean': np.mean(gene_corrs),
        'gene_pearson_std': np.std(gene_corrs),
        'gene_var_ratio_mean': np.mean(var_ratios),
        'gene_var_ratio_median': np.median(var_ratios),
    }


def main(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Linear Head Training for Gene Expression Prediction")
    print("="*70)
    
    # TODO: Load actual data
    # For now, create dummy data
    print("\n⚠️  Using dummy data for demonstration")
    print("   Implement actual data loading from embeddings\n")
    
    n_train = 10000
    n_val = 1000
    input_dim = 768
    output_dim = 2089
    
    # Dummy data
    train_embeddings = torch.randn(n_train, input_dim)
    train_expressions = torch.randn(n_train, output_dim).abs()
    val_embeddings = torch.randn(n_val, input_dim)
    val_expressions = torch.randn(n_val, output_dim).abs()
    
    # Create datasets
    class ExpressionDataset(Dataset):
        def __init__(self, embeddings, expressions):
            self.embeddings = embeddings
            self.expressions = expressions
        
        def __len__(self):
            return len(self.embeddings)
        
        def __getitem__(self, idx):
            return {
                'embedding': self.embeddings[idx],
                'expression': self.expressions[idx]
            }
    
    train_dataset = ExpressionDataset(train_embeddings, train_expressions)
    val_dataset = ExpressionDataset(val_embeddings, val_expressions)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    if args.model_type == 'simple':
        model = SimpleLinearHead(input_dim, output_dim)
    elif args.model_type == 'dropout':
        model = LinearHeadWithDropout(input_dim, output_dim, args.dropout)
    else:
        model = TwoLayerHead(input_dim, args.hidden_dim, output_dim)
    
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training
    print(f"Model: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining for {args.epochs} epochs...\n")
    
    history = []
    best_val_corr = -1
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **val_metrics
        })
        
        print(f"Epoch {epoch+1:02d}: "
              f"Loss={train_loss:.4f}, "
              f"Spot r={val_metrics['spot_pearson_mean']:.4f}, "
              f"Gene r={val_metrics['gene_pearson_mean']:.4f}, "
              f"Var ratio={val_metrics['gene_var_ratio_mean']:.4f}")
        
        # Save best
        if val_metrics['spot_pearson_mean'] > best_val_corr:
            best_val_corr = val_metrics['spot_pearson_mean']
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
    
    # Save results
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best Spot Pearson: {best_val_corr:.4f}")
    print(f"Final Gene Pearson: {history[-1]['gene_pearson_mean']:.4f}")
    print(f"Final Var Ratio: {history[-1]['gene_var_ratio_mean']:.4f}")
    print("="*70)
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", help="Training CSV")
    p.add_argument("--val_csv", help="Validation CSV")
    p.add_argument("--hvg_file", default="/project_antwerp/hbae/HVG_genelist.txt")
    p.add_argument("--model_type", default="simple", 
                   choices=['simple', 'dropout', 'two_layer'])
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output_dir", default="./linear_head_results")
    p.add_argument("--device", default="cuda:0")
    
    args = p.parse_args()
    main(args)