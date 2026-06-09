#!/usr/bin/env python3
"""
diagnose_prediction_variance.py
===============================
예측값의 변동성 문제 진단

Usage:
    python diagnose_prediction_variance.py \
        --predictions_file predictions.npy \
        --ground_truth_file ground_truth.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def diagnose(pred_file, gt_file):
    print("="*70)
    print("Prediction Variance Diagnostic")
    print("="*70)
    
    # Load
    pred = np.load(pred_file)
    gt = np.load(gt_file)
    
    print(f"\nData shape: {pred.shape}")
    
    # Overall statistics
    print("\n1. Overall Statistics:")
    print(f"   Predictions:")
    print(f"     Min:  {pred.min():.6f}")
    print(f"     Max:  {pred.max():.6f}")
    print(f"     Mean: {pred.mean():.6f}")
    print(f"     Std:  {pred.std():.6f}")
    
    print(f"\n   Ground Truth:")
    print(f"     Min:  {gt.min():.6f}")
    print(f"     Max:  {gt.max():.6f}")
    print(f"     Mean: {gt.mean():.6f}")
    print(f"     Std:  {gt.std():.6f}")
    
    # Per-gene variance
    pred_var_genes = pred.var(axis=0)
    gt_var_genes = gt.var(axis=0)
    
    print("\n2. Per-gene Variance (across spots):")
    print(f"   Predictions:")
    print(f"     Mean var: {pred_var_genes.mean():.6f}")
    print(f"     Min var:  {pred_var_genes.min():.6f}")
    print(f"     Max var:  {pred_var_genes.max():.6f}")
    
    print(f"\n   Ground Truth:")
    print(f"     Mean var: {gt_var_genes.mean():.6f}")
    print(f"     Min var:  {gt_var_genes.min():.6f}")
    print(f"     Max var:  {gt_var_genes.max():.6f}")
    
    # Per-spot variance
    pred_var_spots = pred.var(axis=1)
    gt_var_spots = gt.var(axis=1)
    
    print("\n3. Per-spot Variance (across genes):")
    print(f"   Predictions:")
    print(f"     Mean var: {pred_var_spots.mean():.6f}")
    print(f"     Min var:  {pred_var_spots.min():.6f}")
    print(f"     Max var:  {pred_var_spots.max():.6f}")
    
    print(f"\n   Ground Truth:")
    print(f"     Mean var: {gt_var_spots.mean():.6f}")
    print(f"     Min var:  {gt_var_spots.min():.6f}")
    print(f"     Max var:  {gt_var_spots.max():.6f}")
    
    # Sample comparison
    print("\n4. Sample Gene (Gene 0):")
    print(f"   Pred: min={pred[:, 0].min():.4f}, max={pred[:, 0].max():.4f}, "
          f"mean={pred[:, 0].mean():.4f}, var={pred[:, 0].var():.6f}")
    print(f"   GT:   min={gt[:, 0].min():.4f}, max={gt[:, 0].max():.4f}, "
          f"mean={gt[:, 0].mean():.4f}, var={gt[:, 0].var():.6f}")
    
    # Sample spot
    print("\n5. Sample Spot (Spot 0):")
    print(f"   Pred: min={pred[0, :].min():.4f}, max={pred[0, :].max():.4f}, "
          f"mean={pred[0, :].mean():.4f}, var={pred[0, :].var():.6f}")
    print(f"   GT:   min={gt[0, :].min():.4f}, max={gt[0, :].max():.4f}, "
          f"mean={gt[0, :].mean():.4f}, var={gt[0, :].var():.6f}")
    
    # Check if predictions are all similar
    print("\n6. Checking for constant predictions:")
    spot_means = pred.mean(axis=1)
    print(f"   Std of spot means: {spot_means.std():.6f}")
    if spot_means.std() < 0.001:
        print("   ⚠️  WARNING: All spots have very similar predictions!")
        print("   → Predictions are nearly constant across spots")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram of values
    axes[0, 0].hist(pred.flatten(), bins=100, alpha=0.5, label='Predicted', density=True)
    axes[0, 0].hist(gt.flatten(), bins=100, alpha=0.5, label='Ground Truth', density=True)
    axes[0, 0].set_xlabel('Expression Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution of Expression Values')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. Variance comparison
    axes[0, 1].scatter(gt_var_genes, pred_var_genes, alpha=0.5, s=20)
    max_var = max(gt_var_genes.max(), pred_var_genes.max())
    axes[0, 1].plot([0, max_var], [0, max_var], 'r--', label='y=x')
    axes[0, 1].set_xlabel('True Variance (per gene)')
    axes[0, 1].set_ylabel('Predicted Variance (per gene)')
    axes[0, 1].set_title('Variance Comparison')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # 3. Sample gene across spots
    axes[1, 0].plot(gt[:, 0], label='Ground Truth', alpha=0.7)
    axes[1, 0].plot(pred[:, 0], label='Predicted', alpha=0.7)
    axes[1, 0].set_xlabel('Spot Index')
    axes[1, 0].set_ylabel('Expression (Gene 0)')
    axes[1, 0].set_title('Sample Gene Across Spots')
    axes[1, 0].legend()
    
    # 4. Sample spot across genes
    axes[1, 1].plot(gt[0, :], label='Ground Truth', alpha=0.7)
    axes[1, 1].plot(pred[0, :], label='Predicted', alpha=0.7)
    axes[1, 1].set_xlabel('Gene Index')
    axes[1, 1].set_ylabel('Expression (Spot 0)')
    axes[1, 1].set_title('Sample Spot Across Genes')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('prediction_diagnostic.png', dpi=150)
    print(f"\n✓ Diagnostic plot saved: prediction_diagnostic.png")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_file", required=True)
    p.add_argument("--ground_truth_file", required=True)
    args = p.parse_args()
    
    diagnose(args.predictions_file, args.ground_truth_file)