#!/usr/bin/env python3
"""
analyze_temperature_effect.py
=============================
Temperature가 전체 성능에 미치는 영향 분석

Temperature (τ)의 역할:
- 낮음 (0.01): Sharp distribution (top-1 dominant)
- 중간 (0.07): Balanced (논문 기본값)
- 높음 (1.0): Smooth distribution (many contribute)

분석:
1. Fold 01에서 여러 temperature 테스트
2. Spot Pearson, Gene Pearson, Variance Ratio 변화
3. 최적 temperature 찾기

Usage:
    python analyze_temperature_effect.py \
        --predictions_dir /path/to/fold_01 \
        --temperatures 0.01,0.03,0.05,0.07,0.1,0.3,0.5,1.0 \
        --output_dir ./temperature_analysis
"""

import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_embeddings_and_expressions(fold_dir):
    """
    Load pre-computed embeddings and expressions
    
    Note: 이 함수는 실제로 저장된 임베딩을 로드해야 함
    현재는 placeholder
    """
    # 실제 구현 필요
    print("⚠️  Warning: 실제 임베딩 로드 구현 필요")
    print("   현재는 demonstration용 placeholder 사용")
    
    # Placeholder
    n_train = 111330
    n_val = 14145
    n_genes = 2089
    embed_dim = 768
    
    train_text_embs = torch.randn(n_train, embed_dim)
    train_exprs = torch.randn(n_train, n_genes).abs()
    val_img_embs = torch.randn(n_val, embed_dim)
    val_exprs = np.random.randn(n_val, n_genes).clip(0, 5)
    
    # Normalize
    train_text_embs = F.normalize(train_text_embs, dim=-1)
    val_img_embs = F.normalize(val_img_embs, dim=-1)
    
    return train_text_embs, train_exprs, val_img_embs, val_exprs


def loki_predex_with_temperature(val_img_embs, train_text_embs, train_exprs, temperature):
    """
    Loki PredEx with specific temperature
    """
    n_val = len(val_img_embs)
    n_genes = train_exprs.shape[1]
    predictions = np.zeros((n_val, n_genes))
    
    print(f"  Computing predictions with τ={temperature:.3f}...", end=' ', flush=True)
    
    for i in range(n_val):
        test_emb = val_img_embs[i]
        
        # Similarity
        similarities = test_emb @ train_text_embs.T
        
        # Temperature scaling
        similarities = similarities / temperature
        
        # Softmax weights
        weights = F.softmax(similarities, dim=0)
        
        # Weighted average
        pred_expr = (weights[:, None] * train_exprs).sum(dim=0)
        predictions[i] = pred_expr.numpy()
    
    print("✓")
    return predictions


def evaluate_predictions(predictions, ground_truth):
    """Calculate metrics"""
    # Spot-wise Pearson
    spot_corrs = []
    for i in range(len(predictions)):
        if ground_truth[i].std() > 1e-8:
            r, _ = pearsonr(predictions[i], ground_truth[i])
            if np.isfinite(r):
                spot_corrs.append(r)
    
    # Gene-wise Pearson
    gene_corrs = []
    for g in range(predictions.shape[1]):
        if ground_truth[:, g].std() > 1e-8:
            r, _ = pearsonr(predictions[:, g], ground_truth[:, g])
            if np.isfinite(r):
                gene_corrs.append(r)
    
    # Variance ratios
    pred_var_genes = predictions.var(axis=0)
    true_var_genes = ground_truth.var(axis=0)
    
    gene_var_ratio = []
    for i in range(len(pred_var_genes)):
        if true_var_genes[i] > 1e-8:
            gene_var_ratio.append(pred_var_genes[i] / true_var_genes[i])
    
    return {
        'spot_pearson_mean': np.mean(spot_corrs) if spot_corrs else 0,
        'spot_pearson_std': np.std(spot_corrs) if spot_corrs else 0,
        'gene_pearson_mean': np.mean(gene_corrs) if gene_corrs else 0,
        'gene_pearson_std': np.std(gene_corrs) if gene_corrs else 0,
        'gene_var_ratio_mean': np.mean(gene_var_ratio) if gene_var_ratio else 0,
        'gene_var_ratio_median': np.median(gene_var_ratio) if gene_var_ratio else 0,
    }


def plot_temperature_effects(results, output_dir):
    """Visualize temperature effects"""
    output_dir = Path(output_dir)
    
    temps = [r['temperature'] for r in results]
    spot_means = [r['spot_pearson_mean'] for r in results]
    spot_stds = [r['spot_pearson_std'] for r in results]
    gene_means = [r['gene_pearson_mean'] for r in results]
    var_ratios = [r['gene_var_ratio_mean'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Spot Pearson vs Temperature
    axes[0, 0].errorbar(temps, spot_means, yerr=spot_stds,
                        marker='o', markersize=8, capsize=5, capthick=2,
                        linewidth=2, color='steelblue')
    axes[0, 0].set_xlabel('Temperature (τ)', fontsize=12)
    axes[0, 0].set_ylabel('Spot-wise Pearson', fontsize=12)
    axes[0, 0].set_title('Temperature Effect on Spot-wise Performance',
                        fontsize=13, fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].axvline(0.07, color='red', linestyle='--', alpha=0.5, label='Default (0.07)')
    axes[0, 0].legend()
    
    # Mark best
    best_idx = np.argmax(spot_means)
    axes[0, 0].scatter([temps[best_idx]], [spot_means[best_idx]], 
                      color='red', s=100, zorder=5, label=f'Best: τ={temps[best_idx]:.3f}')
    
    # 2. Gene Pearson vs Temperature
    axes[0, 1].plot(temps, gene_means, marker='s', markersize=8,
                   linewidth=2, color='coral')
    axes[0, 1].set_xlabel('Temperature (τ)', fontsize=12)
    axes[0, 1].set_ylabel('Gene-wise Pearson', fontsize=12)
    axes[0, 1].set_title('Temperature Effect on Gene-wise Performance',
                        fontsize=13, fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(0.07, color='red', linestyle='--', alpha=0.5, label='Default (0.07)')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Variance Ratio vs Temperature
    axes[1, 0].plot(temps, var_ratios, marker='d', markersize=8,
                   linewidth=2, color='mediumseagreen')
    axes[1, 0].set_xlabel('Temperature (τ)', fontsize=12)
    axes[1, 0].set_ylabel('Gene-wise Variance Ratio', fontsize=12)
    axes[1, 0].set_title('Temperature Effect on Variance Compression',
                        fontsize=13, fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].axvline(0.07, color='red', linestyle='--', alpha=0.5, label='Default (0.07)')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Summary table
    axes[1, 1].axis('off')
    
    table_data = []
    for i, temp in enumerate(temps):
        table_data.append([
            f'{temp:.3f}',
            f'{spot_means[i]:.4f}',
            f'{gene_means[i]:.4f}',
            f'{var_ratios[i]:.4f}'
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['τ', 'Spot r', 'Gene r', 'Var Ratio'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Highlight best
    for i in range(len(temps)):
        if i == best_idx:
            for j in range(4):
                table[(i+1, j)].set_facecolor('#90EE90')
    
    axes[1, 1].set_title('Summary Table', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temperature_effect_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Temperature effect plot saved")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse temperatures
    temperatures = [float(t) for t in args.temperatures.split(',')]
    print(f"\n[1] Testing {len(temperatures)} temperatures: {temperatures}")
    
    # Load data
    print("\n[2] Loading embeddings and expressions...")
    train_text_embs, train_exprs, val_img_embs, val_exprs = \
        load_embeddings_and_expressions(args.predictions_dir)
    
    print(f"  Train: {len(train_text_embs)} spots")
    print(f"  Val: {len(val_img_embs)} spots")
    
    # Test each temperature
    print("\n[3] Testing different temperatures...")
    results = []
    
    for temp in temperatures:
        print(f"\nTemperature: {temp:.3f}")
        
        # Predict
        predictions = loki_predex_with_temperature(
            val_img_embs, train_text_embs, train_exprs, temp)
        
        # Evaluate
        metrics = evaluate_predictions(predictions, val_exprs)
        metrics['temperature'] = temp
        results.append(metrics)
        
        print(f"  Spot Pearson: {metrics['spot_pearson_mean']:.4f}")
        print(f"  Gene Pearson: {metrics['gene_pearson_mean']:.4f}")
        print(f"  Var Ratio:    {metrics['gene_var_ratio_mean']:.4f}")
    
    # Find best
    best_idx = np.argmax([r['spot_pearson_mean'] for r in results])
    best_temp = results[best_idx]['temperature']
    best_corr = results[best_idx]['spot_pearson_mean']
    
    print("\n" + "="*70)
    print("Temperature Tuning Results")
    print("="*70)
    print(f"Best temperature: {best_temp:.3f}")
    print(f"Best spot Pearson: {best_corr:.4f}")
    print(f"Default (0.07) spot Pearson: {[r['spot_pearson_mean'] for r in results if r['temperature']==0.07][0]:.4f}")
    print("="*70)
    
    # Save results
    with open(output_dir / 'temperature_tuning_results.json', 'w') as f:
        json.dump({
            'temperatures_tested': temperatures,
            'best_temperature': best_temp,
            'best_spot_pearson': best_corr,
            'results': results
        }, f, indent=2)
    
    # Plot
    plot_temperature_effects(results, output_dir)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_dir", required=True,
                   help="Directory with fold predictions")
    p.add_argument("--temperatures", 
                   default="0.01,0.03,0.05,0.07,0.1,0.3,0.5,1.0",
                   help="Comma-separated temperature values")
    p.add_argument("--output_dir", default="./temperature_analysis")
    
    args = p.parse_args()
    main(args)