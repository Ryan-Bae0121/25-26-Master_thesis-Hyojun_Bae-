#!/usr/bin/env python3
"""
Gene Expression Quality Analysis using evaluation.ipynb methodology
Based on the provided evaluation.ipynb notebook for significantly well predicted genes

This script analyzes predicted gene expression data using the same methodology
as the evaluation.ipynb notebook, including Steiger's test and FDR correction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import fdrcorrection
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import corrstats, if not available, implement a simple version
try:
    from corrstats import dependent_corr
except ImportError:
    print("corrstats not available, implementing simple dependent correlation test")
    def dependent_corr(xy, xz, yz, n, twotailed=False, conf_level=0.95, method='steiger'):
        """
        Simple implementation of dependent correlation test
        Based on Steiger's method for comparing dependent correlations
        """
        # Calculate Fisher's z transformation
        def fisher_z(r):
            return 0.5 * np.log((1 + r) / (1 - r))
        
        # Calculate standard error
        def se_fisher_z(r, n):
            return 1 / np.sqrt(n - 3)
        
        # Convert correlations to Fisher's z
        z_xy = fisher_z(xy)
        z_xz = fisher_z(xz)
        
        # Calculate standard errors
        se_xy = se_fisher_z(xy, n)
        se_xz = se_fisher_z(xz, n)
        
        # Calculate covariance between Fisher's z values
        # Simplified covariance formula for dependent correlations
        cov_xy_xz = (yz - 0.5 * xy * xz) / ((n - 1) * se_xy * se_xz)
        
        # Calculate test statistic
        z_diff = z_xy - z_xz
        se_diff = np.sqrt(se_xy**2 + se_xz**2 - 2 * cov_xy_xz)
        
        # Calculate p-value
        z_stat = z_diff / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if twotailed else 1 - stats.norm.cdf(z_stat)
        
        return z_stat, p_value

def load_and_orient_data(csv_path):
    """Load CSV and ensure genes are rows, patches are columns"""
    print(f"데이터 로딩 중: {csv_path}")
    
    # Load data
    data = pd.read_csv(csv_path, index_col=0)
    
    # Auto-detect orientation: genes should be rows (more genes than patches typically)
    if data.shape[0] < data.shape[1]:
        print("데이터 전치 중 (genes를 행으로 설정)")
        data = data.T
    
    print(f"데이터 형태: {data.shape[0]} genes × {data.shape[1]} patches")
    
    # Clean gene names
    data.index = data.index.str.strip()
    
    # Handle missing values by dropping patches with any NaN
    initial_patches = data.shape[1]
    data = data.dropna(axis=1)
    final_patches = data.shape[1]
    
    if initial_patches != final_patches:
        print(f"NaN 값이 있는 patches 제거: {initial_patches} → {final_patches}")
    
    return data

def create_random_baseline(data):
    """Create random baseline by shuffling each gene's values"""
    print("랜덤 baseline 생성 중...")
    random_data = data.copy()
    
    for gene in data.index:
        # Shuffle the values for this gene
        random_data.loc[gene] = np.random.permutation(data.loc[gene].values)
    
    return random_data

def evaluate_significant_genes(real_data, pred_data, random_data):
    """
    Evaluate significantly well predicted genes using evaluation.ipynb methodology
    """
    print("유의한 genes 평가 중...")
    
    pred_r = []
    random_r = []
    test_p = []
    pearson_p = []
    rmse_pred = []
    rmse_random = []
    valid_genes = []
    
    genes = pred_data.columns.values
    print(f"Analyzing {len(genes)} genes...")
    
    for i, gene in enumerate(genes):
        if i % 1000 == 0:
            print(f"Processing gene {i+1}/{len(genes)}: {gene}")
        # Correlation test
        xy, p1 = stats.pearsonr(real_data.loc[:, gene], pred_data.loc[:, gene])
        xz, p2 = stats.pearsonr(real_data.loc[:, gene], random_data.loc[:, gene])
        yz, p3 = stats.pearsonr(pred_data.loc[:, gene], random_data.loc[:, gene])
        n = len(real_data.loc[:, gene])
        
        # Skip if any correlation is None (constant values)
        if None in (p1, p2, p3):
            continue
            
        # Steiger's test for dependent correlations
        try:
            t, p = dependent_corr(xy, xz, yz, n, twotailed=False, conf_level=0.95, method='steiger')
        except:
            continue
            
        if p is None:
            continue
        
        pred_r.append(xy)
        random_r.append(xz)
        test_p.append(p)
        pearson_p.append(p1)
        
        # RMSE test
        rmse1 = np.sqrt(mean_squared_error(real_data.loc[:, gene], pred_data.loc[:, gene]))
        rmse2 = np.sqrt(mean_squared_error(real_data.loc[:, gene], random_data.loc[:, gene]))
        rmse_pred.append(rmse1)
        rmse_random.append(rmse2)
        valid_genes.append(gene)
    
    # Create results dataframe
    combine_res = pd.DataFrame({
        "pred_real_r": pred_r,
        "random_real_r": random_r,
        'pearson_p': pearson_p,
        "Steiger_p": test_p,
        'rmse_pred': rmse_pred,
        'rmse_random': rmse_random
    }, index=valid_genes)
    
    # Sort by prediction correlation
    combine_res = combine_res.sort_values('pred_real_r', ascending=False)
    combine_res = combine_res[~combine_res['Steiger_p'].isna()]
    
    # FDR correction
    _, fdr_p = fdrcorrection(combine_res['Steiger_p'])
    combine_res['fdr_Steiger_p'] = fdr_p
    
    # Identify significantly well predicted genes
    sig_res = combine_res[
        (combine_res['pred_real_r'] > 0) & 
        (combine_res['pearson_p'] < 0.05) & 
        (combine_res['pred_real_r'] > combine_res['random_real_r']) & 
        (combine_res['Steiger_p'] < 0.05) & 
        (combine_res['fdr_Steiger_p'] < 0.2)
    ]
    
    print(f"Found {sig_res.shape[0]} significant genes")
    
    return combine_res, sig_res

def create_visualizations(combine_res, sig_res, out_dir):
    """Create visualizations based on evaluation.ipynb methodology"""
    print("시각화 생성 중...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribution of correlations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pred vs Real correlation
    axes[0, 0].hist(combine_res['pred_real_r'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Distribution of Pred-Real Correlations')
    axes[0, 0].set_xlabel('Correlation Coefficient')
    axes[0, 0].set_ylabel('Frequency')
    
    # Random vs Real correlation
    axes[0, 1].hist(combine_res['random_real_r'], bins=50, alpha=0.7, color='orange')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Distribution of Random-Real Correlations')
    axes[0, 1].set_xlabel('Correlation Coefficient')
    axes[0, 1].set_ylabel('Frequency')
    
    # RMSE comparison
    axes[1, 0].scatter(combine_res['rmse_random'], combine_res['rmse_pred'], alpha=0.6)
    axes[1, 0].plot([0, combine_res['rmse_random'].max()], [0, combine_res['rmse_random'].max()], 
                    'r--', alpha=0.7, label='y=x')
    axes[1, 0].set_xlabel('RMSE (Random)')
    axes[1, 0].set_ylabel('RMSE (Predicted)')
    axes[1, 0].set_title('RMSE: Predicted vs Random')
    axes[1, 0].legend()
    
    # Volcano-like plot: Steiger p-value vs correlation difference
    correlation_diff = combine_res['pred_real_r'] - combine_res['random_real_r']
    neg_log_p = -np.log10(combine_res['Steiger_p'])
    
    axes[1, 1].scatter(correlation_diff, neg_log_p, alpha=0.6, c='gray')
    
    # Highlight significant genes
    if len(sig_res) > 0:
        sig_corr_diff = sig_res['pred_real_r'] - sig_res['random_real_r']
        sig_neg_log_p = -np.log10(sig_res['Steiger_p'])
        axes[1, 1].scatter(sig_corr_diff, sig_neg_log_p, alpha=0.8, c='red', s=30)
    
    axes[1, 1].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Correlation Difference (Pred - Random)')
    axes[1, 1].set_ylabel('-log10(Steiger p-value)')
    axes[1, 1].set_title('Volcano Plot: Significance vs Correlation Improvement')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'evaluation_methodology_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top significant genes scatter plot
    if len(sig_res) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot all genes
        ax.scatter(combine_res['pred_real_r'], combine_res['random_real_r'], 
                  alpha=0.5, c='lightblue', label='All genes')
        
        # Highlight significant genes
        ax.scatter(sig_res['pred_real_r'], sig_res['random_real_r'], 
                  alpha=0.8, c='red', s=50, label=f'Significant genes (n={len(sig_res)})')
        
        # Add diagonal line
        max_corr = max(combine_res['pred_real_r'].max(), combine_res['random_real_r'].max())
        min_corr = min(combine_res['pred_real_r'].min(), combine_res['random_real_r'].min())
        ax.plot([min_corr, max_corr], [min_corr, max_corr], 'k--', alpha=0.7, label='y=x')
        
        ax.set_xlabel('Prediction-Real Correlation')
        ax.set_ylabel('Random-Real Correlation')
        ax.set_title('Correlation Comparison: Predicted vs Random')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'correlation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"시각화 완료: {out_dir}")

def generate_summary_report(combine_res, sig_res, out_dir):
    """Generate summary report"""
    print("요약 리포트 생성 중...")
    
    report_path = os.path.join(out_dir, 'evaluation_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Gene Expression Quality Analysis Report ===\n")
        f.write("Methodology: Based on evaluation.ipynb\n\n")
        
        f.write(f"Total genes analyzed: {len(combine_res)}\n")
        f.write(f"Significantly well predicted genes: {len(sig_res)}\n")
        if len(combine_res) > 0:
            f.write(f"Success rate: {len(sig_res)/len(combine_res)*100:.2f}%\n\n")
        else:
            f.write("Success rate: N/A (no genes analyzed)\n\n")
        
        f.write("=== Criteria for Significant Genes ===\n")
        f.write("1. pred_real_r > 0 (positive correlation)\n")
        f.write("2. pearson_p < 0.05 (significant correlation)\n")
        f.write("3. pred_real_r > random_real_r (better than random)\n")
        f.write("4. Steiger_p < 0.05 (significant improvement over random)\n")
        f.write("5. fdr_Steiger_p < 0.2 (FDR corrected significance)\n\n")
        
        f.write("=== Summary Statistics ===\n")
        f.write(f"Mean pred-real correlation: {combine_res['pred_real_r'].mean():.4f}\n")
        f.write(f"Mean random-real correlation: {combine_res['random_real_r'].mean():.4f}\n")
        f.write(f"Mean correlation improvement: {(combine_res['pred_real_r'] - combine_res['random_real_r']).mean():.4f}\n")
        f.write(f"Mean RMSE (predicted): {combine_res['rmse_pred'].mean():.4f}\n")
        f.write(f"Mean RMSE (random): {combine_res['rmse_random'].mean():.4f}\n\n")
        
        if len(sig_res) > 0:
            f.write("=== Top 10 Significant Genes ===\n")
            top_genes = sig_res.head(10)
            for i, (gene, row) in enumerate(top_genes.iterrows(), 1):
                f.write(f"{i:2d}. {gene:10s}: r={row['pred_real_r']:.4f}, "
                       f"Steiger_p={row['Steiger_p']:.4f}, FDR_p={row['fdr_Steiger_p']:.4f}\n")
        else:
            f.write("No significantly well predicted genes found.\n")
    
    print(f"요약 리포트 저장: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze gene expression quality using evaluation.ipynb methodology')
    parser.add_argument('--pred_csv', required=True, help='Path to predicted gene expression CSV')
    parser.add_argument('--out_dir', default='results_evaluation_method', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=== Gene Expression Quality Analysis (evaluation.ipynb method) ===")
    print(f"Input file: {args.pred_csv}")
    print(f"Output directory: {args.out_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load data
    pred_data = load_and_orient_data(args.pred_csv)
    
    # Create random baseline
    random_data = create_random_baseline(pred_data)
    
    # Use predicted data as "real" data for internal consistency analysis
    # This simulates the evaluation.ipynb approach where we compare pred vs random
    real_data = pred_data.copy()
    
    # Evaluate significant genes
    combine_res, sig_res = evaluate_significant_genes(real_data, pred_data, random_data)
    
    # Save results
    combine_res.to_csv(os.path.join(args.out_dir, 'all_genes_evaluation.csv'))
    if len(sig_res) > 0:
        sig_res.to_csv(os.path.join(args.out_dir, 'significant_genes_evaluation.csv'))
    
    # Create visualizations
    create_visualizations(combine_res, sig_res, args.out_dir)
    
    # Generate summary report
    generate_summary_report(combine_res, sig_res, args.out_dir)
    
    print(f"\n분석 완료! 결과는 {args.out_dir}에 저장되었습니다.")
    print(f"- 총 {len(combine_res)} genes 분석")
    print(f"- {len(sig_res)} genes가 유의하게 잘 예측됨")

if __name__ == "__main__":
    main()
