#!/usr/bin/env python3
"""
Sparse Gene Expression Quality Analysis
For data with many zeros, focusing on non-zero expression patterns and consistency
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

def analyze_sparse_patterns(data):
    """Analyze patterns in sparse gene expression data"""
    print("Sparse 패턴 분석 중...")
    
    results = []
    
    for gene in data.index:
        gene_values = data.loc[gene].values
        
        # Basic statistics
        n_total = len(gene_values)
        n_non_zero = np.sum(gene_values > 0)
        n_zero = n_total - n_non_zero
        
        if n_non_zero == 0:
            # All zeros
            results.append({
                'gene': gene,
                'n_total': n_total,
                'n_non_zero': 0,
                'n_zero': n_zero,
                'sparsity': 1.0,
                'mean_expression': 0.0,
                'std_expression': 0.0,
                'max_expression': 0.0,
                'min_expression': 0.0,
                'expression_range': 0.0,
                'cv': np.nan,
                'consistency_score': 0.0,
                'expression_stability': 0.0,
                'is_expressed': False,
                'is_highly_expressed': False,
                'is_consistent': False
            })
            continue
        
        # Non-zero values analysis
        non_zero_values = gene_values[gene_values > 0]
        
        mean_expr = np.mean(non_zero_values)
        std_expr = np.std(non_zero_values)
        max_expr = np.max(non_zero_values)
        min_expr = np.min(non_zero_values)
        expr_range = max_expr - min_expr
        
        # Coefficient of variation (for non-zero values)
        cv = std_expr / mean_expr if mean_expr > 0 else np.nan
        
        # Sparsity
        sparsity = n_zero / n_total
        
        # Consistency score: based on how consistent non-zero values are
        if len(non_zero_values) >= 2:
            # Calculate consistency as inverse of coefficient of variation
            consistency_score = 1.0 / (1.0 + cv) if not np.isnan(cv) else 0.0
        else:
            consistency_score = 0.0
        
        # Expression stability: based on range relative to mean
        if mean_expr > 0:
            expression_stability = 1.0 - (expr_range / mean_expr)
            expression_stability = max(0.0, expression_stability)
        else:
            expression_stability = 0.0
        
        # Classification
        is_expressed = n_non_zero >= 1
        is_highly_expressed = n_non_zero >= 3 and mean_expr > np.percentile([np.mean(data.loc[g].values[data.loc[g].values > 0]) 
                                                                          for g in data.index 
                                                                          if np.sum(data.loc[g].values > 0) > 0], 75)
        is_consistent = consistency_score > 0.7 and n_non_zero >= 3
        
        results.append({
            'gene': gene,
            'n_total': n_total,
            'n_non_zero': n_non_zero,
            'n_zero': n_zero,
            'sparsity': sparsity,
            'mean_expression': mean_expr,
            'std_expression': std_expr,
            'max_expression': max_expr,
            'min_expression': min_expr,
            'expression_range': expr_range,
            'cv': cv,
            'consistency_score': consistency_score,
            'expression_stability': expression_stability,
            'is_expressed': is_expressed,
            'is_highly_expressed': is_highly_expressed,
            'is_consistent': is_consistent
        })
    
    return pd.DataFrame(results)

def identify_high_quality_genes(results_df):
    """Identify high-quality genes based on sparse data criteria"""
    print("고품질 genes 식별 중...")
    
    # Criteria for high-quality genes in sparse data:
    # 1. Expressed in multiple patches (n_non_zero >= 3)
    # 2. Consistent expression pattern (consistency_score > 0.6)
    # 3. Reasonable expression stability (expression_stability > 0.5)
    # 4. Not too sparse (sparsity < 0.9)
    
    high_quality = results_df[
        (results_df['n_non_zero'] >= 3) &
        (results_df['consistency_score'] > 0.6) &
        (results_df['expression_stability'] > 0.5) &
        (results_df['sparsity'] < 0.9)
    ].copy()
    
    # Sort by consistency score
    high_quality = high_quality.sort_values('consistency_score', ascending=False)
    
    print(f"고품질 genes: {len(high_quality)} genes")
    
    return high_quality

def create_sparse_visualizations(results_df, high_quality_df, out_dir):
    """Create visualizations for sparse data analysis"""
    print("시각화 생성 중...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Sparsity distribution
    axes[0, 0].hist(results_df['sparsity'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution of Sparsity')
    axes[0, 0].set_xlabel('Sparsity (proportion of zeros)')
    axes[0, 0].set_ylabel('Number of genes')
    
    # Non-zero count distribution
    axes[0, 1].hist(results_df['n_non_zero'], bins=min(50, results_df['n_non_zero'].max()+1), 
                   alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Distribution of Non-zero Patches')
    axes[0, 1].set_xlabel('Number of non-zero patches')
    axes[0, 1].set_ylabel('Number of genes')
    
    # Mean expression distribution (non-zero genes only)
    non_zero_genes = results_df[results_df['n_non_zero'] > 0]
    if len(non_zero_genes) > 0:
        axes[0, 2].hist(non_zero_genes['mean_expression'], bins=50, alpha=0.7, color='orange')
        axes[0, 2].set_title('Distribution of Mean Expression (Non-zero genes)')
        axes[0, 2].set_xlabel('Mean expression level')
        axes[0, 2].set_ylabel('Number of genes')
    
    # Consistency score distribution
    valid_consistency = results_df[~np.isnan(results_df['consistency_score'])]
    if len(valid_consistency) > 0:
        axes[1, 0].hist(valid_consistency['consistency_score'], bins=50, alpha=0.7, color='purple')
        axes[1, 0].set_title('Distribution of Consistency Scores')
        axes[1, 0].set_xlabel('Consistency score')
        axes[1, 0].set_ylabel('Number of genes')
    
    # Expression stability distribution
    valid_stability = results_df[~np.isnan(results_df['expression_stability'])]
    if len(valid_stability) > 0:
        axes[1, 1].hist(valid_stability['expression_stability'], bins=50, alpha=0.7, color='red')
        axes[1, 1].set_title('Distribution of Expression Stability')
        axes[1, 1].set_xlabel('Expression stability')
        axes[1, 1].set_ylabel('Number of genes')
    
    # Quality categories
    categories = ['All zeros', 'Low expression', 'Moderate expression', 'High expression', 'High quality']
    counts = [
        np.sum(results_df['n_non_zero'] == 0),
        np.sum((results_df['n_non_zero'] >= 1) & (results_df['n_non_zero'] < 3)),
        np.sum((results_df['n_non_zero'] >= 3) & (results_df['n_non_zero'] < 10)),
        np.sum(results_df['n_non_zero'] >= 10),
        len(high_quality_df)
    ]
    
    axes[1, 2].bar(categories, counts, color=['gray', 'lightblue', 'blue', 'darkblue', 'gold'])
    axes[1, 2].set_title('Gene Expression Categories')
    axes[1, 2].set_ylabel('Number of genes')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sparse_data_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plots for high-quality genes
    if len(high_quality_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Consistency vs Sparsity
        axes[0].scatter(results_df['sparsity'], results_df['consistency_score'], 
                       alpha=0.5, c='lightgray', label='All genes')
        axes[0].scatter(high_quality_df['sparsity'], high_quality_df['consistency_score'], 
                       alpha=0.8, c='red', s=50, label=f'High quality (n={len(high_quality_df)})')
        axes[0].set_xlabel('Sparsity')
        axes[0].set_ylabel('Consistency Score')
        axes[0].set_title('Consistency vs Sparsity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Expression stability vs Mean expression
        axes[1].scatter(results_df['mean_expression'], results_df['expression_stability'], 
                       alpha=0.5, c='lightgray', label='All genes')
        axes[1].scatter(high_quality_df['mean_expression'], high_quality_df['expression_stability'], 
                       alpha=0.8, c='red', s=50, label=f'High quality (n={len(high_quality_df)})')
        axes[1].set_xlabel('Mean Expression')
        axes[1].set_ylabel('Expression Stability')
        axes[1].set_title('Expression Stability vs Mean Expression')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'high_quality_genes_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"시각화 완료: {out_dir}")

def generate_sparse_summary_report(results_df, high_quality_df, out_dir):
    """Generate summary report for sparse data analysis"""
    print("요약 리포트 생성 중...")
    
    report_path = os.path.join(out_dir, 'sparse_analysis_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Sparse Gene Expression Analysis Report ===\n")
        f.write("Methodology: Adapted for highly sparse data with many zeros\n\n")
        
        f.write(f"Total genes analyzed: {len(results_df)}\n")
        f.write(f"High-quality genes: {len(high_quality_df)}\n")
        if len(results_df) > 0:
            f.write(f"Success rate: {len(high_quality_df)/len(results_df)*100:.2f}%\n\n")
        else:
            f.write("Success rate: N/A\n\n")
        
        f.write("=== Data Characteristics ===\n")
        f.write(f"Genes with all zeros: {np.sum(results_df['n_non_zero'] == 0)}\n")
        f.write(f"Genes with 1-2 non-zero patches: {np.sum((results_df['n_non_zero'] >= 1) & (results_df['n_non_zero'] < 3))}\n")
        f.write(f"Genes with 3-9 non-zero patches: {np.sum((results_df['n_non_zero'] >= 3) & (results_df['n_non_zero'] < 10))}\n")
        f.write(f"Genes with 10+ non-zero patches: {np.sum(results_df['n_non_zero'] >= 10)}\n\n")
        
        f.write("=== High-Quality Gene Criteria ===\n")
        f.write("1. n_non_zero >= 3 (expressed in multiple patches)\n")
        f.write("2. consistency_score > 0.6 (consistent expression pattern)\n")
        f.write("3. expression_stability > 0.5 (stable expression levels)\n")
        f.write("4. sparsity < 0.9 (not too sparse)\n\n")
        
        f.write("=== Summary Statistics ===\n")
        f.write(f"Mean sparsity: {results_df['sparsity'].mean():.3f}\n")
        f.write(f"Mean non-zero patches per gene: {results_df['n_non_zero'].mean():.2f}\n")
        
        non_zero_genes = results_df[results_df['n_non_zero'] > 0]
        if len(non_zero_genes) > 0:
            f.write(f"Mean expression (non-zero genes): {non_zero_genes['mean_expression'].mean():.4f}\n")
            f.write(f"Mean consistency score: {non_zero_genes['consistency_score'].mean():.3f}\n")
        
        if len(high_quality_df) > 0:
            f.write(f"Mean consistency score (high-quality): {high_quality_df['consistency_score'].mean():.3f}\n")
            f.write(f"Mean expression stability (high-quality): {high_quality_df['expression_stability'].mean():.3f}\n\n")
            
            f.write("=== Top 10 High-Quality Genes ===\n")
            top_genes = high_quality_df.head(10)
            for i, (_, row) in enumerate(top_genes.iterrows(), 1):
                f.write(f"{i:2d}. {row['gene']:15s}: "
                       f"non_zero={int(row['n_non_zero']):2d}, "
                       f"consistency={row['consistency_score']:.3f}, "
                       f"stability={row['expression_stability']:.3f}, "
                       f"mean_expr={row['mean_expression']:.4f}\n")
        else:
            f.write("\nNo high-quality genes found.\n")
    
    print(f"요약 리포트 저장: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze sparse gene expression data quality')
    parser.add_argument('--pred_csv', required=True, help='Path to predicted gene expression CSV')
    parser.add_argument('--out_dir', default='results_sparse_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=== Sparse Gene Expression Quality Analysis ===")
    print(f"Input file: {args.pred_csv}")
    print(f"Output directory: {args.out_dir}")
    print()
    
    # Load data
    data = load_and_orient_data(args.pred_csv)
    
    # Analyze sparse patterns
    results_df = analyze_sparse_patterns(data)
    
    # Identify high-quality genes
    high_quality_df = identify_high_quality_genes(results_df)
    
    # Save results
    results_df.to_csv(os.path.join(args.out_dir, 'sparse_analysis_results.csv'), index=False)
    if len(high_quality_df) > 0:
        high_quality_df.to_csv(os.path.join(args.out_dir, 'high_quality_genes_sparse.csv'), index=False)
    
    # Create visualizations
    create_sparse_visualizations(results_df, high_quality_df, args.out_dir)
    
    # Generate summary report
    generate_sparse_summary_report(results_df, high_quality_df, args.out_dir)
    
    print(f"\n분석 완료! 결과는 {args.out_dir}에 저장되었습니다.")
    print(f"- 총 {len(results_df)} genes 분석")
    print(f"- {len(high_quality_df)} genes가 고품질로 식별됨")
    
    # Print summary to console
    print(f"\n=== 요약 ===")
    print(f"모든 값이 0인 genes: {np.sum(results_df['n_non_zero'] == 0)}")
    print(f"1-2개 non-zero patches: {np.sum((results_df['n_non_zero'] >= 1) & (results_df['n_non_zero'] < 3))}")
    print(f"3-9개 non-zero patches: {np.sum((results_df['n_non_zero'] >= 3) & (results_df['n_non_zero'] < 10))}")
    print(f"10개 이상 non-zero patches: {np.sum(results_df['n_non_zero'] >= 10)}")
    print(f"고품질 genes: {len(high_quality_df)}")

if __name__ == "__main__":
    main()
