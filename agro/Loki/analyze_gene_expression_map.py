#!/usr/bin/env python3
"""
Gene Expression Map Analysis Script
CSV 파일을 분석하여 상세한 통계와 인사이트를 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data(csv_path):
    """데이터를 로드하고 기본 분석을 수행합니다."""
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    
    print(f"\n=== Data Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Patches: {df.shape[0]}")
    print(f"Genes: {df.shape[1]}")
    print(f"Expression range: [{df.values.min():.4f}, {df.values.max():.4f}]")
    print(f"Mean expression: {df.values.mean():.4f}")
    print(f"Std expression: {df.values.std():.4f}")
    
    # 발현 통계
    non_zero_count = (df > 0).sum().sum()
    total_count = df.size
    sparsity = 1 - (non_zero_count / total_count)
    
    print(f"\n=== Expression Statistics ===")
    print(f"Non-zero values: {non_zero_count:,}")
    print(f"Total values: {total_count:,}")
    print(f"Sparsity: {sparsity:.2%}")
    
    return df

def analyze_gene_expression_patterns(df):
    """유전자 발현 패턴을 분석합니다."""
    
    print(f"\n=== Gene Expression Pattern Analysis ===")
    
    # 유전자별 통계
    gene_stats = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'max': df.max(),
        'min': df.min(),
        'non_zero_count': (df > 0).sum(),
        'non_zero_ratio': (df > 0).mean()
    })
    
    # 상위 발현 유전자
    top_mean_genes = gene_stats.nlargest(10, 'mean')
    top_max_genes = gene_stats.nlargest(10, 'max')
    top_expressed_genes = gene_stats.nlargest(10, 'non_zero_count')
    
    print(f"\nTop 10 genes by mean expression:")
    for i, (gene, stats) in enumerate(top_mean_genes.iterrows(), 1):
        print(f"{i:2d}. {gene:8s}: {stats['mean']:.4f}")
    
    print(f"\nTop 10 genes by max expression:")
    for i, (gene, stats) in enumerate(top_max_genes.iterrows(), 1):
        print(f"{i:2d}. {gene:8s}: {stats['max']:.4f}")
    
    print(f"\nTop 10 genes by expression frequency:")
    for i, (gene, stats) in enumerate(top_expressed_genes.iterrows(), 1):
        print(f"{i:2d}. {gene:8s}: {int(stats['non_zero_count']):2d} patches ({stats['non_zero_ratio']:.1%})")
    
    return gene_stats

def analyze_patch_expression_patterns(df):
    """패치 발현 패턴을 분석합니다."""
    
    print(f"\n=== Patch Expression Pattern Analysis ===")
    
    # 패치별 통계
    patch_stats = pd.DataFrame({
        'mean': df.mean(axis=1),
        'std': df.std(axis=1),
        'max': df.max(axis=1),
        'min': df.min(axis=1),
        'non_zero_count': (df > 0).sum(axis=1),
        'non_zero_ratio': (df > 0).mean(axis=1)
    })
    
    # 상위 발현 패치
    top_mean_patches = patch_stats.nlargest(10, 'mean')
    top_max_patches = patch_stats.nlargest(10, 'max')
    top_expressed_patches = patch_stats.nlargest(10, 'non_zero_count')
    
    print(f"\nTop 10 patches by mean expression:")
    for i, (patch, stats) in enumerate(top_mean_patches.iterrows(), 1):
        print(f"{i:2d}. {patch:15s}: {stats['mean']:.4f}")
    
    print(f"\nTop 10 patches by max expression:")
    for i, (patch, stats) in enumerate(top_max_patches.iterrows(), 1):
        print(f"{patch:15s}: {stats['max']:.4f}")
    
    print(f"\nTop 10 patches by expression frequency:")
    for i, (patch, stats) in enumerate(top_expressed_patches.iterrows(), 1):
        print(f"{i:2d}. {patch:15s}: {int(stats['non_zero_count']):4d} genes ({stats['non_zero_ratio']:.1%})")
    
    return patch_stats

def perform_pca_analysis(df, n_components=10):
    """PCA 분석을 수행합니다."""
    
    print(f"\n=== PCA Analysis ===")
    
    # PCA 수행
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df.T)  # 유전자를 행으로 변환
    
    # 설명 분산 비율
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"Explained variance ratio:")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
        print(f"PC{i+1:2d}: {var:.3f} (cumulative: {cum_var:.3f})")
    
    # 95% 분산을 설명하는 주성분 수
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"\nNumber of components explaining 95% variance: {n_components_95}")
    
    return pca, pca_result, explained_variance_ratio

def perform_clustering_analysis(df, n_clusters=5):
    """클러스터링 분석을 수행합니다."""
    
    print(f"\n=== Clustering Analysis ===")
    
    # 패치별 클러스터링
    kmeans_patches = KMeans(n_clusters=n_clusters, random_state=42)
    patch_clusters = kmeans_patches.fit_predict(df)
    
    # 유전자별 클러스터링 (발현이 있는 유전자만)
    expressed_genes = df.columns[(df > 0).any()]
    df_expressed = df[expressed_genes]
    
    kmeans_genes = KMeans(n_clusters=n_clusters, random_state=42)
    gene_clusters = kmeans_genes.fit_predict(df_expressed.T)
    
    print(f"Patch clustering (K-means, k={n_clusters}):")
    unique, counts = np.unique(patch_clusters, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {int(count)} patches")
    
    print(f"\nGene clustering (K-means, k={n_clusters}):")
    unique, counts = np.unique(gene_clusters, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {int(count)} genes")
    
    return patch_clusters, gene_clusters

def find_correlated_genes(df, top_n=20):
    """상관관계가 높은 유전자 쌍을 찾습니다."""
    
    print(f"\n=== Gene Correlation Analysis ===")
    
    # 발현이 있는 유전자만 선택
    expressed_genes = df.columns[(df > 0).any()]
    df_expressed = df[expressed_genes]
    
    # 상관계수 계산
    correlation_matrix = df_expressed.corr()
    
    # 상삼각 행렬에서 상관계수 추출
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # 상관계수가 높은 쌍 찾기
    high_corr_pairs = []
    for i in range(len(upper_tri.columns)):
        for j in range(i):
            corr_val = upper_tri.iloc[i, j]
            if not np.isnan(corr_val) and abs(corr_val) > 0.5:  # 임계값 0.5
                high_corr_pairs.append((upper_tri.columns[i], upper_tri.index[j], corr_val))
    
    # 상관계수 기준으로 정렬
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"Top {min(top_n, len(high_corr_pairs))} highly correlated gene pairs:")
    for i, (gene1, gene2, corr) in enumerate(high_corr_pairs[:top_n], 1):
        print(f"{i:2d}. {gene1:8s} - {gene2:8s}: {corr:.3f}")
    
    return high_corr_pairs

def create_comprehensive_visualization(df, gene_stats, patch_stats, pca_result, explained_variance_ratio):
    """종합적인 시각화를 생성합니다."""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 유전자 발현 분포
    ax1 = plt.subplot(3, 4, 1)
    gene_means = gene_stats['mean']
    ax1.hist(gene_means, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_title('Distribution of Gene Mean Expression')
    ax1.set_xlabel('Mean Expression')
    ax1.set_ylabel('Number of Genes')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. 패치 발현 분포
    ax2 = plt.subplot(3, 4, 2)
    patch_means = patch_stats['mean']
    ax2.hist(patch_means, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Distribution of Patch Mean Expression')
    ax2.set_xlabel('Mean Expression')
    ax2.set_ylabel('Number of Patches')
    ax2.grid(True, alpha=0.3)
    
    # 3. 발현 빈도 분포
    ax3 = plt.subplot(3, 4, 3)
    non_zero_ratios = gene_stats['non_zero_ratio']
    ax3.hist(non_zero_ratios, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
    ax3.set_title('Distribution of Gene Expression Frequency')
    ax3.set_xlabel('Expression Frequency')
    ax3.set_ylabel('Number of Genes')
    ax3.grid(True, alpha=0.3)
    
    # 4. PCA 설명 분산
    ax4 = plt.subplot(3, 4, 4)
    ax4.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 
            alpha=0.7, color='purple')
    ax4.set_title('PCA Explained Variance Ratio')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Explained Variance Ratio')
    ax4.grid(True, alpha=0.3)
    
    # 5. 상위 발현 유전자
    ax5 = plt.subplot(3, 4, 5)
    top_genes = gene_stats.nlargest(15, 'mean')
    ax5.barh(range(len(top_genes)), top_genes['mean'], alpha=0.7, color='gold')
    ax5.set_yticks(range(len(top_genes)))
    ax5.set_yticklabels(top_genes.index)
    ax5.set_title('Top 15 Genes by Mean Expression')
    ax5.set_xlabel('Mean Expression')
    ax5.grid(True, alpha=0.3)
    
    # 6. 상위 발현 패치
    ax6 = plt.subplot(3, 4, 6)
    top_patches = patch_stats.nlargest(15, 'mean')
    ax6.barh(range(len(top_patches)), top_patches['mean'], alpha=0.7, color='orange')
    ax6.set_yticks(range(len(top_patches)))
    ax6.set_yticklabels([p[:10] + '...' if len(p) > 10 else p for p in top_patches.index])
    ax6.set_title('Top 15 Patches by Mean Expression')
    ax6.set_xlabel('Mean Expression')
    ax6.grid(True, alpha=0.3)
    
    # 7. PCA 스코어 플롯
    ax7 = plt.subplot(3, 4, 7)
    ax7.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, color='red')
    ax7.set_title('PCA Score Plot (PC1 vs PC2)')
    ax7.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
    ax7.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
    ax7.grid(True, alpha=0.3)
    
    # 8. 발현 수준 vs 빈도
    ax8 = plt.subplot(3, 4, 8)
    ax8.scatter(gene_stats['mean'], gene_stats['non_zero_ratio'], 
                alpha=0.6, color='green')
    ax8.set_title('Gene Mean Expression vs Frequency')
    ax8.set_xlabel('Mean Expression')
    ax8.set_ylabel('Expression Frequency')
    ax8.grid(True, alpha=0.3)
    
    # 9. 패치별 발현 유전자 수
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(patch_stats['non_zero_count'], bins=20, alpha=0.7, color='brown', edgecolor='black')
    ax9.set_title('Distribution of Expressed Genes per Patch')
    ax9.set_xlabel('Number of Expressed Genes')
    ax9.set_ylabel('Number of Patches')
    ax9.grid(True, alpha=0.3)
    
    # 10. 발현 수준 히트맵 (상위 유전자)
    ax10 = plt.subplot(3, 4, 10)
    top_genes_for_heatmap = gene_stats.nlargest(20, 'mean').index
    heatmap_data = df[top_genes_for_heatmap].T
    im = ax10.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax10.set_title('Expression Heatmap (Top 20 Genes)')
    ax10.set_xlabel('Patches')
    ax10.set_ylabel('Genes')
    ax10.set_yticks(range(len(top_genes_for_heatmap)))
    ax10.set_yticklabels(top_genes_for_heatmap)
    
    # 11. 발현 수준 히트맵 (상위 패치)
    ax11 = plt.subplot(3, 4, 11)
    top_patches_for_heatmap = patch_stats.nlargest(20, 'mean').index
    heatmap_data2 = df.loc[top_patches_for_heatmap]
    # 상위 발현 유전자만 선택
    top_genes_for_heatmap2 = gene_stats.nlargest(30, 'mean').index
    heatmap_data2 = heatmap_data2[top_genes_for_heatmap2]
    im2 = ax11.imshow(heatmap_data2, cmap='viridis', aspect='auto')
    ax11.set_title('Expression Heatmap (Top 20 Patches)')
    ax11.set_xlabel('Genes')
    ax11.set_ylabel('Patches')
    ax11.set_xticks(range(0, len(top_genes_for_heatmap2), 5))
    ax11.set_xticklabels([top_genes_for_heatmap2[i] for i in range(0, len(top_genes_for_heatmap2), 5)], 
                         rotation=45, ha='right')
    
    # 12. 누적 설명 분산
    ax12 = plt.subplot(3, 4, 12)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    ax12.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 
              marker='o', color='purple', linewidth=2)
    ax12.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95%')
    ax12.set_title('Cumulative Explained Variance')
    ax12.set_xlabel('Number of Components')
    ax12.set_ylabel('Cumulative Explained Variance')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """메인 함수"""
    
    csv_path = '/home/students/hbae/Loki/gene_expression_maps_TCGA-QK-A6IH-01Z-00-DX1.csv'
    
    print("=" * 80)
    print("Gene Expression Map Comprehensive Analysis")
    print("=" * 80)
    
    # 데이터 로드 및 기본 분석
    df = load_and_analyze_data(csv_path)
    
    # 유전자 발현 패턴 분석
    gene_stats = analyze_gene_expression_patterns(df)
    
    # 패치 발현 패턴 분석
    patch_stats = analyze_patch_expression_patterns(df)
    
    # PCA 분석
    pca, pca_result, explained_variance_ratio = perform_pca_analysis(df)
    
    # 클러스터링 분석
    patch_clusters, gene_clusters = perform_clustering_analysis(df)
    
    # 상관관계 분석
    high_corr_pairs = find_correlated_genes(df)
    
    # 종합 시각화
    print(f"\n=== Creating Comprehensive Visualization ===")
    fig = create_comprehensive_visualization(df, gene_stats, patch_stats, pca_result, explained_variance_ratio)
    fig.savefig('/home/students/hbae/Loki/comprehensive_analysis.png', 
                 dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("Generated files:")
    print("- comprehensive_analysis.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
