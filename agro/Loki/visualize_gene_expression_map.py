#!/usr/bin/env python3
"""
Gene Expression Map Visualization Script
CSV 파일을 읽어서 gene expression map을 시각화합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def load_gene_expression_data(csv_path):
    """CSV 파일에서 gene expression 데이터를 로드합니다."""
    print(f"Loading gene expression data from: {csv_path}")
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path, index_col=0)
    
    print(f"Data shape: {df.shape}")
    print(f"Number of patches: {df.shape[0]}")
    print(f"Number of genes: {df.shape[1]}")
    print(f"Expression value range: [{df.values.min():.4f}, {df.values.max():.4f}]")
    
    return df

def create_gene_expression_heatmap(df, genes_to_plot=None, figsize=(15, 10)):
    """Gene expression heatmap을 생성합니다."""
    
    if genes_to_plot is None:
        # 발현 수준이 높은 상위 20개 유전자 선택
        gene_means = df.mean().sort_values(ascending=False)
        genes_to_plot = gene_means.head(20).index.tolist()
    
    # 선택된 유전자들의 데이터만 추출
    plot_data = df[genes_to_plot]
    
    # 히트맵 생성
    plt.figure(figsize=figsize)
    sns.heatmap(plot_data.T, 
                cmap='viridis', 
                cbar_kws={'label': 'Gene Expression Level'},
                xticklabels=False,  # 패치 이름은 너무 많아서 숨김
                yticklabels=True)
    
    plt.title(f'Gene Expression Map - Top {len(genes_to_plot)} Genes', fontsize=16, fontweight='bold')
    plt.xlabel('Image Patches', fontsize=12)
    plt.ylabel('Genes', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()

def create_patch_expression_distribution(df, figsize=(12, 8)):
    """각 패치의 발현 분포를 시각화합니다."""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 각 패치의 평균 발현 수준
    patch_means = df.mean(axis=1)
    axes[0, 0].hist(patch_means, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Mean Expression per Patch')
    axes[0, 0].set_xlabel('Mean Expression Level')
    axes[0, 0].set_ylabel('Number of Patches')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 각 패치의 최대 발현 수준
    patch_max = df.max(axis=1)
    axes[0, 1].hist(patch_max, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Distribution of Max Expression per Patch')
    axes[0, 1].set_xlabel('Max Expression Level')
    axes[0, 1].set_ylabel('Number of Patches')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 각 패치의 발현 유전자 수 (0이 아닌 값)
    patch_nonzero = (df > 0).sum(axis=1)
    axes[1, 0].hist(patch_nonzero, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Distribution of Expressed Genes per Patch')
    axes[1, 0].set_xlabel('Number of Expressed Genes')
    axes[1, 0].set_ylabel('Number of Patches')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 전체 발현 분포
    all_values = df.values.flatten()
    axes[1, 1].hist(all_values, bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_title('Overall Expression Distribution')
    axes[1, 1].set_xlabel('Expression Level')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_gene_expression_statistics(df, figsize=(12, 8)):
    """유전자별 발현 통계를 시각화합니다."""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 유전자별 평균 발현 수준 (상위 30개)
    gene_means = df.mean().sort_values(ascending=False)
    top_genes = gene_means.head(30)
    
    axes[0, 0].bar(range(len(top_genes)), top_genes.values, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Top 30 Genes by Mean Expression')
    axes[0, 0].set_xlabel('Genes')
    axes[0, 0].set_ylabel('Mean Expression Level')
    axes[0, 0].set_xticks(range(len(top_genes)))
    axes[0, 0].set_xticklabels(top_genes.index, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 유전자별 최대 발현 수준 (상위 30개)
    gene_max = df.max().sort_values(ascending=False)
    top_max_genes = gene_max.head(30)
    
    axes[0, 1].bar(range(len(top_max_genes)), top_max_genes.values, color='darkorange', alpha=0.7)
    axes[0, 1].set_title('Top 30 Genes by Max Expression')
    axes[0, 1].set_xlabel('Genes')
    axes[0, 1].set_ylabel('Max Expression Level')
    axes[0, 1].set_xticks(range(len(top_max_genes)))
    axes[0, 1].set_xticklabels(top_max_genes.index, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 유전자별 발현 패치 수 (상위 30개)
    gene_nonzero = (df > 0).sum().sort_values(ascending=False)
    top_nonzero_genes = gene_nonzero.head(30)
    
    axes[1, 0].bar(range(len(top_nonzero_genes)), top_nonzero_genes.values, color='forestgreen', alpha=0.7)
    axes[1, 0].set_title('Top 30 Genes by Number of Expressed Patches')
    axes[1, 0].set_xlabel('Genes')
    axes[1, 0].set_ylabel('Number of Expressed Patches')
    axes[1, 0].set_xticks(range(len(top_nonzero_genes)))
    axes[1, 0].set_xticklabels(top_nonzero_genes.index, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 발현 수준 분포 (로그 스케일)
    gene_means_log = np.log10(gene_means + 1e-10)  # 0을 피하기 위해 작은 값 추가
    axes[1, 1].hist(gene_means_log, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Distribution of Mean Expression (Log Scale)')
    axes[1, 1].set_xlabel('Log10(Mean Expression + 1e-10)')
    axes[1, 1].set_ylabel('Number of Genes')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_specific_gene_expression_map(df, gene_name, figsize=(12, 6)):
    """특정 유전자의 발현 패턴을 시각화합니다."""
    
    if gene_name not in df.columns:
        print(f"Gene '{gene_name}' not found in the data.")
        return None
    
    gene_expression = df[gene_name]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. 패치별 발현 수준 (바 차트)
    patch_indices = range(len(gene_expression))
    bars = ax1.bar(patch_indices, gene_expression.values, 
                   color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_title(f'Expression Pattern of {gene_name}')
    ax1.set_xlabel('Patch Index')
    ax1.set_ylabel('Expression Level')
    ax1.grid(True, alpha=0.3)
    
    # 발현 수준이 높은 패치들을 강조
    high_expr_threshold = gene_expression.quantile(0.8)
    for i, (idx, val) in enumerate(zip(patch_indices, gene_expression.values)):
        if val > high_expr_threshold:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
    
    # 2. 발현 분포 히스토그램
    ax2.hist(gene_expression.values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(gene_expression.mean(), color='red', linestyle='--', 
                label=f'Mean: {gene_expression.mean():.4f}')
    ax2.axvline(gene_expression.median(), color='blue', linestyle='--', 
                label=f'Median: {gene_expression.median():.4f}')
    ax2.set_title(f'Expression Distribution of {gene_name}')
    ax2.set_xlabel('Expression Level')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_spatial_expression_map(df, figsize=(15, 10)):
    """공간적 발현 패턴을 시각화합니다 (패치를 2D로 배열)."""
    
    n_patches = df.shape[0]
    
    # 패치를 2D 그리드로 배열 (가능한 정사각형에 가깝게)
    grid_size = int(np.ceil(np.sqrt(n_patches)))
    
    # 각 패치의 평균 발현 수준 계산
    patch_means = df.mean(axis=1).values
    
    # 2D 그리드 생성
    spatial_map = np.zeros((grid_size, grid_size))
    spatial_map[:] = np.nan  # 빈 공간은 NaN으로 설정
    
    # 패치 데이터를 그리드에 배치
    for i, mean_expr in enumerate(patch_means):
        row = i // grid_size
        col = i % grid_size
        if row < grid_size and col < grid_size:
            spatial_map[row, col] = mean_expr
    
    # 시각화
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(spatial_map, cmap='viridis', aspect='equal')
    
    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Expression Level', rotation=270, labelpad=20)
    
    # 제목과 라벨
    ax.set_title('Spatial Gene Expression Map', fontsize=16, fontweight='bold')
    ax.set_xlabel('Spatial Position (X)')
    ax.set_ylabel('Spatial Position (Y)')
    
    # 그리드 표시
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """메인 함수"""
    
    # CSV 파일 경로
    csv_path = '/home/students/hbae/Loki/gene_expression_maps_TCGA-QK-A6IH-01Z-00-DX1.csv'
    
    print("=" * 60)
    print("Gene Expression Map Visualization")
    print("=" * 60)
    
    # 데이터 로드
    df = load_gene_expression_data(csv_path)
    
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    
    # 1. Gene Expression Heatmap
    print("1. Creating gene expression heatmap...")
    fig1 = create_gene_expression_heatmap(df)
    fig1.savefig('/home/students/hbae/Loki/gene_expression_heatmap.png', 
                  dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Patch Expression Distribution
    print("2. Creating patch expression distribution...")
    fig2 = create_patch_expression_distribution(df)
    fig2.savefig('/home/students/hbae/Loki/patch_expression_distribution.png', 
                  dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Gene Expression Statistics
    print("3. Creating gene expression statistics...")
    fig3 = create_gene_expression_statistics(df)
    fig3.savefig('/home/students/hbae/Loki/gene_expression_statistics.png', 
                  dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Spatial Expression Map
    print("4. Creating spatial expression map...")
    fig4 = create_spatial_expression_map(df)
    fig4.savefig('/home/students/hbae/Loki/spatial_expression_map.png', 
                  dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 특정 유전자 발현 패턴 (상위 5개 유전자)
    print("5. Creating specific gene expression patterns...")
    top_genes = df.mean().sort_values(ascending=False).head(5).index
    for i, gene in enumerate(top_genes):
        print(f"   - {gene}")
        fig5 = create_specific_gene_expression_map(df, gene)
        if fig5 is not None:
            fig5.savefig(f'/home/students/hbae/Loki/gene_{gene}_expression.png', 
                          dpi=300, bbox_inches='tight')
            plt.show()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("Saved files:")
    print("- gene_expression_heatmap.png")
    print("- patch_expression_distribution.png")
    print("- gene_expression_statistics.png")
    print("- spatial_expression_map.png")
    print("- gene_[GENE_NAME]_expression.png (for top 5 genes)")
    print("=" * 60)

if __name__ == "__main__":
    main()
