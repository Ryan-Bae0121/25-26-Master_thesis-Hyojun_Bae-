#!/usr/bin/env python3
"""
도메인 시프트 진단 결과를 바탕으로 Gene Expression Map 생성
Domain Shift Gene Expression Map Creator

이 스크립트는 retrieval_ood_diagnostics.py의 결과를 바탕으로
도메인 시프트가 감지된 상황에서의 gene expression 패턴을 분석하고 시각화합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_domain_shift_results(ood_dir):
    """도메인 시프트 진단 결과 로드"""
    summary_path = os.path.join(ood_dir, 'summary.json')
    weights_path = os.path.join(ood_dir, 'example_patch_weights_topk.csv')
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    weights_df = None
    if os.path.exists(weights_path):
        weights_df = pd.read_csv(weights_path)
    
    return summary, weights_df

def load_gene_expression_data(expr_path):
    """Gene expression 데이터 로드"""
    print(f"Loading gene expression data from {expr_path}")
    expr_data = pd.read_csv(expr_path, index_col=0)
    print(f"Gene expression data shape: {expr_data.shape}")
    return expr_data

def analyze_domain_shift_patterns(expr_data, summary):
    """도메인 시프트 패턴 분석"""
    print("\n=== 도메인 시프트 패턴 분석 ===")
    
    # 기본 통계
    n_patches, n_genes = expr_data.shape
    print(f"총 패치 수: {n_patches}")
    print(f"총 유전자 수: {n_genes}")
    
    # 스파시티 분석
    sparsity = (expr_data == 0).sum().sum() / (n_patches * n_genes)
    print(f"전체 스파시티: {sparsity:.4f} ({sparsity*100:.2f}%)")
    
    # 유전자별 통계
    gene_stats = {
        'mean_expression': expr_data.mean(),
        'std_expression': expr_data.std(),
        'sparsity': (expr_data == 0).mean(),
        'max_expression': expr_data.max(),
        'cv': expr_data.std() / (expr_data.mean() + 1e-8)  # Coefficient of Variation
    }
    
    gene_stats_df = pd.DataFrame(gene_stats)
    
    # 도메인 시프트 관련 분석
    print(f"\n도메인 시프트 진단 결과:")
    print(f"- 중간값 최대 코사인 유사도: {summary['median_max_cosine_similarity']:.4f}")
    print(f"- 중간값 Top-K 엔트로피: {summary['median_top_k_entropy']:.4f}")
    print(f"- OOD 플래그: {summary['ood_flag']}")
    
    if 'ood_reasons' in summary:
        print(f"- OOD 이유: {', '.join(summary['ood_reasons'])}")
    
    return gene_stats_df

def identify_domain_shift_affected_genes(gene_stats_df, expr_data, threshold_cv=2.0, threshold_sparsity=0.8):
    """도메인 시프트의 영향을 받은 유전자 식별"""
    print("\n=== 도메인 시프트 영향 유전자 식별 ===")
    
    # 높은 변동성을 가진 유전자 (CV > threshold_cv)
    high_variance_genes = gene_stats_df[gene_stats_df['cv'] > threshold_cv].index.tolist()
    print(f"높은 변동성 유전자 (CV > {threshold_cv}): {len(high_variance_genes)}개")
    
    # 높은 스파시티를 가진 유전자 (sparsity > threshold_sparsity)
    high_sparsity_genes = gene_stats_df[gene_stats_df['sparsity'] > threshold_sparsity].index.tolist()
    print(f"높은 스파시티 유전자 (sparsity > {threshold_sparsity}): {len(high_sparsity_genes)}개")
    
    # 일관성 있는 발현을 보이는 유전자 (낮은 CV, 낮은 스파시티)
    consistent_genes = gene_stats_df[
        (gene_stats_df['cv'] < 1.0) & 
        (gene_stats_df['sparsity'] < 0.5)
    ].index.tolist()
    print(f"일관성 있는 발현 유전자: {len(consistent_genes)}개")
    
    return {
        'high_variance': high_variance_genes,
        'high_sparsity': high_sparsity_genes,
        'consistent': consistent_genes
    }

def create_domain_shift_visualizations(expr_data, gene_stats_df, affected_genes, summary, output_dir):
    """도메인 시프트 관련 시각화 생성"""
    print("\n=== 도메인 시프트 시각화 생성 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 전체 발현 분포 히스토그램
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 발현값 분포
    plt.subplot(2, 3, 1)
    non_zero_values = expr_data.values[expr_data.values > 0]
    plt.hist(non_zero_values, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Non-zero Gene Expression Distribution')
    plt.xlabel('Expression Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    
    # 서브플롯 2: 유전자별 스파시티 분포
    plt.subplot(2, 3, 2)
    plt.hist(gene_stats_df['sparsity'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Gene Sparsity Distribution')
    plt.xlabel('Sparsity (Fraction of Zeros)')
    plt.ylabel('Number of Genes')
    
    # 서브플롯 3: 유전자별 변동계수 분포
    plt.subplot(2, 3, 3)
    plt.hist(gene_stats_df['cv'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Gene Coefficient of Variation Distribution')
    plt.xlabel('Coefficient of Variation')
    plt.ylabel('Number of Genes')
    plt.xlim(0, 5)  # CV가 너무 높은 값들 제한
    
    # 서브플롯 4: 스파시티 vs 변동계수 산점도
    plt.subplot(2, 3, 4)
    plt.scatter(gene_stats_df['sparsity'], gene_stats_df['cv'], alpha=0.5, s=1)
    plt.xlabel('Sparsity')
    plt.ylabel('Coefficient of Variation')
    plt.title('Sparsity vs CV')
    
    # 서브플롯 5: 도메인 시프트 영향 유전자 하이라이트
    plt.subplot(2, 3, 5)
    colors = []
    for gene in gene_stats_df.index:
        if gene in affected_genes['high_variance']:
            colors.append('red')
        elif gene in affected_genes['high_sparsity']:
            colors.append('blue')
        elif gene in affected_genes['consistent']:
            colors.append('green')
        else:
            colors.append('gray')
    
    plt.scatter(gene_stats_df['sparsity'], gene_stats_df['cv'], 
                c=colors, alpha=0.6, s=2)
    plt.xlabel('Sparsity')
    plt.ylabel('Coefficient of Variation')
    plt.title('Domain Shift Affected Genes\n(Red: High Variance, Blue: High Sparsity, Green: Consistent)')
    
    # 서브플롯 6: 도메인 시프트 진단 요약
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""
Domain Shift Diagnosis Summary:

OOD Flag: {summary['ood_flag']}
Median Max Cosine Similarity: {summary['median_max_cosine_similarity']:.4f}
Median Top-K Entropy: {summary['median_top_k_entropy']:.4f}

Affected Genes:
- High Variance: {len(affected_genes['high_variance'])}
- High Sparsity: {len(affected_genes['high_sparsity'])}
- Consistent: {len(affected_genes['consistent'])}

Total Patches: {summary['total_patches']}
Total Genes: {summary['total_genes']}
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_shift_gene_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"도메인 시프트 분석 시각화 저장: {output_dir}/domain_shift_gene_analysis.png")

def create_expression_heatmaps(expr_data, affected_genes, output_dir, n_samples=1000):
    """발현 패턴 히트맵 생성"""
    print("\n=== 발현 패턴 히트맵 생성 ===")
    
    # 샘플링 (너무 많은 패치가 있으면 일부만 사용)
    if expr_data.shape[0] > n_samples:
        sample_indices = np.random.choice(expr_data.shape[0], n_samples, replace=False)
        expr_sample = expr_data.iloc[sample_indices]
    else:
        expr_sample = expr_data
    
    # 각 카테고리별로 히트맵 생성
    categories = ['high_variance', 'high_sparsity', 'consistent']
    category_names = ['High Variance Genes', 'High Sparsity Genes', 'Consistent Genes']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (category, name) in enumerate(zip(categories, category_names)):
        genes = affected_genes[category]
        if len(genes) > 0:
            # 상위 50개 유전자만 선택 (너무 많으면 시각화가 어려움)
            if len(genes) > 50:
                # 평균 발현값 기준으로 정렬하여 상위 50개 선택
                mean_expr = expr_sample[genes].mean()
                top_genes = mean_expr.nlargest(50).index.tolist()
            else:
                top_genes = genes
            
            # 히트맵 데이터 준비
            heatmap_data = expr_sample[top_genes].T
            
            # 히트맵 생성
            sns.heatmap(heatmap_data, ax=axes[i], cmap='viridis', 
                       cbar=True, xticklabels=False, yticklabels=True)
            axes[i].set_title(f'{name}\n({len(top_genes)} genes)')
            axes[i].set_xlabel('Patches')
            axes[i].set_ylabel('Genes')
        else:
            axes[i].text(0.5, 0.5, f'No {name.lower()}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{name}\n(0 genes)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_shift_expression_heatmaps.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"발현 패턴 히트맵 저장: {output_dir}/domain_shift_expression_heatmaps.png")

def create_pca_visualization(expr_data, affected_genes, output_dir):
    """PCA를 이용한 공간적 패턴 시각화"""
    print("\n=== PCA 공간적 패턴 시각화 ===")
    
    # PCA 적용
    # 먼저 스케일링
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_data.fillna(0))
    
    # PCA 적용
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(expr_scaled)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    # 전체 패치들
    plt.scatter(pca_result[:, 0], pca_result[:, 1], 
               alpha=0.3, s=1, c='gray', label='All Patches')
    
    # 설명 분산 비율
    explained_var = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({explained_var[0]:.2%} variance explained)')
    plt.ylabel(f'PC2 ({explained_var[1]:.2%} variance explained)')
    
    plt.title('PCA Visualization of Gene Expression Patterns\n(Domain Shift Context)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_shift_pca_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA 시각화 저장: {output_dir}/domain_shift_pca_visualization.png")
    print(f"PC1 설명 분산: {explained_var[0]:.2%}")
    print(f"PC2 설명 분산: {explained_var[1]:.2%}")

def save_gene_expression_maps_to_csv(expr_data, gene_stats_df, affected_genes, summary, output_dir):
    """Gene expression map을 CSV 파일로 저장"""
    print("\n=== Gene Expression Map CSV 파일 생성 ===")
    
    # 1. 전체 gene expression 데이터 저장
    expr_data.to_csv(os.path.join(output_dir, 'gene_expression_map_full.csv'))
    print(f"전체 gene expression map 저장: gene_expression_map_full.csv")
    
    # 2. Gene 통계 정보 저장
    gene_stats_df.to_csv(os.path.join(output_dir, 'gene_statistics_summary.csv'))
    print(f"Gene 통계 요약 저장: gene_statistics_summary.csv")
    
    # 3. 도메인 시프트 영향 유전자별로 분류하여 저장
    # 높은 변동성 유전자
    if len(affected_genes['high_variance']) > 0:
        high_variance_expr = expr_data[affected_genes['high_variance']]
        high_variance_expr.to_csv(os.path.join(output_dir, 'high_variance_genes_expression.csv'))
        print(f"높은 변동성 유전자 발현 저장: high_variance_genes_expression.csv ({len(affected_genes['high_variance'])}개 유전자)")
    
    # 높은 스파시티 유전자 (상위 100개만 저장 - 너무 많으므로)
    if len(affected_genes['high_sparsity']) > 0:
        high_sparsity_subset = affected_genes['high_sparsity'][:100]  # 상위 100개만
        high_sparsity_expr = expr_data[high_sparsity_subset]
        high_sparsity_expr.to_csv(os.path.join(output_dir, 'high_sparsity_genes_expression.csv'))
        print(f"높은 스파시티 유전자 발현 저장: high_sparsity_genes_expression.csv ({len(high_sparsity_subset)}개 유전자)")
    
    # 일관성 있는 발현 유전자
    if len(affected_genes['consistent']) > 0:
        consistent_expr = expr_data[affected_genes['consistent']]
        consistent_expr.to_csv(os.path.join(output_dir, 'consistent_genes_expression.csv'))
        print(f"일관성 있는 발현 유전자 저장: consistent_genes_expression.csv ({len(affected_genes['consistent'])}개 유전자)")
    
    # 4. 도메인 시프트 진단 요약을 CSV로 저장
    diagnosis_summary = {
        'metric': [
            'ood_flag', 'median_max_cosine_similarity', 'median_top_k_entropy', 
            'median_effective_k', 'total_patches', 'total_genes', 'overall_sparsity',
            'high_variance_genes_count', 'high_sparsity_genes_count', 'consistent_genes_count'
        ],
        'value': [
            summary['ood_flag'], summary['median_max_cosine_similarity'], 
            summary['median_top_k_entropy'], summary['median_effective_k'],
            summary['total_patches'], summary['total_genes'], 
            gene_stats_df['sparsity'].mean(),
            len(affected_genes['high_variance']), 
            len(affected_genes['high_sparsity']), 
            len(affected_genes['consistent'])
        ]
    }
    
    diagnosis_df = pd.DataFrame(diagnosis_summary)
    diagnosis_df.to_csv(os.path.join(output_dir, 'domain_shift_diagnosis_summary.csv'), index=False)
    print(f"도메인 시프트 진단 요약 저장: domain_shift_diagnosis_summary.csv")
    
    # 5. 패치별 발현 통계 저장
    patch_stats = {
        'patch_id': expr_data.index,
        'mean_expression': expr_data.mean(axis=1),
        'std_expression': expr_data.std(axis=1),
        'max_expression': expr_data.max(axis=1),
        'min_expression': expr_data.min(axis=1),
        'sparsity': (expr_data == 0).mean(axis=1),
        'non_zero_count': (expr_data > 0).sum(axis=1)
    }
    
    patch_stats_df = pd.DataFrame(patch_stats)
    patch_stats_df.to_csv(os.path.join(output_dir, 'patch_expression_statistics.csv'), index=False)
    print(f"패치별 발현 통계 저장: patch_expression_statistics.csv")

def generate_domain_shift_report(gene_stats_df, affected_genes, summary, output_dir):
    """도메인 시프트 분석 보고서 생성"""
    print("\n=== 도메인 시프트 분석 보고서 생성 ===")
    
    report_path = os.path.join(output_dir, 'domain_shift_gene_expression_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 도메인 시프트 Gene Expression 분석 보고서\n\n")
        
        f.write("## 1. 도메인 시프트 진단 결과\n\n")
        f.write(f"- **OOD 플래그**: {summary['ood_flag']}\n")
        f.write(f"- **중간값 최대 코사인 유사도**: {summary['median_max_cosine_similarity']:.4f}\n")
        f.write(f"- **중간값 Top-K 엔트로피**: {summary['median_top_k_entropy']:.4f}\n")
        f.write(f"- **중간값 Effective K**: {summary['median_effective_k']:.2f}\n")
        
        if 'ood_reasons' in summary:
            f.write(f"- **OOD 이유**: {', '.join(summary['ood_reasons'])}\n")
        
        f.write(f"\n- **총 패치 수**: {summary['total_patches']:,}\n")
        f.write(f"- **총 유전자 수**: {summary['total_genes']:,}\n")
        
        f.write("\n## 2. Gene Expression 통계\n\n")
        f.write(f"- **전체 스파시티**: {(gene_stats_df['sparsity'].mean() * 100):.2f}%\n")
        f.write(f"- **평균 발현값**: {gene_stats_df['mean_expression'].mean():.4f}\n")
        f.write(f"- **평균 변동계수**: {gene_stats_df['cv'].mean():.2f}\n")
        
        f.write("\n## 3. 도메인 시프트 영향 유전자\n\n")
        f.write(f"- **높은 변동성 유전자**: {len(affected_genes['high_variance'])}개\n")
        f.write(f"- **높은 스파시티 유전자**: {len(affected_genes['high_sparsity'])}개\n")
        f.write(f"- **일관성 있는 발현 유전자**: {len(affected_genes['consistent'])}개\n")
        
        f.write("\n## 4. 해석 및 권장사항\n\n")
        f.write("### 도메인 시프트 해석:\n")
        if summary['ood_flag']:
            f.write("- **도메인 시프트가 감지됨**: 모델이 새로운 도메인에서 제대로 작동하지 않을 가능성이 높습니다.\n")
            f.write("- **낮은 코사인 유사도**: 쿼리와 훈련 데이터 간의 유사성이 낮습니다.\n")
            f.write("- **높은 엔트로피**: 검색 가중치가 거의 균등하게 분포되어 있습니다.\n")
        else:
            f.write("- **도메인 시프트 없음**: 모델이 현재 도메인에서 잘 작동하고 있습니다.\n")
        
        f.write("\n### 권장사항:\n")
        f.write("1. **도메인 적응**: 새로운 도메인의 데이터로 모델을 재훈련하거나 파인튜닝을 고려하세요.\n")
        f.write("2. **데이터 품질 검토**: 입력 데이터의 품질과 전처리 과정을 검토하세요.\n")
        f.write("3. **특징 공간 분석**: 도메인 간 특징 공간의 차이를 더 자세히 분석하세요.\n")
        f.write("4. **앙상블 방법**: 여러 모델을 결합하여 도메인 시프트에 대한 견고성을 높이세요.\n")
        
        f.write("\n## 5. 생성된 파일들\n\n")
        f.write("- `domain_shift_gene_analysis.png`: 도메인 시프트 분석 시각화\n")
        f.write("- `domain_shift_expression_heatmaps.png`: 발현 패턴 히트맵\n")
        f.write("- `domain_shift_pca_visualization.png`: PCA 공간적 패턴 시각화\n")
        f.write("- `domain_shift_gene_expression_report.md`: 이 보고서\n")
    
    print(f"도메인 시프트 분석 보고서 저장: {report_path}")

def main():
    """메인 함수"""
    print("=== 도메인 시프트 Gene Expression Map 생성기 ===\n")
    
    # 파일 경로 설정
    ood_dir = "/home/students/hbae/Loki/ood_out"
    expr_path = "/home/students/hbae/Loki/gene_expression_maps_TCGA-QK-A6IH-01Z-00-DX1.csv"
    output_dir = "/home/students/hbae/Loki/domain_shift_gene_maps"
    
    # 1. 도메인 시프트 진단 결과 로드
    summary, weights_df = load_domain_shift_results(ood_dir)
    
    # 2. Gene expression 데이터 로드
    expr_data = load_gene_expression_data(expr_path)
    
    # 3. 도메인 시프트 패턴 분석
    gene_stats_df = analyze_domain_shift_patterns(expr_data, summary)
    
    # 4. 도메인 시프트 영향 유전자 식별
    affected_genes = identify_domain_shift_affected_genes(gene_stats_df, expr_data)
    
    # 5. 시각화 생성
    create_domain_shift_visualizations(expr_data, gene_stats_df, affected_genes, summary, output_dir)
    create_expression_heatmaps(expr_data, affected_genes, output_dir)
    create_pca_visualization(expr_data, affected_genes, output_dir)
    
    # 6. CSV 파일 생성
    save_gene_expression_maps_to_csv(expr_data, gene_stats_df, affected_genes, summary, output_dir)
    
    # 7. 보고서 생성
    generate_domain_shift_report(gene_stats_df, affected_genes, summary, output_dir)
    
    print(f"\n=== 분석 완료 ===")
    print(f"결과가 {output_dir} 디렉토리에 저장되었습니다.")
    print(f"생성된 파일들:")
    print(f"📊 시각화 파일들:")
    print(f"- domain_shift_gene_analysis.png")
    print(f"- domain_shift_expression_heatmaps.png") 
    print(f"- domain_shift_pca_visualization.png")
    print(f"📋 CSV 데이터 파일들:")
    print(f"- gene_expression_map_full.csv (전체 gene expression 데이터)")
    print(f"- gene_statistics_summary.csv (유전자별 통계)")
    print(f"- consistent_genes_expression.csv (일관성 있는 발현 유전자)")
    print(f"- high_sparsity_genes_expression.csv (높은 스파시티 유전자)")
    print(f"- domain_shift_diagnosis_summary.csv (도메인 시프트 진단 요약)")
    print(f"- patch_expression_statistics.csv (패치별 발현 통계)")
    print(f"📄 보고서:")
    print(f"- domain_shift_gene_expression_report.md")

if __name__ == "__main__":
    main()
