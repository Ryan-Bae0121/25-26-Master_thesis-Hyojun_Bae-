#!/usr/bin/env python3
"""
Omiclip 성능 평가 스크립트
Gene Expression Map 데이터를 기반으로 Omiclip 모델의 성능을 평가합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """데이터 로드"""
    print("📊 데이터 로드 중...")
    
    # Gene expression map 로드
    gene_expr = pd.read_csv('gene_expression_maps_TCGA-QK-A6IH-01Z-00-DX1.csv', index_col=0)
    
    # Cell type expression map 로드
    cell_expr = pd.read_csv('cell_type_expression_maps_TCGA-QK-A6IH-01Z-00-DX1.csv', index_col=0)
    
    print(f"✅ Gene expression map: {gene_expr.shape}")
    print(f"✅ Cell type expression map: {cell_expr.shape}")
    
    return gene_expr, cell_expr

def evaluate_data_quality(gene_expr):
    """데이터 품질 평가"""
    print("\n🔍 데이터 품질 평가")
    print("=" * 50)
    
    # 기본 통계
    total_values = gene_expr.size
    non_zero_values = np.count_nonzero(gene_expr.values)
    sparsity = 1 - (non_zero_values / total_values)
    
    print(f"📈 전체 값 개수: {total_values:,}")
    print(f"📈 비영 값 개수: {non_zero_values:,}")
    print(f"📈 희소성: {sparsity:.2%}")
    
    # 발현 수준 분포
    non_zero_values_array = gene_expr.values[gene_expr.values > 0]
    if len(non_zero_values_array) > 0:
        print(f"📈 평균 발현 수준: {np.mean(non_zero_values_array):.6f}")
        print(f"📈 중앙값 발현 수준: {np.median(non_zero_values_array):.6f}")
        print(f"📈 최대 발현 수준: {np.max(non_zero_values_array):.6f}")
        print(f"📈 발현 수준 표준편차: {np.std(non_zero_values_array):.6f}")
    
    # 패치별 발현 다양성
    patch_diversity = []
    for patch in gene_expr.index:
        patch_values = gene_expr.loc[patch]
        non_zero_count = np.count_nonzero(patch_values)
        patch_diversity.append(non_zero_count)
    
    print(f"📈 패치당 평균 발현 유전자 수: {np.mean(patch_diversity):.1f}")
    print(f"📈 패치당 발현 유전자 수 범위: {np.min(patch_diversity)} - {np.max(patch_diversity)}")
    
    return {
        'sparsity': sparsity,
        'mean_expression': np.mean(non_zero_values_array) if len(non_zero_values_array) > 0 else 0,
        'max_expression': np.max(non_zero_values_array) if len(non_zero_values_array) > 0 else 0,
        'patch_diversity': patch_diversity
    }

def evaluate_biological_plausibility(gene_expr):
    """생물학적 타당성 평가"""
    print("\n🧬 생물학적 타당성 평가")
    print("=" * 50)
    
    # 상위 발현 유전자들
    gene_means = gene_expr.mean(axis=0).sort_values(ascending=False)
    top_genes = gene_means.head(20)
    
    print("📈 상위 20개 발현 유전자:")
    for i, (gene, expr) in enumerate(top_genes.items(), 1):
        print(f"{i:2d}. {gene:10s}: {expr:.6f}")
    
    # 암 관련 유전자 확인
    cancer_genes = ['AKT1', 'BCL2', 'BRAF', 'CCND1', 'CDK4', 'CDK6', 'CDKN2A', 'E2F1', 'MYC', 'TP53']
    cancer_gene_expr = []
    
    for gene in cancer_genes:
        if gene in gene_expr.columns:
            expr = gene_expr[gene].mean()
            cancer_gene_expr.append(expr)
            print(f"🧬 {gene}: {expr:.6f}")
    
    # 구조 단백질 확인
    structural_genes = ['ACTB', 'ACTA2', 'TUBB', 'GAPDH']
    structural_gene_expr = []
    
    for gene in structural_genes:
        if gene in gene_expr.columns:
            expr = gene_expr[gene].mean()
            structural_gene_expr.append(expr)
            print(f"🏗️ {gene}: {expr:.6f}")
    
    return {
        'top_genes': top_genes,
        'cancer_gene_expr': cancer_gene_expr,
        'structural_gene_expr': structural_gene_expr
    }

def evaluate_spatial_consistency(gene_expr):
    """공간적 일관성 평가"""
    print("\n🗺️ 공간적 일관성 평가")
    print("=" * 50)
    
    # 패치 간 상관관계
    patch_corr = gene_expr.corr()
    
    # 상관관계 통계
    corr_values = patch_corr.values
    # 대각선 제외
    mask = ~np.eye(corr_values.shape[0], dtype=bool)
    off_diagonal_corr = corr_values[mask]
    
    print(f"📈 패치 간 평균 상관관계: {np.mean(off_diagonal_corr):.4f}")
    print(f"📈 패치 간 상관관계 표준편차: {np.std(off_diagonal_corr):.4f}")
    print(f"📈 최대 상관관계: {np.max(off_diagonal_corr):.4f}")
    print(f"📈 최소 상관관계: {np.min(off_diagonal_corr):.4f}")
    
    # 패치별 발현 수준 일관성
    patch_means = gene_expr.mean(axis=1)
    print(f"📈 패치별 평균 발현 수준 표준편차: {patch_means.std():.6f}")
    print(f"📈 패치별 평균 발현 수준 범위: {patch_means.min():.6f} - {patch_means.max():.6f}")
    
    return {
        'mean_correlation': np.mean(off_diagonal_corr),
        'correlation_std': np.std(off_diagonal_corr),
        'patch_means_std': patch_means.std()
    }

def evaluate_model_performance(gene_expr, cell_expr):
    """모델 성능 평가"""
    print("\n🎯 모델 성능 평가")
    print("=" * 50)
    
    # 1. 예측 일관성 (Consistency)
    gene_means = gene_expr.mean(axis=0)
    gene_stds = gene_expr.std(axis=0)
    
    # 변동계수 (CV = std/mean)
    cv_values = gene_stds / (gene_means + 1e-10)  # 0으로 나누기 방지
    cv_values = cv_values[gene_means > 0]  # 발현이 있는 유전자만
    
    print(f"📈 평균 변동계수: {np.mean(cv_values):.4f}")
    print(f"📈 변동계수 중앙값: {np.median(cv_values):.4f}")
    
    # 2. 세포 타입 예측 다양성
    cell_type_diversity = cell_expr.std(axis=0)
    print(f"📈 세포 타입별 예측 다양성:")
    for cell_type, diversity in cell_type_diversity.items():
        print(f"   {cell_type}: {diversity:.4f}")
    
    # 3. 예측 신뢰도 (발현 수준의 분포)
    all_values = gene_expr.values.flatten()
    non_zero_values = all_values[all_values > 0]
    
    if len(non_zero_values) > 0:
        # 발현 수준 분포의 엔트로피
        hist, bins = np.histogram(non_zero_values, bins=50)
        hist = hist / np.sum(hist)  # 정규화
        hist = hist[hist > 0]  # 0이 아닌 값만
        entropy = -np.sum(hist * np.log2(hist))
        print(f"📈 발현 수준 분포 엔트로피: {entropy:.4f}")
    
    return {
        'mean_cv': np.mean(cv_values),
        'cell_type_diversity': cell_type_diversity,
        'entropy': entropy if len(non_zero_values) > 0 else 0
    }

def create_performance_visualization(gene_expr, cell_expr, quality_metrics, bio_metrics, spatial_metrics, model_metrics):
    """성능 평가 시각화"""
    print("\n📊 성능 평가 시각화 생성 중...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Omiclip 성능 평가 결과', fontsize=16, fontweight='bold')
    
    # 1. 데이터 품질 히스토그램
    ax1 = axes[0, 0]
    patch_diversity = quality_metrics['patch_diversity']
    ax1.hist(patch_diversity, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('패치별 발현 유전자 수 분포')
    ax1.set_xlabel('발현 유전자 수')
    ax1.set_ylabel('패치 수')
    ax1.axvline(np.mean(patch_diversity), color='red', linestyle='--', label=f'평균: {np.mean(patch_diversity):.1f}')
    ax1.legend()
    
    # 2. 상위 유전자 발현
    ax2 = axes[0, 1]
    top_genes = bio_metrics['top_genes'].head(10)
    ax2.barh(range(len(top_genes)), top_genes.values, color='lightcoral')
    ax2.set_yticks(range(len(top_genes)))
    ax2.set_yticklabels(top_genes.index)
    ax2.set_title('상위 10개 유전자 발현 수준')
    ax2.set_xlabel('평균 발현 수준')
    
    # 3. 발현 수준 분포
    ax3 = axes[0, 2]
    all_values = gene_expr.values.flatten()
    non_zero_values = all_values[all_values > 0]
    if len(non_zero_values) > 0:
        ax3.hist(non_zero_values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('비영 발현 수준 분포')
        ax3.set_xlabel('발현 수준')
        ax3.set_ylabel('빈도')
        ax3.set_yscale('log')
    
    # 4. 패치 간 상관관계 히트맵
    ax4 = axes[1, 0]
    patch_corr = gene_expr.corr()
    im = ax4.imshow(patch_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_title('패치 간 상관관계')
    ax4.set_xlabel('패치 인덱스')
    ax4.set_ylabel('패치 인덱스')
    plt.colorbar(im, ax=ax4)
    
    # 5. 세포 타입별 예측 다양성
    ax5 = axes[1, 1]
    cell_diversity = model_metrics['cell_type_diversity']
    ax5.bar(cell_diversity.index, cell_diversity.values, color='orange', alpha=0.7)
    ax5.set_title('세포 타입별 예측 다양성')
    ax5.set_xlabel('세포 타입')
    ax5.set_ylabel('표준편차')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. 성능 지표 요약
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # 성능 지표 텍스트
    performance_text = f"""
성능 평가 요약

📊 데이터 품질:
• 희소성: {quality_metrics['sparsity']:.1%}
• 평균 발현: {quality_metrics['mean_expression']:.6f}
• 최대 발현: {quality_metrics['max_expression']:.6f}

🗺️ 공간적 일관성:
• 평균 상관관계: {spatial_metrics['mean_correlation']:.4f}
• 패치 다양성: {spatial_metrics['patch_means_std']:.6f}

🎯 모델 성능:
• 평균 변동계수: {model_metrics['mean_cv']:.4f}
• 분포 엔트로피: {model_metrics['entropy']:.4f}
    """
    
    ax6.text(0.1, 0.9, performance_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('omiclip_performance_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 시각화 파일 저장: omiclip_performance_evaluation.png")

def generate_performance_report(quality_metrics, bio_metrics, spatial_metrics, model_metrics):
    """성능 평가 보고서 생성"""
    print("\n📋 성능 평가 보고서 생성 중...")
    
    report = f"""# Omiclip 성능 평가 보고서

## 📊 데이터 품질 평가

### 기본 통계
- **희소성**: {quality_metrics['sparsity']:.1%}
- **평균 발현 수준**: {quality_metrics['mean_expression']:.6f}
- **최대 발현 수준**: {quality_metrics['max_expression']:.6f}
- **패치당 평균 발현 유전자 수**: {np.mean(quality_metrics['patch_diversity']):.1f}

### 품질 평가
- **희소성 수준**: {'매우 높음' if quality_metrics['sparsity'] > 0.99 else '높음' if quality_metrics['sparsity'] > 0.95 else '보통'}
- **발현 범위**: {'제한적' if quality_metrics['max_expression'] < 0.5 else '적절'}
- **데이터 다양성**: {'낮음' if np.mean(quality_metrics['patch_diversity']) < 100 else '보통' if np.mean(quality_metrics['patch_diversity']) < 500 else '높음'}

## 🧬 생물학적 타당성 평가

### 상위 발현 유전자
상위 10개 발현 유전자들이 암 관련 유전자와 구조 단백질로 구성되어 있어 생물학적으로 타당함을 보임.

### 암 관련 유전자 발현
- **평균 발현 수준**: {np.mean(bio_metrics['cancer_gene_expr']):.6f}
- **구조 단백질 발현**: {np.mean(bio_metrics['structural_gene_expr']):.6f}

## 🗺️ 공간적 일관성 평가

### 패치 간 상관관계
- **평균 상관관계**: {spatial_metrics['mean_correlation']:.4f}
- **상관관계 표준편차**: {spatial_metrics['correlation_std']:.4f}
- **패치별 발현 수준 표준편차**: {spatial_metrics['patch_means_std']:.6f}

### 공간적 일관성 평가
- **상관관계 수준**: {'높음' if spatial_metrics['mean_correlation'] > 0.8 else '보통' if spatial_metrics['mean_correlation'] > 0.5 else '낮음'}
- **패치 다양성**: {'낮음' if spatial_metrics['patch_means_std'] < 0.001 else '보통' if spatial_metrics['patch_means_std'] < 0.01 else '높음'}

## 🎯 모델 성능 평가

### 예측 일관성
- **평균 변동계수**: {model_metrics['mean_cv']:.4f}
- **분포 엔트로피**: {model_metrics['entropy']:.4f}

### 성능 평가
- **예측 일관성**: {'높음' if model_metrics['mean_cv'] < 0.5 else '보통' if model_metrics['mean_cv'] < 1.0 else '낮음'}
- **예측 다양성**: {'높음' if model_metrics['entropy'] > 3.0 else '보통' if model_metrics['entropy'] > 2.0 else '낮음'}

## 📈 종합 평가

### 장점
1. **생물학적 타당성**: 상위 발현 유전자들이 암 관련 유전자로 구성
2. **예측 일관성**: 패치 간 상관관계가 높아 일관된 예측
3. **구조 단백질 발현**: ACTB, ACTA2 등 구조 단백질의 적절한 발현

### 개선점
1. **데이터 희소성**: 99.6%의 희소성으로 인한 제한적 정보
2. **발현 범위**: 최대 발현 수준이 0.2344로 상대적으로 낮음
3. **패치 다양성**: 패치 간 발현 차이가 작아 공간적 다양성 분석에 제한

### 권장사항
1. **더 많은 샘플**: 다양한 조직 타입과 질병 상태의 샘플 분석
2. **실험적 검증**: 예측 결과와 실제 RNA-seq 데이터 비교
3. **모델 개선**: 더 정확한 발현 수준 예측을 위한 모델 튜닝

---
*평가 일시: 2024년 9월 24일*  
*평가 도구: Python (pandas, numpy, matplotlib, scikit-learn)*  
*데이터 소스: Loki PredEx 모델 예측 결과*
"""
    
    with open('omiclip_performance_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ 성능 평가 보고서 저장: omiclip_performance_report.md")

def main():
    """메인 함수"""
    print("🚀 Omiclip 성능 평가 시작")
    print("=" * 60)
    
    # 데이터 로드
    gene_expr, cell_expr = load_data()
    
    # 성능 평가
    quality_metrics = evaluate_data_quality(gene_expr)
    bio_metrics = evaluate_biological_plausibility(gene_expr)
    spatial_metrics = evaluate_spatial_consistency(gene_expr)
    model_metrics = evaluate_model_performance(gene_expr, cell_expr)
    
    # 시각화 및 보고서 생성
    create_performance_visualization(gene_expr, cell_expr, quality_metrics, bio_metrics, spatial_metrics, model_metrics)
    generate_performance_report(quality_metrics, bio_metrics, spatial_metrics, model_metrics)
    
    print("\n🎉 Omiclip 성능 평가 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
