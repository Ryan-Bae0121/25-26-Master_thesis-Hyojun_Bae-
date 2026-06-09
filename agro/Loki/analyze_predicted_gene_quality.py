#!/usr/bin/env python3
"""
예측된 gene expression 데이터의 품질 분석 스크립트

이 스크립트는 예측된 gene expression 데이터에서:
1. 각 gene에 대해 MSE와 sMAPE 계산
2. 통계적 유의성을 검정하여 significantly well predicted gene 식별
3. 결과 시각화 및 리포트 생성

사용법:
    python analyze_predicted_gene_quality.py --input_csv gene_expression_maps_TCGA-QK-A6IH-01Z-00-DX1.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path):
    """
    CSV 파일을 로드하고 데이터를 정리합니다.
    
    Args:
        csv_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: 정리된 gene expression 데이터 (genes x patches)
    """
    print(f"데이터 로딩 중: {csv_path}")
    
    # CSV 파일 로드
    data = pd.read_csv(csv_path, index_col=0)
    
    print(f"원본 데이터 크기: {data.shape}")
    print(f"행(patches): {data.shape[0]}, 열(genes): {data.shape[1]}")
    
    # 데이터가 patches x genes 형태인지 확인하고 필요시 전치
    # 일반적으로 patches가 더 적으므로 patches가 행이면 전치
    if data.shape[0] < data.shape[1]:
        print("데이터를 전치합니다 (genes x patches 형태로 변환)")
        data = data.T
    
    print(f"최종 데이터 크기: {data.shape}")
    print(f"Genes: {data.shape[0]}, Patches: {data.shape[1]}")
    
    # 결측값 처리
    original_shape = data.shape
    data = data.dropna(axis=1)  # 결측값이 있는 patch 제거
    if data.shape != original_shape:
        print(f"결측값 제거 후: {data.shape}")
    
    return data

def calculate_gene_metrics(data):
    """
    각 gene에 대해 MSE와 sMAPE를 계산합니다.
    스파스 데이터(많은 0값)를 고려한 분석
    
    Args:
        data (pd.DataFrame): Gene expression 데이터 (genes x patches)
        
    Returns:
        pd.DataFrame: 각 gene의 메트릭 정보
    """
    print("각 gene의 메트릭 계산 중...")
    
    metrics = []
    
    for gene in data.index:
        gene_values = data.loc[gene].values
        
        # 0이 아닌 값들만 고려
        non_zero_values = gene_values[gene_values > 0]
        n_non_zero = len(non_zero_values)
        n_total = len(gene_values)
        
        if n_non_zero == 0:
            # 모든 값이 0인 경우
            metrics.append({
                'gene': gene,
                'n_patches': n_total,
                'n_non_zero_patches': 0,
                'sparsity': 1.0,
                'mean_expression': 0.0,
                'std_expression': 0.0,
                'mse': 0.0,
                'smape': 0.0,
                'cv': 0.0,
                'max_expression': 0.0,
                'expression_range': 0.0
            })
            continue
        
        # 0이 아닌 값들에 대한 통계
        mean_value = np.mean(non_zero_values)
        std_value = np.std(non_zero_values)
        max_value = np.max(non_zero_values)
        min_value = np.min(non_zero_values)
        
        # MSE 계산 (0이 아닌 값들 내에서의 변동성)
        mse = np.mean((non_zero_values - mean_value) ** 2)
        
        # sMAPE 계산 (0이 아닌 값들 내에서의 변동성)
        eps = 1e-8
        smape = np.mean(2 * np.abs(non_zero_values - mean_value) / 
                       (np.abs(non_zero_values) + np.abs(mean_value) + eps)) * 100
        
        # 변이계수
        cv = std_value / (mean_value + eps)
        
        # 스파시티 (0의 비율)
        sparsity = (n_total - n_non_zero) / n_total
        
        # 표현 범위
        expression_range = max_value - min_value
        
        metrics.append({
            'gene': gene,
            'n_patches': n_total,
            'n_non_zero_patches': n_non_zero,
            'sparsity': sparsity,
            'mean_expression': mean_value,
            'std_expression': std_value,
            'mse': mse,
            'smape': smape,
            'cv': cv,
            'max_expression': max_value,
            'expression_range': expression_range
        })
    
    metrics_df = pd.DataFrame(metrics)
    print(f"메트릭 계산 완료: {len(metrics_df)} genes")
    
    return metrics_df

def calculate_expression_quality_metrics(data):
    """
    스파스 데이터에 적합한 표현 품질 메트릭을 계산합니다.
    하나의 슬라이드만 있는 상황에서 내부 일관성을 평가합니다.
    
    Args:
        data (pd.DataFrame): Gene expression 데이터
        
    Returns:
        pd.DataFrame: 품질 메트릭이 추가된 데이터프레임
    """
    print("표현 품질 메트릭 계산 중...")
    
    metrics = []
    
    for gene in data.index:
        gene_values = data.loc[gene].values
        
        # 0이 아닌 값들만 고려
        non_zero_values = gene_values[gene_values > 0]
        n_non_zero = len(non_zero_values)
        n_total = len(gene_values)
        
        if n_non_zero < 3:  # 최소 3개 이상의 non-zero 값이 필요
            metrics.append({
                'gene': gene,
                'n_patches': n_total,
                'n_non_zero_patches': n_non_zero,
                'sparsity': (n_total - n_non_zero) / n_total,
                'mean_expression': 0.0,
                'std_expression': 0.0,
                'mse': float('inf'),
                'smape': float('inf'),
                'cv': float('inf'),
                'max_expression': 0.0,
                'expression_range': 0.0,
                'quality_score': 0.0,
                'is_high_quality': False
            })
            continue
        
        # 0이 아닌 값들에 대한 통계
        mean_value = np.mean(non_zero_values)
        std_value = np.std(non_zero_values)
        max_value = np.max(non_zero_values)
        min_value = np.min(non_zero_values)
        
        # MSE 계산 (0이 아닌 값들 내에서의 변동성)
        mse = np.mean((non_zero_values - mean_value) ** 2)
        
        # sMAPE 계산 (0이 아닌 값들 내에서의 변동성)
        eps = 1e-8
        smape = np.mean(2 * np.abs(non_zero_values - mean_value) / 
                       (np.abs(non_zero_values) + np.abs(mean_value) + eps)) * 100
        
        # 변이계수
        cv = std_value / (mean_value + eps)
        
        # 스파시티 (0의 비율)
        sparsity = (n_total - n_non_zero) / n_total
        
        # 표현 범위
        expression_range = max_value - min_value
        
        # 품질 점수 계산 (낮은 MSE, 낮은 sMAPE, 적절한 표현 범위)
        # 정규화된 점수 (0-1 범위)
        mse_score = 1.0 / (1.0 + mse)  # MSE가 낮을수록 높은 점수
        smape_score = max(0, 1.0 - smape / 100)  # sMAPE가 낮을수록 높은 점수
        range_score = min(1.0, expression_range / mean_value) if mean_value > 0 else 0  # 적절한 범위
        coverage_score = n_non_zero / n_total  # 커버리지
        
        # 가중 평균으로 최종 품질 점수 계산
        quality_score = (mse_score * 0.3 + smape_score * 0.3 + 
                        range_score * 0.2 + coverage_score * 0.2)
        
        # 고품질 기준: 품질 점수 > 0.7, 최소 5개 이상의 non-zero patches
        is_high_quality = quality_score > 0.7 and n_non_zero >= 5
        
        metrics.append({
            'gene': gene,
            'n_patches': n_total,
            'n_non_zero_patches': n_non_zero,
            'sparsity': sparsity,
            'mean_expression': mean_value,
            'std_expression': std_value,
            'mse': mse,
            'smape': smape,
            'cv': cv,
            'max_expression': max_value,
            'expression_range': expression_range,
            'quality_score': quality_score,
            'is_high_quality': is_high_quality
        })
    
    metrics_df = pd.DataFrame(metrics)
    print(f"품질 메트릭 계산 완료: {len(metrics_df)} genes")
    
    return metrics_df

def identify_high_quality_genes(metrics_df, quality_threshold=0.7, min_coverage=0.1):
    """
    고품질 genes를 식별합니다.
    
    Args:
        metrics_df (pd.DataFrame): 메트릭 데이터프레임
        quality_threshold (float): 품질 점수 임계값
        min_coverage (float): 최소 커버리지 비율
        
    Returns:
        pd.DataFrame: 고품질 genes
    """
    print(f"고품질 genes 식별 중...")
    print(f"기준: quality_score > {quality_threshold} AND coverage > {min_coverage}")
    
    high_quality = metrics_df[
        (metrics_df['quality_score'] > quality_threshold) & 
        (metrics_df['n_non_zero_patches'] / metrics_df['n_patches'] > min_coverage) &
        (metrics_df['n_non_zero_patches'] >= 5)  # 최소 5개 이상의 non-zero patches
    ].copy()
    
    print(f"고품질 genes: {len(high_quality)} genes")
    
    return high_quality

def create_visualizations(metrics_df, high_quality_df, output_dir):
    """
    분석 결과를 시각화합니다.
    
    Args:
        metrics_df (pd.DataFrame): 전체 메트릭 데이터
        high_quality_df (pd.DataFrame): 고품질 genes
        output_dir (str): 출력 디렉토리
    """
    print("시각화 생성 중...")
    
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 무한대 값 제거
    metrics_df_clean = metrics_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 1. 품질 점수 분포
    plt.figure(figsize=fig_size)
    plt.hist(metrics_df_clean['quality_score'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(metrics_df_clean['quality_score'].median(), color='red', linestyle='--', 
                label=f'Median: {metrics_df_clean["quality_score"].median():.3f}')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Quality Scores across Genes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hist_quality_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 스파시티 분포
    plt.figure(figsize=fig_size)
    plt.hist(metrics_df['sparsity'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(metrics_df['sparsity'].median(), color='red', linestyle='--',
                label=f'Median: {metrics_df["sparsity"].median():.2f}')
    plt.xlabel('Sparsity (Proportion of Zeros)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sparsity across Genes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hist_sparsity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. MSE 분포 (유한한 값만)
    plt.figure(figsize=fig_size)
    finite_mse = metrics_df_clean['mse']
    plt.hist(finite_mse, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(finite_mse.median(), color='red', linestyle='--', 
                label=f'Median: {finite_mse.median():.6f}')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('Distribution of MSE across Genes (Finite Values)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hist_mse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. sMAPE 분포 (유한한 값만)
    plt.figure(figsize=fig_size)
    finite_smape = metrics_df_clean['smape']
    plt.hist(finite_smape, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(finite_smape.median(), color='red', linestyle='--',
                label=f'Median: {finite_smape.median():.2f}%')
    plt.xlabel('sMAPE (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of sMAPE across Genes (Finite Values)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hist_smape.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 품질 점수 vs 스파시티
    plt.figure(figsize=fig_size)
    colors = ['red' if gene in high_quality_df['gene'].values else 'blue' 
              for gene in metrics_df['gene']]
    
    plt.scatter(metrics_df['sparsity'], metrics_df['quality_score'], 
                c=colors, alpha=0.6, s=20)
    
    # 고품질 genes 라벨링 (상위 10개만)
    if len(high_quality_df) > 0:
        top_genes = high_quality_df.nlargest(10, 'quality_score')
        for _, row in top_genes.iterrows():
            plt.annotate(row['gene'], 
                        (row['sparsity'], row['quality_score']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    plt.xlabel('Sparsity')
    plt.ylabel('Quality Score')
    plt.title('Quality Score vs Sparsity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/quality_vs_sparsity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 표현 레벨 vs 품질
    plt.figure(figsize=fig_size)
    colors = ['red' if gene in high_quality_df['gene'].values else 'blue' 
              for gene in metrics_df['gene']]
    
    plt.scatter(metrics_df['mean_expression'], metrics_df['quality_score'], 
                c=colors, alpha=0.6, s=20)
    
    plt.xlabel('Mean Expression Level')
    plt.ylabel('Quality Score')
    plt.title('Expression Level vs Quality Score')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/expression_vs_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("시각화 완료")

def generate_summary_report(metrics_df, well_predicted_df, output_dir):
    """
    분석 결과 요약 리포트를 생성합니다.
    
    Args:
        metrics_df (pd.DataFrame): 전체 메트릭 데이터
        well_predicted_df (pd.DataFrame): Well-predicted genes
        output_dir (str): 출력 디렉토리
    """
    print("요약 리포트 생성 중...")
    
    report = []
    report.append("=" * 60)
    report.append("예측된 Gene Expression 품질 분석 리포트")
    report.append("=" * 60)
    report.append("")
    
    # 전체 통계
    report.append("1. 전체 통계")
    report.append("-" * 30)
    report.append(f"총 분석된 genes: {len(metrics_df)}")
    report.append(f"평균 patches per gene: {metrics_df['n_patches'].mean():.1f}")
    report.append(f"평균 expression level: {metrics_df['mean_expression'].mean():.4f}")
    report.append(f"평균 MSE: {metrics_df['mse'].mean():.6f}")
    report.append(f"평균 sMAPE: {metrics_df['smape'].mean():.2f}%")
    report.append("")
    
    # Well-predicted genes 통계
    report.append("2. Well-predicted Genes 통계")
    report.append("-" * 30)
    report.append(f"Well-predicted genes 수: {len(well_predicted_df)}")
    report.append(f"전체 대비 비율: {len(well_predicted_df)/len(metrics_df)*100:.2f}%")
    
    if len(well_predicted_df) > 0:
        report.append(f"Well-predicted genes 평균 MSE: {well_predicted_df['mse'].mean():.6f}")
        report.append(f"Well-predicted genes 평균 sMAPE: {well_predicted_df['smape'].mean():.2f}%")
        report.append(f"Well-predicted genes 평균 expression: {well_predicted_df['mean_expression'].mean():.4f}")
        report.append("")
        
        # 상위 10개 genes
        report.append("3. 상위 10개 Well-predicted Genes")
        report.append("-" * 30)
        top_genes = well_predicted_df.nsmallest(10, 'q_value')
        for i, (_, row) in enumerate(top_genes.iterrows(), 1):
            report.append(f"{i:2d}. {row['gene']:15s} | "
                         f"MSE: {row['mse']:.6f} | "
                         f"sMAPE: {row['smape']:6.2f}% | "
                         f"q-value: {row['q_value']:.4f}")
    else:
        report.append("Well-predicted genes가 없습니다.")
    
    report.append("")
    report.append("4. 분석 기준")
    report.append("-" * 30)
    report.append("Well-predicted gene 기준:")
    report.append("- q-value < 0.05 (FDR 보정된 p-value)")
    report.append("- sMAPE < 30%")
    report.append("- n_patches >= 10")
    report.append("")
    
    # 파일 저장
    report_text = "\n".join(report)
    with open(f'{output_dir}/analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n리포트가 {output_dir}/analysis_summary.txt에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='예측된 gene expression 데이터 품질 분석')
    parser.add_argument('--input_csv', required=True, 
                       help='입력 CSV 파일 경로')
    parser.add_argument('--output_dir', default='predicted_gene_analysis',
                       help='출력 디렉토리 (기본값: predicted_gene_analysis)')
    parser.add_argument('--n_perm', type=int, default=1000,
                       help='Permutation test 횟수 (기본값: 1000)')
    parser.add_argument('--q_threshold', type=float, default=0.05,
                       help='q-value 임계값 (기본값: 0.05)')
    parser.add_argument('--smape_threshold', type=float, default=30.0,
                       help='sMAPE 임계값 (기본값: 30.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("예측된 Gene Expression 품질 분석 시작")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드 및 준비
        data = load_and_prepare_data(args.input_csv)
        
        # 2. 메트릭 계산
        metrics_df = calculate_gene_metrics(data)
        
        # 3. Permutation test
        metrics_df = permutation_test_for_consistency(data, args.n_perm, args.seed)
        
        # 4. Well-predicted genes 식별
        well_predicted_df = identify_well_predicted_genes(
            metrics_df, args.q_threshold, args.smape_threshold
        )
        
        # 5. 결과 저장
        metrics_df.to_csv(f'{args.output_dir}/metrics_per_gene.csv', index=False)
        well_predicted_df.to_csv(f'{args.output_dir}/well_predicted_genes.csv', index=False)
        
        # 6. 시각화
        create_visualizations(metrics_df, well_predicted_df, args.output_dir)
        
        # 7. 리포트 생성
        generate_summary_report(metrics_df, well_predicted_df, args.output_dir)
        
        print("\n" + "=" * 60)
        print("분석 완료!")
        print(f"결과가 {args.output_dir} 디렉토리에 저장되었습니다.")
        print("=" * 60)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()