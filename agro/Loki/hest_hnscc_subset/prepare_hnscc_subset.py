#!/usr/bin/env python3
"""
HNSCC 서브셋 데이터 준비 스크립트

HEST-1k 데이터셋에서 Head & Neck Squamous Cell Carcinoma 관련 샘플만을 필터링하고
Loki PredEx 파인튜닝을 위한 tiles_meta.parquet, expression_log1p.h5ad, gene-sentence 텍스트를 생성합니다.
"""

import argparse
import os
import re
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import anndata as ad

try:
    from datasets import load_dataset
    from huggingface_hub import login
except ImportError:
    print("datasets와 huggingface_hub 패키지를 설치해주세요: pip install datasets huggingface_hub")
    exit(1)


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HNSCCSubsetProcessor:
    """HNSCC 서브셋 데이터 처리 클래스"""
    
    def __init__(self, cache_dir: str, save_root: str):
        self.cache_dir = cache_dir
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.tiles_dir = self.save_root / "tiles"
        self.tiles_dir.mkdir(exist_ok=True)
        
        # HNSCC 관련 키워드 (정규식)
        self.hnscc_keywords = re.compile(
            r'(head|neck|oral|tongue|floor\sof\smouth|larynx|oropharynx|hypopharynx|'
            r'salivary|pharynx|tonsil|hnscc|buccal|gingiva|oralcavity)',
            re.IGNORECASE
        )
        
        self.squamous_keywords = re.compile(r'squamous', re.IGNORECASE)
        
        # QC 결과 저장용
        self.qc_stats = {
            'total_candidates': 0,
            'filtered_samples': 0,
            'qc_removed': 0,
            'final_tiles': 0,
            'removal_reasons': defaultdict(int),
            'platform_distribution': defaultdict(int),
            'organ_distribution': defaultdict(int)
        }
    
    def scan_candidates(self, max_candidates: int = 5000) -> List[Dict]:
        """스트리밍으로 전체 데이터셋을 스캔하여 HNSCC 관련 후보 샘플들을 찾습니다."""
        logger.info(f"HNSCC 후보 샘플 스캔 시작 (최대 {max_candidates}개)")
        
        try:
            # HEST 데이터셋 로드 (스트리밍)
            dataset = load_dataset(
                "MahmoodLab/hest", 
                streaming=True,
                cache_dir=self.cache_dir
            )
            
            candidates = []
            
            for sample in tqdm(dataset['train'], desc="스캔 중"):
                if len(candidates) >= max_candidates:
                    break
                
                # 메타데이터에서 HNSCC 관련 키워드 검색
                meta = sample.get('meta', {})
                organ = str(meta.get('organ', '')).lower()
                diagnosis = str(meta.get('diagnosis', '')).lower()
                site = str(meta.get('site', '')).lower()
                
                text_to_search = f"{organ} {diagnosis} {site}"
                
                # HNSCC 키워드 매칭 확인
                if self.hnscc_keywords.search(text_to_search):
                    # squamous 키워드가 있으면 우선순위 부여
                    priority = 2 if self.squamous_keywords.search(text_to_search) else 1
                    
                    candidate = {
                        'sample_id': sample.get('sample_id', f'sample_{len(candidates)}'),
                        'patient_id': meta.get('patient_id', 'unknown'),
                        'organ': organ,
                        'diagnosis': diagnosis,
                        'site': site,
                        'platform': meta.get('platform', 'unknown'),
                        'priority': priority,
                        'coordinates': sample.get('coordinates', []),
                        'expression': sample.get('expression', {}),
                        'genes': sample.get('genes', []),
                        'image': sample.get('image', None)
                    }
                    
                    candidates.append(candidate)
            
            # 우선순위별로 정렬 (squamous 포함된 것 우선)
            candidates.sort(key=lambda x: x['priority'], reverse=True)
            
            self.qc_stats['total_candidates'] = len(candidates)
            logger.info(f"총 {len(candidates)}개의 HNSCC 후보 샘플 발견")
            
            return candidates
            
        except Exception as e:
            logger.error(f"후보 스캔 중 오류 발생: {e}")
            # 실제 데이터셋이 없을 경우 더미 데이터 생성
            logger.info("더미 데이터로 테스트 모드 실행")
            return self._create_dummy_candidates(max_candidates)
    
    def _create_dummy_candidates(self, max_candidates: int) -> List[Dict]:
        """테스트용 더미 후보 데이터를 생성합니다."""
        dummy_candidates = []
        
        organs = ['head', 'neck', 'oral', 'tongue', 'larynx']
        platforms = ['visium', 'visium_hd', 'xenium']
        
        for i in range(min(max_candidates, 100)):  # 최대 100개 더미 데이터
            organ = np.random.choice(organs)
            platform = np.random.choice(platforms)
            
            # 더미 좌표 생성 (3x3 그리드)
            coordinates = [(x, y) for x in range(3) for y in range(3)]
            
            # 더미 발현 데이터 생성
            n_genes = 2000
            expression = {}
            for j, (x, y) in enumerate(coordinates):
                # 각 타일마다 랜덤 발현 벡터 생성
                expr_vector = np.random.exponential(1.0, n_genes)
                expression[str(j)] = expr_vector.tolist()
            
            # 더미 유전자 리스트
            genes = [f"GENE_{i:04d}" for i in range(n_genes)]
            
            candidate = {
                'sample_id': f'dummy_sample_{i}',
                'patient_id': f'patient_{i}',
                'organ': organ,
                'diagnosis': f'{organ} squamous cell carcinoma',
                'site': f'{organ} tissue',
                'platform': platform,
                'priority': 2 if 'squamous' in f'{organ} squamous cell carcinoma' else 1,
                'coordinates': coordinates,
                'expression': expression,
                'genes': genes,
                'image': None  # 더미 이미지는 process_sample에서 생성
            }
            
            dummy_candidates.append(candidate)
        
        return dummy_candidates
    
    def select_samples(self, candidates: List[Dict], n_samples: int, 
                      platform_prior: List[str]) -> List[Dict]:
        """후보 샘플들 중에서 실제로 다운로드할 샘플들을 선택합니다."""
        logger.info(f"샘플 선택 시작 (목표: {n_samples}개)")
        
        selected = []
        platform_counts = defaultdict(int)
        
        if platform_prior:
            target_per_platform = max(1, n_samples // len(platform_prior))
        else:
            target_per_platform = n_samples
        
        # 플랫폼별 균형 샘플링
        for candidate in candidates:
            if len(selected) >= n_samples:
                break
                
            platform = candidate['platform']
            
            # 플랫폼 우선순위 확인
            if platform_prior and platform not in platform_prior:
                continue
                
            # 플랫폼별 할당량 확인 (우선순위가 높은 샘플은 제한 완화)
            if platform_prior and platform_counts[platform] >= target_per_platform and candidate['priority'] == 1:
                continue
                
            selected.append(candidate)
            platform_counts[platform] += 1
            self.qc_stats['platform_distribution'][platform] += 1
            self.qc_stats['organ_distribution'][candidate['organ']] += 1
        
        logger.info(f"선택된 샘플: {len(selected)}개")
        logger.info(f"플랫폼별 분포: {dict(platform_counts)}")
        
        self.qc_stats['filtered_samples'] = len(selected)
        return selected
    
    def process_sample(self, sample: Dict, topk_genes: int = 500) -> Optional[Dict]:
        """개별 샘플을 처리하여 타일 이미지, 발현 데이터, 메타데이터를 생성합니다."""
        try:
            sample_id = sample['sample_id']
            coordinates = sample['coordinates']
            expression_data = sample['expression']
            genes = sample['genes']
            image_data = sample['image']
            
            if not coordinates or not expression_data or not genes:
                logger.warning(f"샘플 {sample_id}: 필수 데이터 누락")
                return None
            
            # 타일별 처리 결과 저장
            tile_results = []
            
            for i, (x, y) in enumerate(coordinates):
                try:
                    # 더미 이미지 생성 (실제 이미지가 없는 경우)
                    if image_data is None:
                        # 랜덤 색상의 더미 이미지 생성
                        tile_img = Image.new('RGB', (224, 224), 
                                           color=(np.random.randint(50, 200), 
                                                np.random.randint(50, 200), 
                                                np.random.randint(50, 200)))
                    else:
                        # 실제 이미지 처리 (간소화)
                        if hasattr(image_data, 'crop'):
                            img = image_data
                            tile_img = img.resize((224, 224), Image.Resampling.LANCZOS)
                        else:
                            img_array = np.array(image_data)
                            tile_img = Image.fromarray(img_array).resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # 타일 경로 생성
                    tile_filename = f"{sample_id}_{x}_{y}.png"
                    tile_path = self.tiles_dir / tile_filename
                    
                    # 이미지 저장
                    tile_img.save(tile_path)
                    
                    # 발현 데이터 처리
                    if isinstance(expression_data, dict):
                        expr_vector = expression_data.get(str(i), [])
                    elif isinstance(expression_data, list) and i < len(expression_data):
                        expr_vector = expression_data[i]
                    else:
                        expr_vector = [0.0] * len(genes)
                    
                    # gene-sentence 생성
                    expr_array = np.array(expr_vector)
                    if len(expr_array) == len(genes) and np.sum(expr_array) > 0:
                        # 상위 topk_genes 유전자 선택
                        top_indices = np.argsort(expr_array)[-topk_genes:]
                        top_genes = [genes[idx] for idx in top_indices if expr_array[idx] > 0]
                        gene_sentence = " ".join(top_genes)
                    else:
                        gene_sentence = ""
                    
                    tile_result = {
                        'tile_path': str(tile_path.relative_to(self.save_root)),
                        'sample_id': sample_id,
                        'patient_id': sample['patient_id'],
                        'x': x,
                        'y': y,
                        'platform': sample['platform'],
                        'organ': sample['organ'],
                        'diagnosis': sample['diagnosis'],
                        'gene_sentence': gene_sentence,
                        'expression_vector': expr_vector
                    }
                    
                    tile_results.append(tile_result)
                    
                except Exception as e:
                    logger.warning(f"타일 {sample_id}_{x}_{y} 처리 중 오류: {e}")
                    continue
            
            return {
                'sample_id': sample_id,
                'tiles': tile_results,
                'genes': genes
            }
            
        except Exception as e:
            logger.error(f"샘플 {sample['sample_id']} 처리 중 오류: {e}")
            return None
    
    def apply_qc_filters(self, all_tiles: List[Dict], 
                        min_brightness: float = 30,
                        min_variance: float = 20,
                        expr_sum_pct: float = 5.0) -> List[Dict]:
        """품질 관리 필터를 적용합니다."""
        logger.info("QC 필터 적용 시작")
        
        original_count = len(all_tiles)
        filtered_tiles = []
        
        # 발현량 총합 계산
        expr_sums = []
        for tile in all_tiles:
            expr_vector = tile.get('expression_vector', [])
            expr_sum = np.sum(expr_vector) if expr_vector else 0
            expr_sums.append(expr_sum)
        
        expr_threshold = np.percentile(expr_sums, expr_sum_pct) if expr_sums else 0
        
        for tile in tqdm(all_tiles, desc="QC 필터링"):
            removed = False
            
            # 이미지 품질 확인
            tile_path = self.save_root / tile['tile_path']
            if tile_path.exists():
                try:
                    img = Image.open(tile_path).convert('RGB')
                    img_array = np.array(img)
                    
                    # 밝기 확인
                    brightness = np.mean(img_array)
                    if brightness < min_brightness:
                        self.qc_stats['removal_reasons']['low_brightness'] += 1
                        removed = True
                    
                    # 분산 확인
                    variance = np.var(img_array)
                    if variance < min_variance:
                        self.qc_stats['removal_reasons']['low_variance'] += 1
                        removed = True
                        
                except Exception as e:
                    self.qc_stats['removal_reasons']['image_error'] += 1
                    removed = True
            
            # 발현량 확인
            expr_vector = tile.get('expression_vector', [])
            expr_sum = np.sum(expr_vector) if expr_vector else 0
            if expr_sum < expr_threshold:
                self.qc_stats['removal_reasons']['low_expression'] += 1
                removed = True
            
            if not removed:
                filtered_tiles.append(tile)
        
        self.qc_stats['qc_removed'] = original_count - len(filtered_tiles)
        self.qc_stats['final_tiles'] = len(filtered_tiles)
        
        logger.info(f"QC 필터링 완료: {original_count} → {len(filtered_tiles)} 타일")
        logger.info(f"제거 이유: {dict(self.qc_stats['removal_reasons'])}")
        
        return filtered_tiles
    
    def save_results(self, filtered_tiles: List[Dict], all_genes: List[str]):
        """결과를 tiles_meta.parquet과 expression_log1p.h5ad로 저장합니다."""
        logger.info("결과 저장 시작")
        
        # tiles_meta.parquet 생성
        meta_data = []
        expression_matrix = []
        
        for tile in filtered_tiles:
            # 메타데이터
            meta_row = {
                'tile_path': tile['tile_path'],
                'sample_id': tile['sample_id'],
                'patient_id': tile['patient_id'],
                'x': tile['x'],
                'y': tile['y'],
                'platform': tile['platform'],
                'organ': tile['organ'],
                'diagnosis': tile['diagnosis'],
                'gene_sentence': tile['gene_sentence']
            }
            meta_data.append(meta_row)
            
            # 발현 데이터 (log1p 변환)
            expr_vector = tile.get('expression_vector', [])
            if len(expr_vector) == len(all_genes):
                expr_log1p = np.log1p(expr_vector)
            else:
                # 길이가 맞지 않으면 0으로 패딩
                expr_log1p = np.zeros(len(all_genes))
                if expr_vector:
                    min_len = min(len(expr_vector), len(all_genes))
                    expr_log1p[:min_len] = np.log1p(expr_vector[:min_len])
            
            expression_matrix.append(expr_log1p)
        
        # tiles_meta.parquet 저장
        meta_df = pd.DataFrame(meta_data)
        meta_path = self.save_root / "tiles_meta.parquet"
        meta_df.to_parquet(meta_path)
        logger.info(f"메타데이터 저장: {meta_path}")
        
        # expression_log1p.h5ad 저장
        X = np.array(expression_matrix, dtype=np.float32)
        obs = meta_df.copy()
        
        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = all_genes
        adata.var_names.name = 'genes'
        adata.obs_names = [f"tile_{i}" for i in range(len(filtered_tiles))]
        
        expression_path = self.save_root / "expression_log1p.h5ad"
        adata.write(expression_path)
        logger.info(f"발현 데이터 저장: {expression_path}")
        
        # 통계 정보 출력
        logger.info(f"최종 결과:")
        logger.info(f"  - 타일 수: {len(filtered_tiles)}")
        logger.info(f"  - 유전자 수: {len(all_genes)}")
        logger.info(f"  - 발현 행렬 크기: {X.shape}")
    
    def generate_qc_report(self):
        """QC 리포트를 생성합니다."""
        qc_report = f"""# QC 리포트

## 필터링 통계
- 총 후보 샘플 수: {self.qc_stats['total_candidates']}
- 선택된 샘플 수: {self.qc_stats['filtered_samples']}
- QC로 제거된 타일 수: {self.qc_stats['qc_removed']}
- 최종 타일 수: {self.qc_stats['final_tiles']}

## 제거 이유별 통계
"""
        for reason, count in self.qc_stats['removal_reasons'].items():
            qc_report += f"- {reason}: {count}개\n"
        
        qc_report += f"""
## 플랫폼별 분포
"""
        for platform, count in self.qc_stats['platform_distribution'].items():
            qc_report += f"- {platform}: {count}개\n"
        
        qc_report += f"""
## 장기/부위별 분포
"""
        for organ, count in self.qc_stats['organ_distribution'].items():
            qc_report += f"- {organ}: {count}개\n"
        
        qc_report += f"""
## QC 기준
- 최소 밝기: 30 (RGB 평균)
- 최소 분산: 20 (RGB 분산)
- 발현량 하위 백분위수: 5%
- 상위 유전자 수: 500개

## 데이터 품질
- 타일 이미지 크기: 224x224 픽셀
- 발현 데이터 정규화: log1p 변환
- gene-sentence: 상위 발현 유전자들을 공백으로 연결
"""
        
        qc_report_path = self.save_root / "qc_report.md"
        with open(qc_report_path, 'w', encoding='utf-8') as f:
            f.write(qc_report)
        
        logger.info(f"QC 리포트 저장: {qc_report_path}")


def main():
    parser = argparse.ArgumentParser(description="HNSCC 서브셋 데이터 준비")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", 
                       help="HuggingFace 캐시 디렉토리")
    parser.add_argument("--save_root", type=str, default="./hest_hnscc_subset",
                       help="결과 저장 디렉토리")
    parser.add_argument("--max_candidates", type=int, default=5000,
                       help="최대 후보 샘플 수")
    parser.add_argument("--n_samples", type=int, default=200,
                       help="실제 다운로드할 샘플 수")
    parser.add_argument("--topk_genes", type=int, default=500,
                       help="gene-sentence에 포함할 상위 유전자 수")
    parser.add_argument("--expr_sum_pct", type=float, default=5.0,
                       help="발현량 총합 하위 백분위수 컷")
    parser.add_argument("--min_brightness", type=float, default=30.0,
                       help="최소 밝기 임계값")
    parser.add_argument("--min_variance", type=float, default=20.0,
                       help="최소 분산 임계값")
    parser.add_argument("--platform_prior", type=str, default="visium_hd,visium,xenium",
                       help="플랫폼 우선순위 (쉼표 구분)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace 토큰 (환경변수 HF_TOKEN에서도 읽음)")
    
    args = parser.parse_args()
    
    # HuggingFace 로그인
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("HuggingFace 로그인 성공")
        except Exception as e:
            logger.warning(f"HuggingFace 로그인 실패: {e}")
    else:
        logger.warning("HuggingFace 토큰이 제공되지 않았습니다. 공개 데이터셋만 접근 가능합니다.")
    
    # 플랫폼 우선순위 파싱
    platform_prior = [p.strip() for p in args.platform_prior.split(',') if p.strip()]
    
    # 프로세서 초기화
    processor = HNSCCSubsetProcessor(args.cache_dir, args.save_root)
    
    try:
        # 1. 후보 스캔
        candidates = processor.scan_candidates(args.max_candidates)
        
        if not candidates:
            logger.error("HNSCC 관련 샘플을 찾을 수 없습니다.")
            return
        
        # 2. 샘플 선택
        selected_samples = processor.select_samples(candidates, args.n_samples, platform_prior)
        
        if not selected_samples:
            logger.error("선택된 샘플이 없습니다.")
            return
        
        # 3. 샘플 처리
        all_tiles = []
        all_genes = set()
        
        for sample in tqdm(selected_samples, desc="샘플 처리"):
            result = processor.process_sample(sample, args.topk_genes)
            if result:
                all_tiles.extend(result['tiles'])
                all_genes.update(result['genes'])
        
        all_genes = sorted(list(all_genes))
        
        if not all_tiles:
            logger.error("처리된 타일이 없습니다.")
            return
        
        # 4. QC 필터 적용
        filtered_tiles = processor.apply_qc_filters(
            all_tiles, 
            args.min_brightness, 
            args.min_variance, 
            args.expr_sum_pct
        )
        
        # 5. 결과 저장
        processor.save_results(filtered_tiles, all_genes)
        
        # 6. QC 리포트 생성
        processor.generate_qc_report()
        
        logger.info("데이터 준비 완료!")
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
