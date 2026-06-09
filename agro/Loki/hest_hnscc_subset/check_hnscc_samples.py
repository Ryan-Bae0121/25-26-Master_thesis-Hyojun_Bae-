#!/usr/bin/env python3
"""
HEST-1k 데이터셋에서 HNSCC 관련 샘플 검색 스크립트

이 스크립트는 HEST-1k 데이터셋을 스트리밍으로 로드하여
Head and Neck Squamous Cell Carcinoma (HNSCC) 관련 샘플을 찾습니다.
"""

import os
import re
import logging
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from tqdm import tqdm

try:
    from datasets import load_dataset
    from huggingface_hub import login
except ImportError:
    print("datasets와 huggingface_hub 패키지를 설치해주세요:")
    print("pip install datasets huggingface_hub")
    exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def search_hnscc_samples(max_samples: int = 10000) -> List[Dict[str, Any]]:
    """
    HEST-1k 데이터셋에서 HNSCC 관련 샘플을 검색합니다.
    
    Args:
        max_samples: 검색할 최대 샘플 수
        
    Returns:
        HNSCC 관련 샘플들의 메타데이터 리스트
    """
    
    # HuggingFace 토큰 설정 (환경변수에서 읽기: export HF_TOKEN=...)
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        login(token=hf_token)
        logger.info("HuggingFace 로그인 성공")
    except Exception as e:
        logger.warning(f"HuggingFace 로그인 실패: {e}")
        logger.info("공개 데이터셋만 접근 가능합니다.")
    
    # HNSCC 관련 키워드 (정규식)
    hnscc_keywords = [
        "head", "neck", "oral", "tongue", "pharynx", "larynx", 
        "tonsil", "salivary", "hnscc", "oropharynx", "hypopharynx",
        "floor of mouth", "buccal", "gingiva"
    ]
    
    # 키워드들을 하나의 정규식으로 결합
    keyword_pattern = re.compile(
        "|".join(hnscc_keywords), 
        re.IGNORECASE
    )
    
    logger.info(f"HNSCC 관련 키워드: {hnscc_keywords}")
    logger.info(f"최대 검색 샘플 수: {max_samples}")
    
    matched_samples = []
    
    try:
        # HEST 데이터셋 로드 (스트리밍)
        logger.info("HEST 데이터셋 로드 중...")
        dataset = load_dataset(
            "MahmoodLab/hest", 
            streaming=True,
            token=hf_token
        )
        
        logger.info("데이터셋 로드 성공, 샘플 검색 시작...")
        
        # 샘플 검색
        for i, sample in enumerate(tqdm(dataset['train'], desc="샘플 검색", total=max_samples)):
            if i >= max_samples:
                break
            
            # 메타데이터 확인
            meta = sample.get('meta', {})
            
            # 검색할 필드들
            search_fields = []
            for field in ['organ', 'site', 'diagnosis', 'disease_state', 'tissue']:
                if field in meta and meta[field] is not None:
                    search_fields.append(str(meta[field]))
            
            # 모든 검색 필드를 하나의 문자열로 결합
            search_text = " ".join(search_fields)
            
            # HNSCC 키워드 검색
            if keyword_pattern.search(search_text):
                matched_sample = {
                    'sample_index': i,
                    'sample_id': sample.get('sample_id', f'sample_{i}'),
                    'meta': meta,
                    'search_text': search_text,
                    'matched_keywords': keyword_pattern.findall(search_text)
                }
                matched_samples.append(matched_sample)
                
                logger.info(f"매칭 샘플 발견! 샘플 {i}: {meta}")
                
                # 충분한 샘플을 찾았으면 조기 종료
                if len(matched_samples) >= 10:
                    logger.info("충분한 샘플을 찾았습니다. 검색을 종료합니다.")
                    break
    
    except Exception as e:
        logger.error(f"데이터셋 로드 중 오류 발생: {e}")
        
        # 오류 발생시 CSV 파일에서 검색 시도
        logger.info("CSV 파일에서 검색을 시도합니다...")
        try:
            csv_path = '/home/students/hbae/.cache/huggingface/hub/datasets--MahmoodLab--hest/snapshots/50d244f17d799c2ea9218635c6f33d24801ffb13/HEST_v1_1_0.csv'
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info(f"CSV 파일 로드 성공: {df.shape}")
                
                # CSV에서 HNSCC 샘플 검색
                for idx, row in df.iterrows():
                    if idx >= max_samples:
                        break
                    
                    search_text = " ".join([
                        str(row.get('organ', '')),
                        str(row.get('disease_state', '')),
                        str(row.get('tissue', '')),
                        str(row.get('dataset_title', ''))
                    ])
                    
                    if keyword_pattern.search(search_text):
                        matched_sample = {
                            'sample_index': idx,
                            'sample_id': row.get('id', f'sample_{idx}'),
                            'meta': row.to_dict(),
                            'search_text': search_text,
                            'matched_keywords': keyword_pattern.findall(search_text)
                        }
                        matched_samples.append(matched_sample)
                        
                        logger.info(f"CSV에서 매칭 샘플 발견! 샘플 {idx}")
                        
                        if len(matched_samples) >= 10:
                            break
            else:
                logger.warning("CSV 파일을 찾을 수 없습니다.")
                
        except Exception as csv_error:
            logger.error(f"CSV 파일 검색 중 오류: {csv_error}")
    
    return matched_samples


def main():
    """메인 함수"""
    logger.info("=== HEST-1k 데이터셋 HNSCC 샘플 검색 시작 ===")
    
    # HNSCC 샘플 검색
    matched_samples = search_hnscc_samples(max_samples=10000)
    
    # 결과 출력
    logger.info("=== 검색 결과 ===")
    logger.info(f"매칭된 샘플 수: {len(matched_samples)}")
    
    if matched_samples:
        logger.info("\n=== 매칭된 샘플 예시 ===")
        for i, sample in enumerate(matched_samples[:5]):  # 최대 5개만 출력
            logger.info(f"\n샘플 {i+1}:")
            logger.info(f"  샘플 ID: {sample['sample_id']}")
            logger.info(f"  매칭 키워드: {sample['matched_keywords']}")
            logger.info(f"  메타데이터:")
            
            meta = sample['meta']
            for key, value in meta.items():
                if value is not None and str(value).strip():
                    logger.info(f"    {key}: {value}")
            
            logger.info(f"  검색 텍스트: {sample['search_text'][:200]}...")
    else:
        logger.info("No HNSCC-like samples found in HEST-1k")
        logger.info("HEST-1k 데이터셋에는 직접적인 HNSCC 샘플이 없는 것 같습니다.")
        logger.info("다른 암 관련 샘플들을 사용하여 파이프라인을 테스트할 수 있습니다.")
    
    logger.info("\n=== 검색 완료 ===")


if __name__ == "__main__":
    main()

