#!/usr/bin/env python3
"""
PyTorch DataLoader 템플릿 for HNSCC Tiles

Loki PredEx 파인튜닝을 위한 두 가지 데이터셋 클래스를 제공합니다:
1. HNSCCTilesRegression: 이미지 → 다중 유전자 회귀
2. HNSCCTilesCLIP: 이미지 ↔ gene-sentence 텍스트 대조학습
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    import anndata as ad
except ImportError:
    print("anndata 패키지를 설치해주세요: pip install anndata")
    exit(1)


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HNSCCTilesRegression(Dataset):
    """
    HNSCC 타일 이미지 → 다중 유전자 발현 회귀를 위한 데이터셋
    
    Args:
        data_root: 데이터 루트 디렉토리 (tiles_meta.parquet과 expression_log1p.h5ad가 있는 곳)
        transform: 이미지 변환 (기본: resize, ToTensor, Normalize)
        target_transform: 타겟 변환
    """
    
    def __init__(self, 
                 data_root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.data_root = Path(data_root)
        self.target_transform = target_transform
        
        # 기본 이미지 변환
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # 데이터 로드
        self._load_data()
    
    def _load_data(self):
        """메타데이터와 발현 데이터를 로드합니다."""
        # 메타데이터 로드
        meta_path = self.data_root / "tiles_meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {meta_path}")
        
        self.meta_df = pd.read_parquet(meta_path)
        logger.info(f"메타데이터 로드: {len(self.meta_df)} 타일")
        
        # 발현 데이터 로드
        expr_path = self.data_root / "expression_log1p.h5ad"
        if not expr_path.exists():
            raise FileNotFoundError(f"발현 데이터 파일을 찾을 수 없습니다: {expr_path}")
        
        self.adata = ad.read_h5ad(expr_path)
        logger.info(f"발현 데이터 로드: {self.adata.shape}")
        
        # 인덱스 정렬 확인
        if len(self.meta_df) != len(self.adata):
            raise ValueError(f"메타데이터와 발현 데이터의 길이가 다릅니다: {len(self.meta_df)} vs {len(self.adata)}")
        
        # 유전자 정보
        self.gene_names = self.adata.var_names.tolist()
        self.n_genes = len(self.gene_names)
        
        logger.info(f"유전자 수: {self.n_genes}")
    
    def __len__(self) -> int:
        return len(self.meta_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """인덱스에 해당하는 이미지와 발현 벡터를 반환합니다."""
        # 메타데이터 가져오기
        meta_row = self.meta_df.iloc[idx]
        
        # 이미지 로드
        # 이미지 경로 (tile_path 또는 image_path 사용)
        image_path = self.data_root / meta_row.get('tile_path', meta_row.get('image_path', ''))
        if not image_path.exists():
            logger.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            # 빈 이미지 생성
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.error(f"이미지 로드 실패 {image_path}: {e}")
                # 빈 이미지 생성
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 이미지 변환
        if self.transform:
            image = self.transform(image)
        
        # 발현 벡터 가져오기
        expression = torch.tensor(self.adata.X[idx], dtype=torch.float32)
        
        # 타겟 변환
        if self.target_transform:
            expression = self.target_transform(expression)
        
        return image, expression
    
    def get_gene_names(self) -> List[str]:
        """유전자 이름 리스트를 반환합니다."""
        return self.gene_names
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """특정 인덱스의 샘플 정보를 반환합니다."""
        meta_row = self.meta_df.iloc[idx]
        return {
            'sample_id': meta_row['sample_id'],
            'patient_id': meta_row['patient_id'],
            'x': meta_row['x'],
            'y': meta_row['y'],
            'platform': meta_row['platform'],
            'organ': meta_row['organ'],
            'diagnosis': meta_row['diagnosis'],
            'gene_sentence': meta_row['gene_sentence']
        }


class HNSCCTilesCLIP(Dataset):
    """
    HNSCC 타일 이미지 ↔ gene-sentence 텍스트 대조학습을 위한 데이터셋
    
    Args:
        data_root: 데이터 루트 디렉토리
        transform: 이미지 변환
        text_encoder: 텍스트 인코더 (토크나이저 등, 사용자가 주입)
        max_text_length: 최대 텍스트 길이
    """
    
    def __init__(self, 
                 data_root: str,
                 transform: Optional[Callable] = None,
                 text_encoder: Optional[Any] = None,
                 max_text_length: int = 512):
        
        self.data_root = Path(data_root)
        self.text_encoder = text_encoder
        self.max_text_length = max_text_length
        
        # 기본 이미지 변환
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # 데이터 로드
        self._load_data()
    
    def _load_data(self):
        """메타데이터를 로드합니다."""
        meta_path = self.data_root / "tiles_meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {meta_path}")
        
        self.meta_df = pd.read_parquet(meta_path)
        logger.info(f"메타데이터 로드: {len(self.meta_df)} 타일")
        
        # gene-sentence가 비어있는 샘플 필터링
        valid_mask = self.meta_df['gene_sentence'].notna() & (self.meta_df['gene_sentence'] != '')
        self.meta_df = self.meta_df[valid_mask].reset_index(drop=True)
        
        logger.info(f"유효한 gene-sentence가 있는 타일: {len(self.meta_df)}")
    
    def __len__(self) -> int:
        return len(self.meta_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """인덱스에 해당하는 이미지, 텍스트, 텍스트 토큰을 반환합니다."""
        # 메타데이터 가져오기
        meta_row = self.meta_df.iloc[idx]
        
        # 이미지 로드
        # 이미지 경로 (tile_path 또는 image_path 사용)
        image_path = self.data_root / meta_row.get('tile_path', meta_row.get('image_path', ''))
        if not image_path.exists():
            logger.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            # 빈 이미지 생성
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.error(f"이미지 로드 실패 {image_path}: {e}")
                # 빈 이미지 생성
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 이미지 변환
        if self.transform:
            image = self.transform(image)
        
        # 텍스트 가져오기
        text = str(meta_row['gene_sentence'])
        
        # 텍스트 인코딩
        if self.text_encoder:
            try:
                # HuggingFace 토크나이저인 경우
                if hasattr(self.text_encoder, 'encode'):
                    text_tokens = self.text_encoder.encode(
                        text, 
                        max_length=self.max_text_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    ).squeeze(0)
                # 일반 토크나이저인 경우
                elif hasattr(self.text_encoder, 'tokenize'):
                    tokens = self.text_encoder.tokenize(text)
                    if len(tokens) > self.max_text_length:
                        tokens = tokens[:self.max_text_length]
                    text_tokens = torch.tensor(tokens, dtype=torch.long)
                else:
                    # 텍스트를 단순히 인덱스로 변환 (placeholder)
                    text_tokens = torch.zeros(self.max_text_length, dtype=torch.long)
            except Exception as e:
                logger.warning(f"텍스트 인코딩 실패: {e}")
                text_tokens = torch.zeros(self.max_text_length, dtype=torch.long)
        else:
            # 텍스트 인코더가 없으면 placeholder 반환
            text_tokens = torch.zeros(self.max_text_length, dtype=torch.long)
        
        return image, text, text_tokens
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """특정 인덱스의 샘플 정보를 반환합니다."""
        meta_row = self.meta_df.iloc[idx]
        return {
            'sample_id': meta_row['sample_id'],
            'patient_id': meta_row['patient_id'],
            'x': meta_row['x'],
            'y': meta_row['y'],
            'platform': meta_row['platform'],
            'organ': meta_row['organ'],
            'diagnosis': meta_row['diagnosis'],
            'gene_sentence': meta_row['gene_sentence']
        }


def create_regression_dataloader(data_root: str,
                                batch_size: int = 32,
                                shuffle: bool = True,
                                num_workers: int = 4,
                                transform: Optional[Callable] = None) -> DataLoader:
    """회귀용 DataLoader를 생성합니다."""
    dataset = HNSCCTilesRegression(data_root, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    return dataloader


def create_clip_dataloader(data_root: str,
                          batch_size: int = 32,
                          shuffle: bool = True,
                          num_workers: int = 4,
                          text_encoder: Optional[Any] = None,
                          transform: Optional[Callable] = None) -> DataLoader:
    """CLIP용 DataLoader를 생성합니다."""
    dataset = HNSCCTilesCLIP(data_root, transform=transform, text_encoder=text_encoder)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    return dataloader


def smoke_test(data_root: str, num_batches: int = 3):
    """DataLoader 동작을 검증하는 스모크 테스트를 실행합니다."""
    logger.info("DataLoader 스모크 테스트 시작")
    
    try:
        # 회귀 데이터셋 테스트
        logger.info("회귀 데이터셋 테스트...")
        regression_loader = create_regression_dataloader(
            data_root, 
            batch_size=8, 
            num_workers=0  # 디버깅을 위해 0으로 설정
        )
        
        for i, (images, expressions) in enumerate(regression_loader):
            logger.info(f"배치 {i+1}: 이미지 {images.shape}, 발현 {expressions.shape}")
            
            # 기본 검증
            assert images.shape[0] == expressions.shape[0], "배치 크기가 다릅니다"
            assert images.shape[1:] == (3, 224, 224), f"이미지 크기 오류: {images.shape[1:]}"
            assert expressions.dtype == torch.float32, "발현 데이터 타입 오류"
            
            if i >= num_batches - 1:
                break
        
        logger.info("회귀 데이터셋 테스트 성공!")
        
        # CLIP 데이터셋 테스트
        logger.info("CLIP 데이터셋 테스트...")
        clip_loader = create_clip_dataloader(
            data_root, 
            batch_size=8, 
            num_workers=0
        )
        
        for i, (images, texts, text_tokens) in enumerate(clip_loader):
            logger.info(f"배치 {i+1}: 이미지 {images.shape}, 텍스트 토큰 {text_tokens.shape}")
            logger.info(f"텍스트 예시: {texts[0][:100]}...")
            
            # 기본 검증
            assert images.shape[0] == text_tokens.shape[0], "배치 크기가 다릅니다"
            assert images.shape[1:] == (3, 224, 224), f"이미지 크기 오류: {images.shape[1:]}"
            assert text_tokens.dtype == torch.long, "텍스트 토큰 타입 오류"
            
            if i >= num_batches - 1:
                break
        
        logger.info("CLIP 데이터셋 테스트 성공!")
        
        # 데이터셋 정보 출력
        regression_dataset = regression_loader.dataset
        clip_dataset = clip_loader.dataset
        
        logger.info(f"\n데이터셋 정보:")
        logger.info(f"회귀 데이터셋 크기: {len(regression_dataset)}")
        logger.info(f"CLIP 데이터셋 크기: {len(clip_dataset)}")
        logger.info(f"유전자 수: {regression_dataset.n_genes}")
        
        # 샘플 정보 출력
        sample_info = regression_dataset.get_sample_info(0)
        logger.info(f"\n샘플 정보 예시:")
        for key, value in sample_info.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\n스모크 테스트 완료!")
        
    except Exception as e:
        logger.error(f"스모크 테스트 실패: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("사용법: python dataloaders.py <data_root>")
        sys.exit(1)
    
    data_root = sys.argv[1]
    smoke_test(data_root)
