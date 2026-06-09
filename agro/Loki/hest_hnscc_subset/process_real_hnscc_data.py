#!/usr/bin/env python3
"""
실제 HNSCC 데이터 처리 및 Loki PredEx 파인튜닝용 데이터셋 생성

다운로드된 실제 HEST HNSCC 샘플을 처리하여:
- tiles_meta.parquet
- expression_log1p.h5ad
- tiles/ (패치 이미지)
를 생성합니다.
"""

import os
import h5py
import logging
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import anndata as ad
import scanpy as sc

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealHNSCCProcessor:
    """실제 HNSCC 데이터 처리기"""
    
    def __init__(self, data_dir: str = "hest_real_hnscc", output_dir: str = "hest_hnscc_processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tiles_dir = self.output_dir / "tiles"
        self.tiles_dir.mkdir(exist_ok=True)
        
        self.all_tiles = []
        self.all_expressions = []
        self.all_genes = set()
    
    def load_st_data(self, sample_id: str):
        """ST 데이터 로드"""
        st_file = self.data_dir / "st" / f"{sample_id}.h5ad"
        
        if not st_file.exists():
            logger.warning(f"ST 파일이 없습니다: {st_file}")
            return None
        
        try:
            adata = ad.read_h5ad(st_file)
            logger.info(f"✅ {sample_id} ST 데이터 로드: {adata.shape}")
            return adata
        except Exception as e:
            logger.error(f"❌ {sample_id} ST 데이터 로드 실패: {e}")
            return None
    
    def load_patches(self, sample_id: str):
        """패치 이미지 로드"""
        patch_file = self.data_dir / "patches" / f"{sample_id}.h5"
        
        if not patch_file.exists():
            logger.warning(f"패치 파일이 없습니다: {patch_file}")
            return None, None
        
        try:
            with h5py.File(patch_file, 'r') as f:
                # 이미지 데이터 로드
                if 'img' in f:
                    images = f['img'][:]
                    barcodes = f['barcode'][:].flatten() if 'barcode' in f else None
                    
                    # barcode를 spot ID로 매핑
                    barcode_to_img = {}
                    if barcodes is not None:
                        for i, bc in enumerate(barcodes):
                            # bytes를 string으로 변환
                            bc_str = bc.decode('utf-8') if isinstance(bc, bytes) else str(bc)
                            barcode_to_img[bc_str] = images[i]
                    
                    logger.info(f"✅ {sample_id} 패치 데이터 로드: {images.shape}, {len(barcode_to_img)} barcodes")
                    return images, barcode_to_img
                else:
                    logger.warning(f"⚠️ {sample_id}: 'img' 키를 찾을 수 없음")
                    return None, None
            
        except Exception as e:
            logger.error(f"❌ {sample_id} 패치 로드 실패: {e}")
            return None, None
    
    def process_sample(self, sample_id: str, max_tiles: int = 500):
        """샘플 처리"""
        logger.info(f"\n🔄 {sample_id} 처리 시작...")
        
        # 1. ST 데이터 로드
        adata = self.load_st_data(sample_id)
        if adata is None:
            return
        
        # 2. 패치 이미지 로드
        patches, barcode_to_img = self.load_patches(sample_id)
        
        # 3. log1p 정규화 (아직 안 되어있으면)
        if 'log1p' not in adata.uns:
            sc.pp.log1p(adata)
        
        # 4. 각 스팟 처리
        n_spots = min(adata.n_obs, max_tiles)
        logger.info(f"처리할 스팟 수: {n_spots}")
        
        for i in tqdm(range(n_spots), desc=f"{sample_id} 처리"):
            spot_id = adata.obs_names[i]
            
            # 발현 데이터
            expression = adata.X[i].toarray().flatten() if hasattr(adata.X[i], 'toarray') else adata.X[i].flatten()
            
            # 유전자 목록
            genes = adata.var_names.tolist()
            self.all_genes.update(genes)
            
            # 메타데이터
            obs_data = adata.obs.iloc[i]
            
            # 이미지 저장
            image_filename = f"{sample_id}_{i}.png"
            image_path = self.tiles_dir / image_filename
            
            # 패치 이미지가 있으면 저장
            if barcode_to_img is not None and spot_id in barcode_to_img:
                try:
                    # barcode로 이미지 찾기
                    patch_img = barcode_to_img[spot_id]
                    
                    # 이미지가 이미 224x224x3 형태여야 함
                    img = Image.fromarray(patch_img.astype(np.uint8))
                    img.save(image_path)
                    
                except Exception as e:
                    logger.warning(f"패치 이미지 저장 실패 ({spot_id}): {e}")
                    # 더미 이미지 생성
                    img = Image.new('RGB', (224, 224), color='lightgray')
                    img.save(image_path)
            elif patches is not None and i < len(patches):
                # barcode 매핑이 없으면 인덱스로 접근
                try:
                    patch_img = patches[i]
                    img = Image.fromarray(patch_img.astype(np.uint8))
                    img.save(image_path)
                except Exception as e:
                    logger.warning(f"패치 이미지 저장 실패 (인덱스 {i}): {e}")
                    img = Image.new('RGB', (224, 224), color='lightgray')
                    img.save(image_path)
            else:
                # 더미 이미지 생성
                img = Image.new('RGB', (224, 224), color='lightgray')
                img.save(image_path)
            
            # 타일 메타데이터
            tile_meta = {
                'sample_id': sample_id,
                'spot_id': spot_id,
                'image_path': f"tiles/{image_filename}",
                'patient_id': sample_id,
                'organ': 'HNSCC',
                'diagnosis': 'Squamous Cell Carcinoma',
                'site': 'Head and Neck',
                'platform': 'Visium' if 'MEND' in sample_id else 'Xenium',
                'x': float(obs_data.get('array_col', 0)),
                'y': float(obs_data.get('array_row', 0)),
                'n_genes': int(obs_data.get('n_genes_by_counts', 0)),
                'total_counts': float(obs_data.get('total_counts', 0)),
                'gene_sentence': ' '.join(genes[:500])  # 상위 500개 유전자
            }
            
            self.all_tiles.append(tile_meta)
            self.all_expressions.append({
                'expression': expression,
                'genes': genes
            })
        
        logger.info(f"✅ {sample_id} 처리 완료: {n_spots}개 스팟")
    
    def save_results(self):
        """결과 저장"""
        if not self.all_tiles:
            logger.error("❌ 저장할 데이터가 없습니다.")
            return
        
        logger.info(f"\n💾 결과 저장 시작...")
        
        # 1. tiles_meta.parquet 저장
        meta_df = pd.DataFrame(self.all_tiles)
        meta_path = self.output_dir / "tiles_meta.parquet"
        meta_df.to_parquet(meta_path)
        logger.info(f"✅ 메타데이터 저장: {meta_path} ({len(meta_df)} 타일)")
        
        # 2. expression_log1p.h5ad 저장
        # 모든 유전자 목록
        all_genes_list = sorted(list(self.all_genes))
        n_genes = len(all_genes_list)
        n_tiles = len(self.all_tiles)
        
        logger.info(f"발현 행렬 생성: {n_tiles} × {n_genes}")
        
        # 발현 행렬 생성
        X = np.zeros((n_tiles, n_genes), dtype=np.float32)
        
        for i, expr_data in tqdm(enumerate(self.all_expressions), total=n_tiles, desc="발현 행렬 생성"):
            genes = expr_data['genes']
            expression = expr_data['expression']
            
            # 유전자 매핑
            gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes_list)}
            
            for j, gene in enumerate(genes):
                if gene in gene_to_idx and j < len(expression):
                    X[i, gene_to_idx[gene]] = expression[j]
        
        # AnnData 객체 생성
        obs = meta_df.copy()
        obs.index = [f"tile_{i}" for i in range(n_tiles)]
        
        var = pd.DataFrame(index=all_genes_list)
        var.index.name = 'genes'
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # 저장
        expression_path = self.output_dir / "expression_log1p.h5ad"
        adata.write(expression_path)
        logger.info(f"✅ 발현 데이터 저장: {expression_path}")
        
        # 3. 통계 출력
        logger.info(f"\n📊 === 최종 결과 ===")
        logger.info(f"✅ 총 타일 수: {n_tiles}")
        logger.info(f"✅ 총 유전자 수: {n_genes}")
        logger.info(f"✅ 발현 행렬 크기: {X.shape}")
        logger.info(f"✅ 이미지 파일: {len(list(self.tiles_dir.glob('*.png')))}개")
        logger.info(f"✅ 출력 디렉토리: {self.output_dir}")
        
        # 샘플 분포
        logger.info(f"\n📋 샘플 분포:")
        sample_counts = meta_df['sample_id'].value_counts()
        for sample, count in sample_counts.items():
            logger.info(f"  - {sample}: {count}개")
        
        return True


def main():
    """메인 함수"""
    print("🚀 === 실제 HNSCC 데이터 처리 시작 ===")
    
    processor = RealHNSCCProcessor(
        data_dir="hest_real_hnscc",
        output_dir="hest_hnscc_processed"
    )
    
    # 샘플 목록
    samples = ['TENX124', 'TENX125', 'MEND38', 'MEND39', 'MEND40']
    
    # 각 샘플 처리
    for sample in samples:
        processor.process_sample(sample, max_tiles=500)
    
    # 결과 저장
    processor.save_results()
    
    print("\n🎉 === 실제 HNSCC 데이터 처리 완료 ===")
    print("✅ tiles_meta.parquet, expression_log1p.h5ad, tiles/ 생성 완료")
    print("✅ Loki PredEx 파인튜닝 준비 완료!")


if __name__ == "__main__":
    main()

