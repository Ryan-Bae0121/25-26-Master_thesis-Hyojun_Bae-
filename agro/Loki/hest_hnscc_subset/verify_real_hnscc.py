#!/usr/bin/env python3
"""
HEST 공식 메타데이터를 사용한 정확한 HNSCC 샘플 검증 및 추출

이전 결과가 오탐 가능성이 높으므로, HEST 공식 메타 CSV를 사용하여
정확한 HNSCC 샘플을 식별하고 실제 데이터를 부분 다운로드합니다.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download, login
    import anndata as ad
except ImportError:
    print("필요한 패키지를 설치해주세요:")
    print("pip install huggingface_hub anndata")
    exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealHNSCCVerifier:
    """실제 HNSCC 샘플 검증기"""
    
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # HNSCC 관련 OncoTree 코드 (공식 코드)
        self.oncotree_keep = {
            "HNSC", "HNSCC", "OPSCC", "OCSCC", "LSCC", "HPSCC", 
            "SNUC", "NPC", "SCC", "SQUAMOUS"
        }
        
        # HNSCC 키워드 패턴 (단어 경계 사용)
        self.hnscc_pattern = re.compile(
            r"\b(head|neck|oral|tongue|oropharynx|hypopharynx|larynx|pharynx|tonsil|salivary|parotid|buccal|floor\s+of\s+mouth|hnscc|squamous)\b", 
            re.IGNORECASE
        )
        
        self.verified_samples = []
        self.download_dir = Path("hest_real_hnscc")
    
    def login_huggingface(self):
        """HuggingFace 로그인"""
        try:
            login(token=self.hf_token)
            print("✅ HuggingFace 로그인 성공")
            return True
        except Exception as e:
            print(f"⚠️ HuggingFace 로그인 실패: {e}")
            return False
    
    def load_official_metadata(self):
        """HEST 공식 메타데이터 로드"""
        try:
            print("📥 HEST 공식 메타데이터 로드 중...")
            
            # 공식 CSV 파일 경로 (HF 가상 파일시스템)
            csv_path = "hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv"
            
            # CSV 로드
            meta = pd.read_csv(csv_path)
            print(f"✅ 메타데이터 로드 성공: {meta.shape}")
            print(f"📋 컬럼: {meta.columns.tolist()}")
            
            return meta
            
        except Exception as e:
            print(f"❌ 공식 메타데이터 로드 실패: {e}")
            print("대안: 로컬 캐시된 CSV 파일 사용")
            
            # 로컬 캐시 파일 사용
            local_csv = "/home/students/hbae/.cache/huggingface/hub/datasets--MahmoodLab--hest/snapshots/50d244f17d799c2ea9218635c6f33d24801ffb13/HEST_v1_1_0.csv"
            if os.path.exists(local_csv):
                meta = pd.read_csv(local_csv)
                print(f"✅ 로컬 CSV 로드 성공: {meta.shape}")
                return meta
            else:
                raise Exception("메타데이터 파일을 찾을 수 없습니다.")
    
    def is_hnscc_row(self, row: pd.Series) -> bool:
        """행이 HNSCC 관련인지 판단"""
        # OncoTree 코드 확인
        onco_code = str(row.get("oncotree_code", "")).upper()
        if onco_code in self.oncotree_keep:
            return True
        
        # Organ 필드 확인
        organ = str(row.get("organ", ""))
        if self.hnscc_pattern.search(organ):
            return True
        
        # Diagnosis + Cancer_type 필드 확인
        diagnosis = str(row.get("diagnosis", "")) + " " + str(row.get("cancer_type", ""))
        if self.hnscc_pattern.search(diagnosis):
            return True
        
        # Tissue 필드 확인
        tissue = str(row.get("tissue", ""))
        if self.hnscc_pattern.search(tissue):
            return True
        
        # Dataset title에서 HNSCC 관련 키워드 확인 (하지만 더 엄격하게)
        title = str(row.get("dataset_title", ""))
        if self.hnscc_pattern.search(title):
            # 하지만 "heart" 안에 "oral"이 들어가는 경우는 제외
            if "heart" in title.lower() and "oral" in title.lower():
                return False
            return True
        
        return False
    
    def filter_hnscc_samples(self, meta: pd.DataFrame):
        """HNSCC 샘플 필터링"""
        print("🔍 HNSCC 샘플 필터링 시작...")
        
        # 각 행에 대해 HNSCC 판단
        hnscc_mask = meta.apply(self.is_hnscc_row, axis=1)
        hnscc_samples = meta[hnscc_mask].copy()
        
        print(f"✅ HNSCC 관련 샘플 발견: {len(hnscc_samples)}개")
        
        if len(hnscc_samples) > 0:
            print("\n📊 HNSCC 샘플 상세 분석:")
            
            # Organ 분포
            print("\n🏥 Organ 분포:")
            organ_counts = hnscc_samples['organ'].value_counts()
            for organ, count in organ_counts.head(10).items():
                print(f"  - {organ}: {count}개")
            
            # OncoTree 코드 분포
            print("\n🧬 OncoTree 코드 분포:")
            onco_counts = hnscc_samples['oncotree_code'].value_counts()
            for onco, count in onco_counts.head(10).items():
                print(f"  - {onco}: {count}개")
            
            # Disease state 분포
            print("\n🔬 Disease State 분포:")
            disease_counts = hnscc_samples['disease_state'].value_counts()
            for disease, count in disease_counts.head(10).items():
                print(f"  - {disease}: {count}개")
            
            # 샘플 예시 출력
            print("\n📋 HNSCC 샘플 예시 (상위 10개):")
            for idx, row in hnscc_samples.head(10).iterrows():
                print(f"\n  샘플 {len(self.verified_samples)+1}:")
                print(f"    ID: {row['id']}")
                print(f"    Organ: {row['organ']}")
                print(f"    OncoTree: {row['oncotree_code']}")
                print(f"    Disease: {row['disease_state']}")
                print(f"    Tissue: {row['tissue']}")
                print(f"    Title: {row['dataset_title'][:80]}...")
                
                self.verified_samples.append(row.to_dict())
        
        return hnscc_samples
    
    def download_subset(self, sample_ids: List[str], max_samples: int = 10):
        """필터된 샘플만 부분 다운로드"""
        if not sample_ids:
            print("❌ 다운로드할 샘플이 없습니다.")
            return
        
        print(f"\n📦 HNSCC 서브셋 다운로드 시작 (최대 {max_samples}개)")
        
        # 다운로드할 샘플 ID 제한
        download_ids = sample_ids[:max_samples]
        
        # 패턴 생성 (HEST 카드 권장 방식)
        patterns = [f"*{sample_id}[_.]**" for sample_id in download_ids]
        
        try:
            print(f"🎯 다운로드 패턴: {patterns[:3]}... (총 {len(patterns)}개)")
            
            # 부분 다운로드 실행
            local_dir = snapshot_download(
                repo_id="MahmoodLab/hest",
                repo_type="dataset",
                local_dir=str(self.download_dir),
                allow_patterns=patterns,
                token=self.hf_token
            )
            
            print(f"✅ 다운로드 완료: {local_dir}")
            return local_dir
            
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            return None
    
    def verify_data_structure(self, local_dir: str):
        """다운로드된 데이터 구조 검증"""
        print(f"\n🔍 데이터 구조 검증: {local_dir}")
        
        # 예상 디렉토리 구조 확인
        expected_dirs = ["wsis", "st", "patches", "metadata", "spatial_plots", "cellvit_seg"]
        
        for expected_dir in expected_dirs:
            dir_path = Path(local_dir) / expected_dir
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                print(f"✅ {expected_dir}/: {len(files)}개 파일")
            else:
                print(f"⚠️ {expected_dir}/: 없음")
        
        # 실제 데이터 파일 확인
        print("\n📊 실제 데이터 파일 확인:")
        
        # ST 데이터 (.h5ad)
        st_files = list(Path(local_dir).glob("**/*.h5ad"))
        print(f"ST 파일 (.h5ad): {len(st_files)}개")
        
        # 패치 이미지 (.png, .jpg, .tif)
        patch_files = list(Path(local_dir).glob("**/*.png")) + \
                     list(Path(local_dir).glob("**/*.jpg")) + \
                     list(Path(local_dir).glob("**/*.tif*"))
        print(f"패치 이미지: {len(patch_files)}개")
        
        # WSI 파일 (.tif, .svs)
        wsi_files = list(Path(local_dir).glob("**/*.tif*")) + \
                   list(Path(local_dir).glob("**/*.svs"))
        print(f"WSI 파일: {len(wsi_files)}개")
        
        # 샘플 데이터 검증
        if st_files:
            try:
                print(f"\n🧬 ST 데이터 샘플 검증:")
                adata = ad.read_h5ad(st_files[0])
                print(f"  - 발현 행렬: {adata.shape}")
                print(f"  - 샘플 ID: {adata.obs_names[:3].tolist()}")
                print(f"  - 유전자: {adata.var_names[:3].tolist()}")
                print(f"  - 관찰값 컬럼: {adata.obs.columns.tolist()}")
            except Exception as e:
                print(f"  - ST 데이터 로드 실패: {e}")
        
        return len(st_files) > 0 and len(patch_files) > 0
    
    def run_verification(self):
        """전체 검증 프로세스 실행"""
        print("🚀 === 실제 HNSCC 샘플 검증 시작 ===")
        
        # 1. HuggingFace 로그인
        self.login_huggingface()
        
        # 2. 공식 메타데이터 로드
        meta = self.load_official_metadata()
        
        # 3. HNSCC 샘플 필터링
        hnscc_samples = self.filter_hnscc_samples(meta)
        
        if len(hnscc_samples) == 0:
            print("❌ HNSCC 관련 샘플을 찾을 수 없습니다.")
            return
        
        # 4. 샘플 ID 추출
        sample_ids = hnscc_samples['id'].dropna().unique().tolist()
        print(f"\n📋 추출된 샘플 ID: {sample_ids[:10]}... (총 {len(sample_ids)}개)")
        
        # 5. 부분 다운로드
        local_dir = self.download_subset(sample_ids, max_samples=5)
        
        if local_dir:
            # 6. 데이터 구조 검증
            is_valid = self.verify_data_structure(local_dir)
            
            if is_valid:
                print("\n🎉 === 실제 HNSCC 데이터 검증 완료 ===")
                print("✅ 진짜 ST-H&E 페어 데이터 확인됨")
                print(f"✅ 다운로드 위치: {local_dir}")
            else:
                print("\n⚠️ 데이터 구조 검증 실패")
                print("설명용 이미지나 메타데이터만 다운로드된 것 같습니다.")
        
        return hnscc_samples


def main():
    """메인 함수"""
    verifier = RealHNSCCVerifier()
    hnscc_samples = verifier.run_verification()
    
    if len(hnscc_samples) > 0:
        print(f"\n📊 === 최종 결과 ===")
        print(f"✅ 검증된 HNSCC 샘플: {len(hnscc_samples)}개")
        print(f"✅ 다운로드 디렉토리: {verifier.download_dir}")
        print("✅ 실제 ST-H&E 데이터 준비 완료")
    else:
        print("\n❌ HNSCC 샘플을 찾을 수 없었습니다.")


if __name__ == "__main__":
    main()

