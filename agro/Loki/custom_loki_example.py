#!/usr/bin/env python3
"""
Loki PredEx를 사용한 커스텀 데이터 분석 예시
"""

import sys
sys.path.append('/home/students/hbae/Loki/src')

import loki
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import os

def run_loki_prediction(
    image_paths: list,
    st_sentences: list, 
    train_gene_expression: np.ndarray,
    model_path: str = None,
    device: str = "cuda"
):
    """
    Loki PredEx를 사용하여 유전자 발현 예측
    
    Args:
        image_paths: 분석할 이미지 파일 경로 리스트
        st_sentences: 각 이미지에 대응하는 ST sentence 리스트
        train_gene_expression: 훈련용 유전자 발현 데이터 (n_samples, n_genes)
        model_path: OmiCLIP 모델 경로 (None이면 기본 모델 사용)
        device: 사용할 디바이스 ("cuda" 또는 "cpu")
    
    Returns:
        predicted_expression: 예측된 유전자 발현 데이터
    """
    
    print("=== Loki PredEx 실행 시작 ===")
    
    # 1. 모델 로드
    print("1. OmiCLIP 모델 로딩...")
    if model_path is None:
        # 기본 모델 경로 (실제 경로로 수정 필요)
        model_path = "/path/to/omiclip_model.pth"
    
    model, preprocess, tokenizer = loki.utils.load_model(
        model_path=model_path,
        device=device
    )
    
    # 2. 이미지 인코딩
    print("2. 이미지 인코딩...")
    image_embeddings = loki.utils.encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        device=device
    )
    print(f"   이미지 임베딩 형태: {image_embeddings.shape}")
    
    # 3. 텍스트 인코딩
    print("3. 텍스트 인코딩...")
    text_embeddings = loki.utils.encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=st_sentences,
        device=device
    )
    print(f"   텍스트 임베딩 형태: {text_embeddings.shape}")
    
    # 4. 유사성 계산
    print("4. 이미지-텍스트 유사성 계산...")
    similarity_matrix = loki.utils.compute_similarity(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings
    )
    print(f"   유사성 행렬 형태: {similarity_matrix.shape}")
    
    # 5. 유전자 발현 예측
    print("5. 유전자 발현 예측...")
    predicted_expression = loki.predex.predict_st_gene_expr(
        image_text_similarity=similarity_matrix,
        train_data=train_gene_expression
    )
    print(f"   예측 결과 형태: {predicted_expression.shape}")
    
    print("=== Loki PredEx 실행 완료 ===")
    
    return predicted_expression

def prepare_sample_data():
    """
    샘플 데이터 준비 예시
    """
    print("=== 샘플 데이터 준비 ===")
    
    # 1. 샘플 이미지 경로 (실제 이미지 파일로 교체 필요)
    image_paths = [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg", 
        "/path/to/image3.jpg"
    ]
    
    # 2. 샘플 ST sentences (실제 유전자 발현 패턴으로 교체 필요)
    st_sentences = [
        "NPPA MYH6 MYL7 TNNT2 DES MB PTGDS MYL4 CRYAB",
        "GENE1 GENE2 GENE3 GENE4 GENE5 GENE6 GENE7",
        "ANOTHER SET OF GENES WITH DIFFERENT PATTERN"
    ]
    
    # 3. 샘플 훈련 데이터 (실제 유전자 발현 데이터로 교체 필요)
    # 형태: (n_samples, n_genes)
    train_gene_expression = np.random.rand(3, 100)  # 3개 샘플, 100개 유전자
    
    return image_paths, st_sentences, train_gene_expression

def main():
    """
    메인 실행 함수
    """
    print("Loki PredEx 커스텀 데이터 분석 예시")
    print("=" * 50)
    
    # 샘플 데이터 준비
    image_paths, st_sentences, train_data = prepare_sample_data()
    
    # Loki 예측 실행
    try:
        predicted_expression = run_loki_prediction(
            image_paths=image_paths,
            st_sentences=st_sentences,
            train_gene_expression=train_data,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"\n예측 완료! 결과 형태: {predicted_expression.shape}")
        print(f"예측된 유전자 발현 데이터:\n{predicted_expression}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("실제 데이터 경로와 모델 경로를 확인해주세요.")

if __name__ == "__main__":
    main()




