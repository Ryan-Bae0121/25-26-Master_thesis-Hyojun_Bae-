#!/usr/bin/env python3
"""
aggregate_visium_hd.py
======================
Visium HD bin count aggregation 스크립트

각 bin의 count를 그대로 사용 (square_008um 기준)
→ AnnData로 저장 (bin 1개 = count 벡터 1개)

Usage:
    python aggregate_visium_hd.py \
        --visium_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um \
        --barcodes_path /project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/visium_hd_barcodes.npy \
        --gene_list /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
        --output_path /project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/visium_hd_adata.h5ad
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--visium_path',   type=str, required=True,
                        help='square_008um 폴더 경로')
    parser.add_argument('--barcodes_path', type=str, required=True,
                        help='visium_hd_barcodes.npy 경로')
    parser.add_argument('--gene_list',     type=str, required=True,
                        help='HVG 유전자 리스트 경로 (.txt)')
    parser.add_argument('--output_path',   type=str, required=True,
                        help='저장할 AnnData 경로 (.h5ad)')
    parser.add_argument('--normalize',     action='store_true', default=True,
                        help='normalize_total + log1p 적용')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # ── [1] count matrix 로드
    print(f'\n[1] count matrix 로드: {args.visium_path}')
    adata = sc.read_10x_h5(f'{args.visium_path}/filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    print(f'    shape: {adata.shape}')

    # ── [2] spatial 좌표 추가
    print(f'\n[2] spatial 좌표 로드')
    xy = pd.read_parquet(
        f'{args.visium_path}/spatial/tissue_positions.parquet',
        engine='pyarrow'
    )
    scalefactors = json.load(open(f'{args.visium_path}/spatial/scalefactors_json.json'))
    scalef = scalefactors['tissue_hires_scalef']

    xy.set_index('barcode', inplace=True)
    xy = xy.loc[adata.obs.index]

    # fullres → hires 변환
    spatial = xy[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy() * scalef
    adata.obsm['spatial'] = spatial
    adata.uns['spatial'] = {'scalefactors': scalefactors}
    print(f'    scalef: {scalef}')

    # ── [3] 임베딩 추출한 바코드만 필터링
    print(f'\n[3] 바코드 필터링')
    target_barcodes = np.load(args.barcodes_path, allow_pickle=True)
    print(f'    임베딩 바코드 수: {len(target_barcodes):,}')

    # 공통 바코드만
    common = adata.obs.index.isin(target_barcodes)
    adata = adata[common].copy()
    print(f'    필터링 후: {adata.shape[0]:,}개 bin')

    # 바코드 순서를 임베딩과 동일하게 맞추기
    barcode_to_idx = {b: i for i, b in enumerate(adata.obs.index)}
    order = [barcode_to_idx[b] for b in target_barcodes if b in barcode_to_idx]
    adata = adata[order].copy()
    print(f'    순서 정렬 완료: {adata.shape[0]:,}개')

    # ── [4] 유전자 필터링
    print(f'\n[4] 유전자 필터링: {args.gene_list}')
    with open(args.gene_list) as f:
        hvg_genes = [line.strip() for line in f.readlines()]
    adata = adata[:, adata.var_names.isin(hvg_genes)].copy()
    print(f'    HVG 필터링 후: {adata.shape[1]}개 유전자')

    # ── [5] normalize
    if args.normalize:
        print(f'\n[5] normalize_total + log1p 적용')
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    # ── [6] 저장
    print(f'\n[6] 저장: {args.output_path}')
    adata.write_h5ad(args.output_path)

    print(f'\n✅ 완료!')
    print(f'   shape: {adata.shape} (bin × 유전자)')
    print(f'   저장: {args.output_path}')