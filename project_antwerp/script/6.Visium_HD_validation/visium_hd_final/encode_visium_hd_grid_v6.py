#!/usr/bin/env python3
"""
encode_visium_hd_grid_v6.py
============================
array_col, array_row 기반으로 타일 정의
→ 좌표 틀어짐 문제 해결, 정확히 n×n bin 보장

타일 크기: 9×9 bin = 72um × 72um (≈68um)
          또는 8×8 bin = 64um × 64um

Usage:
    python encode_visium_hd_grid_v6.py \
        --image_path /project_antwerp/hbae/data/visium_hd_tonsil/spatial/tissue_hires_image.png \
        --positions_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/tissue_positions.parquet \
        --scalefactors_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/scalefactors_json.json \
        --count_h5 /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/filtered_feature_bc_matrix.h5 \
        --h5_path /project_antwerp/hbae/data/visium_hd_tonsil/Visium_HD_FF_Human_Tonsil_feature_slice.h5 \
        --gene_list /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
        --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \
        --output_dir /project_antwerp/hbae/Loki_output/visium_hd_array_embeddings/fold_01/ \
        --bins_per_tile 9 \
        --device cuda:0
"""

import os
import argparse
import numpy as np
import json
import h5py
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
from anndata import AnnData
from PIL import Image
from tqdm import tqdm
from scipy.sparse import csr_matrix


def load_omiclip(checkpoint_path, device):
    from open_clip import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained(
        'coca_ViT-L-14',
        pretrained=checkpoint_path,
        device=device
    )
    model.eval()
    return model, preprocess


def encode_batch(model, batch_imgs, device):
    img_tensor = torch.stack(batch_imgs).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',        type=str, required=True)
    parser.add_argument('--positions_path',    type=str, required=True)
    parser.add_argument('--scalefactors_path', type=str, required=True)
    parser.add_argument('--count_h5',          type=str, required=True)
    parser.add_argument('--h5_path',           type=str, required=True)
    parser.add_argument('--gene_list',         type=str, required=True)
    parser.add_argument('--pretrained',        type=str, required=True)
    parser.add_argument('--output_dir',        type=str, required=True)
    parser.add_argument('--bins_per_tile',     type=int, default=9,
                        help='타일 한 변의 bin 수 (9→9×9=81bins≈72um, 8→8×8=64bins≈64um)')
    parser.add_argument('--device',            type=str, default='cuda:0')
    parser.add_argument('--batch_size',        type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── [1] 이미지 로드
    print(f'\n[1] 이미지 로드')
    img = np.array(Image.open(args.image_path).convert('RGB'))
    img_height, img_width = img.shape[:2]
    print(f'    이미지 크기: {img_width} x {img_height} px (hires)')

    # ── [2] scalefactors 로드
    print(f'\n[2] scalefactors 로드')
    with open(args.scalefactors_path) as f:
        sf = json.load(f)
    scalef              = sf['tissue_hires_scalef']
    microns_per_pixel   = sf['microns_per_pixel']
    um_per_px_hires     = microns_per_pixel / scalef
    bin_size_um         = sf.get('bin_size_um', 8.0)
    bin_px_hires        = bin_size_um / um_per_px_hires
    tile_px_hires       = int(round(args.bins_per_tile * bin_px_hires))
    tile_um             = args.bins_per_tile * bin_size_um

    print(f'    tissue_hires_scalef: {scalef}')
    print(f'    um/px (hires):       {um_per_px_hires:.4f}')
    print(f'    bin 크기:            {bin_size_um}um = {bin_px_hires:.2f}px (hires)')
    print(f'    bins_per_tile:       {args.bins_per_tile} × {args.bins_per_tile} = {args.bins_per_tile**2}개')
    print(f'    tile 크기:           {tile_px_hires}px (hires) = {tile_um}um')

    # ── [3] tissue_positions 로드
    print(f'\n[3] tissue_positions 로드')
    df = pd.read_parquet(args.positions_path, engine='pyarrow')
    df = df[df['in_tissue'] == 1].reset_index(drop=True)
    df['hires_col'] = df['pxl_col_in_fullres'] * scalef
    df['hires_row'] = df['pxl_row_in_fullres'] * scalef
    print(f'    in_tissue=1 bin: {len(df):,}')
    print(f'    array_col 범위: {df.array_col.min()} ~ {df.array_col.max()}')
    print(f'    array_row 범위: {df.array_row.min()} ~ {df.array_row.max()}')

    # ── [4] count matrix 로드 (sparse)
    print(f'\n[4] count matrix 로드 (sparse)')
    adata = sc.read_10x_h5(args.count_h5)
    adata.var_names_make_unique()
    with open(args.gene_list) as f:
        hvg_genes = [line.strip() for line in f.readlines()]
    adata = adata[:, adata.var_names.isin(hvg_genes)].copy()

    # 유전자 순서 정렬
    gene_order   = {g: i for i, g in enumerate(hvg_genes)}
    sorted_genes = sorted(adata.var_names, key=lambda g: gene_order.get(g, 99999))
    adata        = adata[:, sorted_genes].copy()
    gene_names   = np.array(list(adata.var_names))

    print(f'    shape: {adata.X.shape}')
    print(f'    유전자 순서 첫 3개: {list(gene_names[:3])}')
    print(f'    순서 일치: {list(gene_names[:3]) == hvg_genes[:3]}')

    # sparse 유지
    X_sparse = adata.X.tocsr() if hasattr(adata.X, 'tocsr') else csr_matrix(adata.X)

    # barcode → count matrix index 매핑
    barcode_to_idx = {b: i for i, b in enumerate(adata.obs.index)}
    df['count_idx'] = df['barcode'].map(barcode_to_idx)
    df = df.dropna(subset=['count_idx'])
    df['count_idx'] = df['count_idx'].astype(int)

    # ── [5] ★ array_col, array_row 기반으로 타일 목록 생성
    print(f'\n[5] 타일 목록 생성 (array_col/row 기반)')

    n = args.bins_per_tile
    col_min = df['array_col'].min()
    col_max = df['array_col'].max()
    row_min = df['array_row'].min()
    row_max = df['array_row'].max()
    print(f'    array grid: col {col_min}~{col_max}, row {row_min}~{row_max}')

    # n개씩 묶어서 타일 만들기
    tile_col_starts = range(col_min, col_max - n + 2, n)
    tile_row_starts = range(row_min, row_max - n + 2, n)

    # 각 타일의 대표 hires 좌표 계산용 lookup
    # array_col, array_row → hires 좌표 평균
    df_indexed = df.set_index(['array_row', 'array_col'])

    valid_tiles = []
    print(f'    타일 후보: {len(list(tile_col_starts)) * len(list(tile_row_starts)):,}개')

    for row_start in tile_row_starts:
        for col_start in tile_col_starts:
            # 이 타일에 속하는 bin들
            in_tile = (
                (df['array_col'] >= col_start) & (df['array_col'] < col_start + n) &
                (df['array_row'] >= row_start) & (df['array_row'] < row_start + n)
            )
            if in_tile.sum() == 0:
                continue

            # 타일 중심의 hires 좌표 계산
            tile_bins    = df[in_tile]
            center_col_h = tile_bins['hires_col'].mean()
            center_row_h = tile_bins['hires_row'].mean()

            # 이미지에서 자를 좌표 (중심 기준 ±tile_px_hires/2)
            half = tile_px_hires // 2
            img_col_start = int(round(center_col_h)) - half
            img_row_start = int(round(center_row_h)) - half
            img_col_end   = img_col_start + tile_px_hires
            img_row_end   = img_row_start + tile_px_hires

            # 이미지 범위 체크
            if img_col_start < 0 or img_row_start < 0:
                continue
            if img_col_end > img_width or img_row_end > img_height:
                continue

            valid_tiles.append({
                'array_col_start': col_start,
                'array_row_start': row_start,
                'img_col_start':   img_col_start,
                'img_row_start':   img_row_start,
                'center_col_h':    center_col_h,
                'center_row_h':    center_row_h,
                'bin_mask':        in_tile,
            })

    print(f'    유효 타일: {len(valid_tiles):,}개')
    print(f'    타일당 bin 수 (이론): {n}×{n} = {n**2}개')

    # ── [6] 모델 로드
    print(f'\n[6] 모델 로드')
    model, preprocess = load_omiclip(args.pretrained, device)
    print(f'    완료')

    # ── [7] 임베딩 추출 + bin aggregation
    print(f'\n[7] 임베딩 추출 + bin aggregation')

    all_embs   = []
    all_exprs  = []
    all_coords = []
    batch_imgs   = []
    batch_coords = []
    batch_exprs  = []

    bin_counts_check = []

    for tile in tqdm(valid_tiles, desc='처리 중'):
        # 이미지 크롭
        r0, r1 = tile['img_row_start'], tile['img_row_start'] + tile_px_hires
        c0, c1 = tile['img_col_start'], tile['img_col_start'] + tile_px_hires
        patch = img[r0:r1, c0:c1]
        batch_imgs.append(preprocess(Image.fromarray(patch)))
        batch_coords.append((tile['img_col_start'], tile['img_row_start']))

        # ★ array 기반 bin 선택 → sparse 합산
        tile_bin_idx = df[tile['bin_mask']]['count_idx'].values
        bin_counts_check.append(len(tile_bin_idx))
        if len(tile_bin_idx) > 0:
            tile_expr = np.array(X_sparse[tile_bin_idx].sum(axis=0)).flatten()
        else:
            tile_expr = np.zeros(X_sparse.shape[1], dtype=np.float32)
        batch_exprs.append(tile_expr)

        if len(batch_imgs) >= args.batch_size:
            all_embs.append(encode_batch(model, batch_imgs, device))
            all_coords.extend(batch_coords)
            all_exprs.append(np.array(batch_exprs))
            batch_imgs   = []
            batch_coords = []
            batch_exprs  = []

    if len(batch_imgs) > 0:
        all_embs.append(encode_batch(model, batch_imgs, device))
        all_coords.extend(batch_coords)
        all_exprs.append(np.array(batch_exprs))

    # ── [8] 결합
    all_embs   = np.concatenate(all_embs,  axis=0)
    all_exprs  = np.concatenate(all_exprs, axis=0).astype(np.float32)
    all_coords = np.array(all_coords)
    bin_counts_check = np.array(bin_counts_check)

    print(f'\n    tile_img_embs: {all_embs.shape}')
    print(f'    tile_exprs (normalize 전): {all_exprs.shape}')
    print(f'    zeros 비율 (normalize 전): {(all_exprs==0).mean():.4f}')
    print(f'    타일당 실제 bin 수: mean={bin_counts_check.mean():.1f}, min={bin_counts_check.min()}, max={bin_counts_check.max()}')

    # tile별 normalize_total + log1p
    print('\n    tile별 normalize_total + log1p 적용')
    agg_adata = AnnData(X=csr_matrix(all_exprs))
    sc.pp.normalize_total(agg_adata)
    sc.pp.log1p(agg_adata)
    all_exprs_norm = agg_adata.X.toarray()

    print(f'    normalize 후 mean: {all_exprs_norm.mean():.4f}, std: {all_exprs_norm.std():.4f}')
    print(f'    (HNSCC train 비교: mean≈0.29, std≈0.64)')

    # ── [9] 저장
    print(f'\n[8] 저장: {args.output_dir}')
    np.save(os.path.join(args.output_dir, 'tile_img_embs.npy'),   all_embs)
    np.save(os.path.join(args.output_dir, 'tile_exprs.npy'),      all_exprs_norm)
    np.save(os.path.join(args.output_dir, 'tile_coords.npy'),     all_coords)
    np.save(os.path.join(args.output_dir, 'tile_gene_names.npy'), gene_names)

    print(f'\n✅ 완료!')
    print(f'   tile_img_embs: {all_embs.shape}')
    print(f'   tile_exprs:    {all_exprs_norm.shape}')
    print(f'   tile_coords:   {all_coords.shape}')
    print(f'   타일당 bin 수: {args.bins_per_tile}×{args.bins_per_tile} = {args.bins_per_tile**2}개 ({tile_um}um)')