#!/usr/bin/env python3
"""
encode_visium_hd_grid.py
========================
Visium HD 그리드 타일링 + 임베딩 추출 + bin aggregation

Usage:
    python encode_visium_hd_grid.py \
        --image_path /project_antwerp/hbae/data/visium_hd_tonsil/spatial/tissue_hires_image.png \
        --positions_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/tissue_positions.parquet \
        --scalefactors_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/scalefactors_json.json \
        --count_h5 /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/filtered_feature_bc_matrix.h5 \
        --h5_path /project_antwerp/hbae/data/visium_hd_tonsil/Visium_HD_FF_Human_Tonsil_feature_slice.h5 \
        --gene_list /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
        --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \
        --output_dir /project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/ \
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


def calc_tile_size(h5_path, img_width, img_height, target_um=68.0):
    with h5py.File(h5_path, 'r') as f:
        meta = json.loads(f.attrs['metadata_json'])
    spot_pitch  = meta['spot_pitch']
    nrows       = meta['nrows']
    ncols       = meta['ncols']
    um_per_px_x = (ncols * spot_pitch) / img_width
    um_per_px_y = (nrows * spot_pitch) / img_height
    tile_px_x   = target_um / um_per_px_x
    tile_px_y   = target_um / um_per_px_y
    tile_size   = int(round((tile_px_x + tile_px_y) / 2))
    print(f'  µm/px (x): {um_per_px_x:.4f}, µm/px (y): {um_per_px_y:.4f}')
    print(f'  {target_um}µm → tile_size: {tile_size}px')
    return tile_size


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
    parser.add_argument('--target_um',         type=float, default=68.0)
    parser.add_argument('--device',            type=str,  default='cuda:0')
    parser.add_argument('--batch_size',        type=int,  default=64)
    parser.add_argument('--brightness_thresh', type=float, default=220.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── [1] 이미지 로드
    print(f'\n[1] 이미지 로드')
    img = np.array(Image.open(args.image_path).convert('RGB'))
    img_height, img_width = img.shape[:2]
    print(f'    이미지 크기: {img_width} x {img_height} px')

    # ── [2] scalefactors 로드
    print(f'\n[2] scalefactors 로드')
    with open(args.scalefactors_path) as f:
        scalefactors = json.load(f)
    scalef = scalefactors['tissue_hires_scalef']
    print(f'    tissue_hires_scalef: {scalef}')

    # ── [3] tissue_positions 로드
    print(f'\n[3] tissue_positions 로드')
    df = pd.read_parquet(args.positions_path, engine='pyarrow')
    df = df[df['in_tissue'] == 1].reset_index(drop=True)
    df['hires_col'] = df['pxl_col_in_fullres'] * scalef
    df['hires_row'] = df['pxl_row_in_fullres'] * scalef
    print(f'    in_tissue=1 bin: {len(df):,}')

    # ── [4] count matrix 로드
    print(f'\n[4] count matrix 로드')
    adata = sc.read_10x_h5(args.count_h5)
    adata.var_names_make_unique()
    with open(args.gene_list) as f:
        hvg_genes = [line.strip() for line in f.readlines()]
    adata = adata[:, adata.var_names.isin(hvg_genes)].copy()
    print(f'    shape: {adata.shape}')

    barcode_to_idx = {b: i for i, b in enumerate(adata.obs.index)}
    df['count_idx'] = df['barcode'].map(barcode_to_idx)
    df = df.dropna(subset=['count_idx'])
    df['count_idx'] = df['count_idx'].astype(int)

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    gene_names = np.array(list(adata.var_names))

    bin_col = df['hires_col'].values
    bin_row = df['hires_row'].values
    bin_idx = df['count_idx'].values

    # ── [5] 타일 크기 계산
    print(f'\n[5] 타일 크기 계산')
    tile_size = calc_tile_size(args.h5_path, img_width, img_height, args.target_um)
    stride    = tile_size

    # ── [6] 유효 타일 목록 생성 (bin bounding box 범위만 순회)
    print(f'\n[6] 유효 타일 목록 생성 (bin 좌표 기준 bounding box)')

    # bin 좌표 bounding box
    col_min = int(bin_col.min())
    col_max = int(bin_col.max())
    row_min = int(bin_row.min())
    row_max = int(bin_row.max())
    print(f'    bin bounding box: col {col_min}~{col_max}, row {row_min}~{row_max}')

    valid_tiles = []
    for row_start in range(row_min, row_max, stride):
        for col_start in range(col_min, col_max, stride):
            # 이미지 범위 체크
            if row_start + tile_size > img_height or col_start + tile_size > img_width:
                continue
            # 타일 안에 bin이 있는지 확인
            in_tile = (
                (bin_col >= col_start) & (bin_col < col_start + tile_size) &
                (bin_row >= row_start) & (bin_row < row_start + tile_size)
            )
            if in_tile.sum() > 0:
                valid_tiles.append((row_start, col_start))
    print(f'    유효 타일: {len(valid_tiles):,}개')

    # ── [7] 모델 로드
    print(f'\n[7] 모델 로드')
    model, preprocess = load_omiclip(args.pretrained, device)
    print(f'    완료')

    # ── [8] 임베딩 추출 + bin aggregation
    print(f'\n[8] 임베딩 추출 + bin aggregation')

    all_embs   = []
    all_exprs  = []
    all_coords = []

    batch_imgs   = []
    batch_coords = []
    batch_exprs  = []

    for row_start, col_start in tqdm(valid_tiles, desc='처리 중'):

        # 이미지 크롭
        patch = img[row_start:row_start+tile_size, col_start:col_start+tile_size]
        batch_imgs.append(preprocess(Image.fromarray(patch)))
        batch_coords.append((col_start, row_start))

        # bin aggregation
        in_tile = (
            (bin_col >= col_start) & (bin_col < col_start + tile_size) &
            (bin_row >= row_start) & (bin_row < row_start + tile_size)
        )
        tile_bin_idx = bin_idx[in_tile]
        if len(tile_bin_idx) > 0:
            tile_expr = X[tile_bin_idx].sum(axis=0)
        else:
            tile_expr = np.zeros(X.shape[1], dtype=np.float32)
        batch_exprs.append(tile_expr)

        # 배치 처리
        if len(batch_imgs) >= args.batch_size:
            all_embs.append(encode_batch(model, batch_imgs, device))
            all_coords.extend(batch_coords)
            all_exprs.append(np.array(batch_exprs))
            batch_imgs   = []
            batch_coords = []
            batch_exprs  = []

    # 남은 배치
    if len(batch_imgs) > 0:
        all_embs.append(encode_batch(model, batch_imgs, device))
        all_coords.extend(batch_coords)
        all_exprs.append(np.array(batch_exprs))

    # ── [9] 결합
    all_embs   = np.concatenate(all_embs,  axis=0)
    all_exprs  = np.concatenate(all_exprs, axis=0).astype(np.float32)
    all_coords = np.array(all_coords)

    print(f'\n    tile_img_embs: {all_embs.shape}')
    print(f'    tile_exprs:    {all_exprs.shape}')
    print(f'    zeros 비율:    {(all_exprs==0).mean():.4f}')

    # normalize
    print('    normalize_total + log1p 적용')
    agg_adata = AnnData(X=csr_matrix(all_exprs))
    sc.pp.normalize_total(agg_adata)
    sc.pp.log1p(agg_adata)
    all_exprs_norm = agg_adata.X.toarray()

    # ── [10] 저장
    print(f'\n[9] 저장: {args.output_dir}')
    np.save(os.path.join(args.output_dir, 'tile_img_embs.npy'),  all_embs)
    np.save(os.path.join(args.output_dir, 'tile_exprs.npy'),     all_exprs_norm)
    np.save(os.path.join(args.output_dir, 'tile_coords.npy'),    all_coords)
    np.save(os.path.join(args.output_dir, 'tile_gene_names.npy'), gene_names)

    print(f'\n✅ 완료!')
    print(f'   tile_img_embs: {all_embs.shape}')
    print(f'   tile_exprs:    {all_exprs_norm.shape}')
    print(f'   tile_coords:   {all_coords.shape}')