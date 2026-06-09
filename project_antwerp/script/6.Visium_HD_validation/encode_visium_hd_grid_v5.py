#!/usr/bin/env python3
"""
encode_visium_hd_grid_v5.py
============================
2um bin 지원 - sparse matrix로 메모리 절약
X.toarray() 전체 변환 없이 타일별로 sparse 슬라이싱

Usage:
    python encode_visium_hd_grid_v5.py \
        --image_path /project_antwerp/hbae/data/visium_hd_tonsil/spatial/tissue_hires_image.png \
        --positions_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_002um/spatial/tissue_positions.parquet \
        --scalefactors_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_002um/spatial/scalefactors_json.json \
        --count_h5 /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_002um/filtered_feature_bc_matrix.h5 \
        --h5_path /project_antwerp/hbae/data/visium_hd_tonsil/Visium_HD_FF_Human_Tonsil_feature_slice.h5 \
        --gene_list /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
        --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \
        --output_dir /project_antwerp/hbae/Loki_output/visium_hd_23px_2um_embeddings/fold_01/ \
        --device cuda:0
        (base) hyobaeug@b8f4d64a576f:/project_antwerp/hbae/script/0208_start/Visium_HD$ python predict_visium_hd.py \
        --train_emb_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01 \
        --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_23px_2um_embeddings/fold_01 \
        --pred_style loki \
        --top_k 500 \
        --device cuda:0

[1] train 임베딩 로드 (HNSCC fold)
    train_text_embs: torch.Size([64157, 768])
    train_exprs:     torch.Size([64157, 2000])

[2] Visium HD 임베딩 로드
    tile_img_embs: torch.Size([9762, 768])

[3] Visium HD 실제 발현값 로드
    tile_exprs: (9762, 2000)
    zeros 비율: 0.5111

★ Top 300 genes selected (out of 2000 HVG genes)

[4] 예측 시작 (pred_style=loki, top_k=500)
    총 타일 수: 9,762
Predicting: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:06<00:00,  3.15it/s]

[5] Variation 확인
    예측값 std (tile간): 0.1126
    실제값 std (tile간): 0.9139

[6] PCC 계산

============================================================
pred_style:  loki
top_k:       500
eval genes:  top 300 expressed in Visium HD
============================================================
Spot-wise PCC: mean=0.2651, median=0.2614
Gene-wise PCC: mean=0.0234, median=0.0239
============================================================

✅ 결과 저장: /project_antwerp/hbae/Loki_output/visium_hd_23px_2um_embeddings/fold_01/visium_hd_pcc_loki.npy
(base) hyobaeug@b8f4d64a576f:/project_antwerp/hbae/script/0208_start/Visium_HD$ python predict_visium_hd.py \
        --train_emb_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01 \
        --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_23px_2um_embeddings/fold_01 \
        --pred_style loki \
        --device cuda:0

[1] train 임베딩 로드 (HNSCC fold)
    train_text_embs: torch.Size([64157, 768])
    train_exprs:     torch.Size([64157, 2000])

[2] Visium HD 임베딩 로드
    tile_img_embs: torch.Size([9762, 768])

[3] Visium HD 실제 발현값 로드
    tile_exprs: (9762, 2000)
    zeros 비율: 0.5111

★ Top 300 genes selected (out of 2000 HVG genes)

[4] 예측 시작 (pred_style=loki, top_k=None)
    총 타일 수: 9,762
Predicting: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:15<00:00,  1.33it/s]

[5] Variation 확인
    예측값 std (tile간): 0.0109
    실제값 std (tile간): 0.9139

[6] PCC 계산

============================================================
pred_style:  loki
top_k:       None
eval genes:  top 300 expressed in Visium HD
============================================================
Spot-wise PCC: mean=0.2675, median=0.2530
Gene-wise PCC: mean=0.0116, median=0.0088
============================================================

✅ 결과 저장: /project_antwerp/hbae/Loki_output/visium_hd_23px_2um_embeddings/fold_01/visium_hd_pcc_loki.npy
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


def calc_tile_size(scalefactors_path, target_um=68.0):
    with open(scalefactors_path) as f:
        sf = json.load(f)
    microns_per_pixel   = sf['microns_per_pixel']
    tissue_hires_scalef = sf['tissue_hires_scalef']
    um_per_px_hires     = microns_per_pixel / tissue_hires_scalef
    tile_size           = int(round(target_um / um_per_px_hires))
    print(f'  microns_per_pixel:   {microns_per_pixel:.6f} um/px_fullres')
    print(f'  tissue_hires_scalef: {tissue_hires_scalef:.8f}')
    print(f'  um/px_hires:         {um_per_px_hires:.4f} um/px_hires')
    print(f'  {target_um}um → tile_size: {tile_size}px')
    print(f'  검증: {tile_size}px × {um_per_px_hires:.4f} = {tile_size * um_per_px_hires:.2f}um')
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── [1] 이미지 로드
    print(f'\n[1] 이미지 로드')
    img = np.array(Image.open(args.image_path).convert('RGB'))
    img_height, img_width = img.shape[:2]
    print(f'    이미지 크기: {img_width} x {img_height} px (hires)')

    # ── [2] scalefactors + 타일 크기 계산
    print(f'\n[2] scalefactors 로드 + 타일 크기 계산')
    with open(args.scalefactors_path) as f:
        scalefactors = json.load(f)
    scalef    = scalefactors['tissue_hires_scalef']
    tile_size = calc_tile_size(args.scalefactors_path, args.target_um)
    stride    = tile_size

    # ── [3] tissue_positions 로드
    print(f'\n[3] tissue_positions 로드')
    df = pd.read_parquet(args.positions_path, engine='pyarrow')
    df = df[df['in_tissue'] == 1].reset_index(drop=True)
    df['hires_col'] = df['pxl_col_in_fullres'] * scalef
    df['hires_row'] = df['pxl_row_in_fullres'] * scalef
    print(f'    in_tissue=1 bin: {len(df):,}')

    # ── [4] count matrix 로드 (★ sparse 유지, toarray() 안 함)
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

    # ★ sparse matrix 그대로 유지 (csr로 변환만)
    X_sparse = adata.X.tocsr() if hasattr(adata.X, 'tocsr') else csr_matrix(adata.X)
    print(f'    sparse format: {X_sparse.format}, nnz: {X_sparse.nnz:,}')

    barcode_to_idx = {b: i for i, b in enumerate(adata.obs.index)}
    df['count_idx'] = df['barcode'].map(barcode_to_idx)
    df = df.dropna(subset=['count_idx'])
    df['count_idx'] = df['count_idx'].astype(int)

    bin_col = df['hires_col'].values
    bin_row = df['hires_row'].values
    bin_idx = df['count_idx'].values

    # ── [5] 유효 타일 목록 생성
    print(f'\n[5] 유효 타일 목록 생성')
    col_min = int(bin_col.min())
    col_max = int(bin_col.max())
    row_min = int(bin_row.min())
    row_max = int(bin_row.max())
    print(f'    bin bounding box: col {col_min}~{col_max}, row {row_min}~{row_max}')

    valid_tiles = []
    for row_start in range(row_min, row_max, stride):
        for col_start in range(col_min, col_max, stride):
            if row_start + tile_size > img_height or col_start + tile_size > img_width:
                continue
            in_tile = (
                (bin_col >= col_start) & (bin_col < col_start + tile_size) &
                (bin_row >= row_start) & (bin_row < row_start + tile_size)
            )
            if in_tile.sum() > 0:
                valid_tiles.append((row_start, col_start))
    print(f'    유효 타일: {len(valid_tiles):,}개')

    # ── [6] 모델 로드
    print(f'\n[6] 모델 로드')
    model, preprocess = load_omiclip(args.pretrained, device)
    print(f'    완료')

    # ── [7] 임베딩 추출 + bin aggregation (sparse 슬라이싱)
    print(f'\n[7] 임베딩 추출 + bin aggregation (sparse)')
    print(f'    방식: sparse 슬라이싱 → raw count 합산 → tile별 normalize_total + log1p')

    all_embs   = []
    all_exprs  = []
    all_coords = []
    batch_imgs   = []
    batch_coords = []
    batch_exprs  = []

    for row_start, col_start in tqdm(valid_tiles, desc='처리 중'):

        patch = img[row_start:row_start+tile_size, col_start:col_start+tile_size]
        batch_imgs.append(preprocess(Image.fromarray(patch)))
        batch_coords.append((col_start, row_start))

        # ★ sparse 슬라이싱 (dense 변환 없음)
        in_tile = (
            (bin_col >= col_start) & (bin_col < col_start + tile_size) &
            (bin_row >= row_start) & (bin_row < row_start + tile_size)
        )
        tile_bin_idx = bin_idx[in_tile]
        if len(tile_bin_idx) > 0:
            # sparse 행 슬라이싱 후 합산
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

    print(f'\n    tile_img_embs: {all_embs.shape}')
    print(f'    tile_exprs (normalize 전): {all_exprs.shape}')
    print(f'    zeros 비율 (normalize 전): {(all_exprs==0).mean():.4f}')

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