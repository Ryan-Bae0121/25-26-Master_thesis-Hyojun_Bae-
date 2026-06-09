#!/usr/bin/env python3
"""
crop_visium_hd_bin.py
=====================
Visium HD bin 좌표 기반 타일링 스크립트 (벡터화 버전)

Usage:
    python crop_visium_hd_bin.py \
        --image_path /project_antwerp/hbae/data/visium_hd_tonsil/spatial/tissue_hires_image.png \
        --positions_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/tissue_positions.parquet \
        --scalefactors_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/scalefactors_json.json \
        --h5_path /project_antwerp/hbae/data/visium_hd_tonsil/Visium_HD_FF_Human_Tonsil_feature_slice.h5 \
        --dest_path /project_antwerp/hbae/data/visium_hd_tonsil/tiles_bin_008um/
"""

import os
import argparse
import numpy as np
import json
import h5py
import pandas as pd
from PIL import Image
from tqdm import tqdm


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
    print(f'  {target_um}µm → tile_px_x: {tile_px_x:.1f}px, tile_px_y: {tile_px_y:.1f}px')
    print(f'  → 최종 tile_size: {tile_size}×{tile_size}px')
    return tile_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',        type=str, required=True)
    parser.add_argument('--positions_path',    type=str, required=True)
    parser.add_argument('--scalefactors_path', type=str, required=True)
    parser.add_argument('--h5_path',           type=str, required=True)
    parser.add_argument('--dest_path',         type=str, required=True)
    parser.add_argument('--target_um',         type=float, default=68.0)
    parser.add_argument('--output_size',       type=int,   default=224)
    args = parser.parse_args()

    os.makedirs(args.dest_path, exist_ok=True)

    # ── [1] 이미지 로드
    print(f'\n[1] 이미지 로드: {args.image_path}')
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
    print(f'    in_tissue=1 bin: {len(df):,}')

    # ── [4] hires 픽셀 좌표 변환 (벡터 연산)
    hires_col = (df['pxl_col_in_fullres'].values * scalef)
    hires_row = (df['pxl_row_in_fullres'].values * scalef)
    barcodes  = df['barcode'].values
    print(f'    hires_col: {hires_col.min():.1f}~{hires_col.max():.1f}')
    print(f'    hires_row: {hires_row.min():.1f}~{hires_row.max():.1f}')

    # ── [5] 타일 크기 계산
    print(f'\n[4] 타일 크기 계산 (target: {args.target_um}µm)')
    tile_size = calc_tile_size(args.h5_path, img_width, img_height, args.target_um)
    half = tile_size // 2

    # ── [6] 경계 밖 bin 미리 필터링 (벡터 연산)
    x1s = np.round(hires_col - half).astype(int)
    y1s = np.round(hires_row - half).astype(int)
    x2s = x1s + tile_size
    y2s = y1s + tile_size

    valid = (x1s >= 0) & (y1s >= 0) & (x2s <= img_width) & (y2s <= img_height)
    print(f'\n[5] 유효 bin: {valid.sum():,} / {len(valid):,} (경계밖 스킵: {(~valid).sum():,})')

    x1s     = x1s[valid]
    y1s     = y1s[valid]
    x2s     = x2s[valid]
    y2s     = y2s[valid]
    barcodes = barcodes[valid]

    # ── [7] 크롭 및 저장
    print(f'\n[6] 타일링 시작')
    print(f'    crop: {tile_size}×{tile_size}px → resize: {args.output_size}×{args.output_size}px')

    out_size = (args.output_size, args.output_size)
    saved = 0

    for i in tqdm(range(len(barcodes)), desc='크롭 중'):
        patch = img[y1s[i]:y2s[i], x1s[i]:x2s[i]]
        patch_resized = np.array(
            Image.fromarray(patch).resize(out_size, Image.BILINEAR)
        )
        Image.fromarray(patch_resized).save(
            os.path.join(args.dest_path, f'{barcodes[i]}.png')
        )
        saved += 1

    print(f'\n[완료] 저장: {saved:,}개')
    print(f'출력 경로: {args.dest_path}')