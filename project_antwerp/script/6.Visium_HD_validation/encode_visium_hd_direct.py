#!/usr/bin/env python3
"""
encode_visium_hd_direct.py
==========================
Visium HD bin 좌표 기반 타일링 + 임베딩 추출 (PNG 저장 없이 바로 임베딩)
# 저장된 타일 삭제
rm -rf /project_antwerp/hbae/data/visium_hd_tonsil/tiles_bin_008um/

# 실행
python encode_visium_hd_direct.py \
    --image_path /project_antwerp/hbae/data/visium_hd_tonsil/spatial/tissue_hires_image.png \
    --positions_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/tissue_positions.parquet \
    --scalefactors_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/scalefactors_json.json \
    --h5_path /project_antwerp/hbae/data/visium_hd_tonsil/Visium_HD_FF_Human_Tonsil_feature_slice.h5 \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \ 
    --output_dir /project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/ \
    --device cuda:0
```

완료되면 저장 용량:
```
70만 × 768차원 float32 ≈ 2GB  ← PNG 105GB 대비 훨씬 작음!
Usage:
    python encode_visium_hd_direct.py \
        --image_path /project_antwerp/hbae/data/visium_hd_tonsil/spatial/tissue_hires_image.png \
        --positions_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/tissue_positions.parquet \
        --scalefactors_path /project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/scalefactors_json.json \
        --h5_path /project_antwerp/hbae/data/visium_hd_tonsil/Visium_HD_FF_Human_Tonsil_feature_slice.h5 \
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
from PIL import Image
from tqdm import tqdm


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',        type=str, required=True)
    parser.add_argument('--positions_path',    type=str, required=True)
    parser.add_argument('--scalefactors_path', type=str, required=True)
    parser.add_argument('--h5_path',           type=str, required=True)
    parser.add_argument('--pretrained',        type=str, required=True)
    parser.add_argument('--output_dir',        type=str, required=True)
    parser.add_argument('--target_um',         type=float, default=68.0)
    parser.add_argument('--device',            type=str,  default='cuda:0')
    parser.add_argument('--batch_size',        type=int,  default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

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

    # ── [4] hires 픽셀 좌표 변환
    hires_col = (df['pxl_col_in_fullres'].values * scalef)
    hires_row = (df['pxl_row_in_fullres'].values * scalef)
    barcodes  = df['barcode'].values

    # ── [5] 타일 크기 계산
    print(f'\n[4] 타일 크기 계산 (target: {args.target_um}µm)')
    tile_size = calc_tile_size(args.h5_path, img_width, img_height, args.target_um)
    half = tile_size // 2

    # ── [6] 경계 밖 bin 필터링
    x1s = np.round(hires_col - half).astype(int)
    y1s = np.round(hires_row - half).astype(int)
    x2s = x1s + tile_size
    y2s = y1s + tile_size

    valid    = (x1s >= 0) & (y1s >= 0) & (x2s <= img_width) & (y2s <= img_height)
    x1s      = x1s[valid]
    y1s      = y1s[valid]
    x2s      = x2s[valid]
    y2s      = y2s[valid]
    barcodes = barcodes[valid]
    print(f'    유효 bin: {len(barcodes):,} (경계밖 스킵: {(~valid).sum():,})')

    # ── [7] 모델 로드
    print(f'\n[5] 모델 로드: {args.pretrained}')
    model, preprocess = load_omiclip(args.pretrained, device)
    print(f'    모델 로드 완료')

    # ── [8] 배치 단위로 크롭 + 임베딩 추출
    print(f'\n[6] 임베딩 추출 시작 (batch_size={args.batch_size})')

    all_embs     = []
    valid_barcodes = []
    n = len(barcodes)

    for start in tqdm(range(0, n, args.batch_size), desc='임베딩 추출'):
        end = min(start + args.batch_size, n)

        # 배치 크롭
        batch_imgs = []
        batch_barcodes = []
        for i in range(start, end):
            patch = img[y1s[i]:y2s[i], x1s[i]:x2s[i]]
            pil_patch = Image.fromarray(patch)
            batch_imgs.append(preprocess(pil_patch))
            batch_barcodes.append(barcodes[i])

        # 임베딩 추출
        img_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            emb = model.encode_image(img_tensor)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = F.normalize(emb, dim=-1)

        all_embs.append(emb.cpu().numpy())
        valid_barcodes.extend(batch_barcodes)

    # ── [9] 저장
    all_embs = np.concatenate(all_embs, axis=0)
    valid_barcodes = np.array(valid_barcodes)

    print(f'\n[7] 저장: {args.output_dir}')
    np.save(os.path.join(args.output_dir, 'visium_hd_img_embs.npy'),  all_embs)
    np.save(os.path.join(args.output_dir, 'visium_hd_barcodes.npy'),  valid_barcodes)

    print(f'\n✅ 완료!')
    print(f'   visium_hd_img_embs.npy:  {all_embs.shape}')
    print(f'   visium_hd_barcodes.npy:  {len(valid_barcodes):,}개')
    print(f'   용량: {all_embs.nbytes / 1e9:.2f} GB')