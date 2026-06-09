#!/usr/bin/env python3
"""
encode_visium_hd.py
===================
Visium HD 타일 PNG → 이미지 임베딩 추출

Usage:
    python encode_visium_hd.py \
        --tile_dir /project_antwerp/hbae/data/visium_hd_tonsil/tiles_68um/ \
        --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \
        --output_dir /project_antwerp/hbae/Loki_output/visium_hd_embeddings/ \
        --device cuda:0
"""

import argparse
from pathlib import Path
import numpy as np
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


@torch.no_grad()
def encode_images_batch(model, image_paths, preprocess, device, batch_size=64):
    all_embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(preprocess(img))
            except:
                print(f"  Warning: failed to load {path}, using zeros")
                images.append(torch.zeros(3, 224, 224))
        img_tensor = torch.stack(images).to(device)
        emb = model.encode_image(img_tensor)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def main(args):
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── [1] 타일 PNG 목록 로드
    print(f'\n[1] 타일 목록 로드: {args.tile_dir}')
    tile_dir = Path(args.tile_dir)
    tile_paths = sorted(tile_dir.glob('*.png'))
    tile_names = [p.name for p in tile_paths]
    print(f'    타일 수: {len(tile_paths)}')

    # ── [2] 모델 로드
    print(f'\n[2] 모델 로드: {args.pretrained}')
    model, preprocess = load_omiclip(args.pretrained, device)
    print(f'    모델 로드 완료')

    # ── [3] 이미지 임베딩 추출
    print(f'\n[3] 이미지 임베딩 추출')
    img_embs = encode_images_batch(
        model,
        [str(p) for p in tile_paths],
        preprocess,
        device,
        batch_size=args.batch_size
    )
    print(f'    임베딩 shape: {img_embs.shape}')

    # ── [4] 저장
    print(f'\n[4] 저장: {out}')
    np.save(out / 'visium_hd_img_embs.npy',  img_embs.numpy())
    np.save(out / 'visium_hd_tile_names.npy', np.array(tile_names))

    print(f'\n✅ 완료!')
    print(f'   visium_hd_img_embs.npy:  {img_embs.shape}')
    print(f'   visium_hd_tile_names.npy: {len(tile_names)}개')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tile_dir',   required=True,  help='타일 PNG 폴더')
    p.add_argument('--pretrained', required=True,  help='Loki checkpoint 경로')
    p.add_argument('--output_dir', required=True,  help='임베딩 저장 폴더')
    p.add_argument('--device',     default='cuda:0')
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()
    main(args)