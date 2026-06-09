"""
Open-ST HNSCC External Validation - Step 2: Embedding Extraction
=================================================================
기존 save_embeddings.py 와 동일한 모델 로딩 방식 사용
(open_clip.create_model_and_transforms + model_state_dict 분기)

Usage:
    python openst_step2_embed.py \
        --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \
        --output_dir /project_antwerp/hbae/Loki_output/openst_validation/fold_01 \
        --device cuda:0

zero-shot:
    python openst_step2_embed.py \
        --pretrained /project_antwerp/assets/loki_ckpts/checkpoint.pt \
        --output_dir /project_antwerp/hbae/Loki_output/openst_validation/zeroshot \
        --device cuda:0
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import h5py
import open_clip


def load_omiclip(checkpoint_path, device):
    """기존 save_embeddings.py 와 동일한 로딩 방식"""
    model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', pretrained=None)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']   # 파인튜닝 체크포인트
    else:
        state_dict = ckpt.get('state_dict', ckpt)  # 원본 pretrained

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def encode_images_from_h5(model, preprocess, h5_path, device, batch_size=64):
    """HDF5에서 patch를 읽어 image embedding 추출 (기존 encode_images_batch와 동일 방식)"""
    with h5py.File(h5_path, 'r') as f:
        n_cells = f.attrs['n_cells']

    all_embs = []
    for i in tqdm(range(0, n_cells, batch_size), desc="Encoding images"):
        end = min(i + batch_size, n_cells)
        with h5py.File(h5_path, 'r') as f:
            patches = f['patches'][i:end]   # (B, 224, 224, 3) uint8

        images = []
        for patch in patches:
            img = Image.fromarray(patch).convert('RGB')
            images.append(preprocess(img))

        img_tensor = torch.stack(images).to(device)
        emb = model.encode_image(img_tensor)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)   # (N, D)


def main(args):
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print("[1] Loading OmiCLIP ...")
    model, preprocess = load_omiclip(args.pretrained, device)
    print(f"  Checkpoint: {args.pretrained}")

    # 2. Encode Open-ST patches
    print("[2] Encoding Open-ST patches ...")
    embs = encode_images_from_h5(model, preprocess, args.h5_path, device, args.batch_size)
    print(f"  Embeddings shape: {embs.shape}")

    # 3. Save
    out_path = out / 'openst_img_embs.npy'
    np.save(out_path, embs.numpy())
    print(f"\n✅ Saved to {out_path}")
    print(f"  openst_img_embs: {embs.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path",    default="/project_antwerp/hbae/data/Open_ST/openst_patches.h5")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    main(args)