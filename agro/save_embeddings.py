#!/usr/bin/env python3
"""
save_embeddings.py
==================
fold당 한 번만 실행 - embedding 저장
이후 predict_fast.py로 빠르게 실험 가능

Usage:
    python save_embeddings.py \
        --train_csv fold_01_train.csv \
        --val_csv fold_01_val.csv \
        --hvg_file HVG_genelist.txt \
        --pretrained /path/to/checkpoint.pt \
        --output_dir /path/to/embeddings/fold_01 \
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
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, preprocess = create_model_from_pretrained('coca_ViT-L-14', pretrained=checkpoint_path, device=device)
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
                images.append(torch.zeros(3, 224, 224))
        img_tensor = torch.stack(images).to(device)
        emb = model.encode_image(img_tensor)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_gene_sentences(model, gene_lists, tokenizer, device, batch_size=256):
    all_embs = []
    for i in tqdm(range(0, len(gene_lists), batch_size), desc="Encoding genes"):
        batch = gene_lists[i:i+batch_size]
        sentences = [' '.join(genes) for genes in batch]
        tokens = tokenizer(sentences).to(device)
        emb = model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def main(args):
    import pandas as pd
    import open_clip

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print("[1] Loading OmiCLIP...")
    model, preprocess = load_omiclip(args.pretrained, device)
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

    # 2. Load GT
    print("[2] Loading GT...")
    gt_expr = np.load(args.gt_expr)
    gt_obs  = np.load(args.gt_obs, allow_pickle=True)
    all_genes = open(args.gene_list).read().strip().split('\n')
    obs_to_idx = {b: i for i, b in enumerate(gt_obs)}
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    hvg_genes = open(args.hvg_file).read().strip().split('\n')
    hvg_indices = [gene_to_idx[g] for g in hvg_genes if g in gene_to_idx]
    print(f"  HVG: {len(hvg_indices)}")

    # 3. Load CSVs
    def extract_obs_key(filepath):
        parts = filepath.split('/')
        gsm = [p for p in parts if p.startswith('GSM') or (len(p) == 7 and p[0].isdigit())]
        sample_id = gsm[0].split('_')[0] if gsm else parts[6].split('_')[0]
        barcode = parts[-1].replace('.png', '')
        return f"{sample_id}_{barcode}_hires"

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    if 'obs_key' not in train_df.columns:
        train_df['obs_key'] = train_df['filepath'].apply(extract_obs_key)
    if 'obs_key' not in val_df.columns:
        val_df['obs_key']   = val_df['filepath'].apply(extract_obs_key)

    train_df = train_df[train_df['obs_key'].isin(obs_to_idx)].reset_index(drop=True)
    val_df   = val_df[val_df['obs_key'].isin(obs_to_idx)].reset_index(drop=True)
    print(f"  Train: {len(train_df):,} / Val: {len(val_df):,}")

    # 4. Encode train
    print("[3] Encoding train...")
    train_img_embs = encode_images_batch(model, train_df['filepath'].tolist(), preprocess, device)

    train_gene_lists, train_exprs_list = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Preparing train"):
        train_gene_lists.append(row['title'].split())
        spot_idx = obs_to_idx[row['obs_key']]
        train_exprs_list.append(gt_expr[spot_idx, hvg_indices])

    train_text_embs = encode_gene_sentences(model, train_gene_lists, tokenizer, device)
    train_exprs = np.array(train_exprs_list)

    # 5. Encode val
    print("[4] Encoding val...")
    val_img_embs = encode_images_batch(model, val_df['filepath'].tolist(), preprocess, device)

    val_exprs_list = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Preparing val"):
        spot_idx = obs_to_idx[row['obs_key']]
        val_exprs_list.append(gt_expr[spot_idx, hvg_indices])
    val_exprs = np.array(val_exprs_list)

    # 6. Save
    print("[5] Saving embeddings...")
    np.save(out / 'train_img_embs.npy',  train_img_embs.numpy())
    np.save(out / 'train_text_embs.npy', train_text_embs.numpy())
    np.save(out / 'train_exprs.npy',     train_exprs)
    np.save(out / 'val_img_embs.npy',    val_img_embs.numpy())
    np.save(out / 'val_exprs.npy',       val_exprs)

    print(f"\n✅ Saved to {out}")
    print(f"  train_img_embs:  {train_img_embs.shape}")
    print(f"  train_text_embs: {train_text_embs.shape}")
    print(f"  train_exprs:     {train_exprs.shape}")
    print(f"  val_img_embs:    {val_img_embs.shape}")
    print(f"  val_exprs:       {val_exprs.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--val_csv",    required=True)
    p.add_argument("--hvg_file",   required=True)
    p.add_argument("--gt_expr",    default="/project_antwerp/hbae/data/combined_expression_matrix.npy")
    p.add_argument("--gt_obs",     default="/project_antwerp/hbae/data/combined_obs.npy")
    p.add_argument("--gene_list",  default="/project_antwerp/hbae/data/all_shared_genes.txt")
    p.add_argument("--pretrained", default="/project_antwerp/assets/loki_ckpts/checkpoint.pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device",     default="cuda:0")
    args = p.parse_args()
    main(args)