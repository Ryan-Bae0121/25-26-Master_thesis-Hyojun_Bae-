#!/usr/bin/env python3
"""
generate_embeddings_fixed.py
============================
Fixed version that works with your CSV structure

CSV format:
filepath,title
/path/to/BARCODE.png,GENE1 GENE2 GENE3

Extracts barcode from filename instead of using obs_key column
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


def load_omiclip(checkpoint_path, device):
    """Load pretrained OmiCLIP"""
    model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', pretrained=None)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def encode_images_batch(model, img_paths, preprocess, device, batch_size=32):
    """Encode images in batches"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Encoding images"):
        batch_paths = img_paths[i:i+batch_size]
        
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = preprocess(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                images.append(torch.zeros(3, 224, 224))
        
        batch = torch.stack(images).to(device)
        embeddings = model.encode_image(batch)
        
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)


def extract_barcode_from_path(filepath):
    """Extract barcode from filepath
    
    Example:
    /path/to/AAACACCAATAACTGC-1.png -> AAACACCAATAACTGC-1
    """
    filename = Path(filepath).stem  # Remove .png
    return filename


def main(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Generate Embeddings (Fixed Version)")
    print("="*70)
    
    # Load model
    print("\n[1] Loading OmiCLIP model...")
    model, preprocess = load_omiclip(args.pretrained, device)
    print("  ✓ Model loaded")
    
    # Load CSVs
    print("\n[2] Loading data splits...")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    print(f"  Train: {len(train_df):,} spots")
    print(f"  Val:   {len(val_df):,} spots")
    
    # Load HVG
    with open(args.hvg_file) as f:
        hvg_genes = [line.strip() for line in f]
    print(f"  HVG: {len(hvg_genes)} genes")
    
    # Load ground truth
    print("\n[3] Loading ground truth expression...")
    gt_expr = np.load(args.gt_expr, mmap_mode='r')
    gt_obs = np.load(args.gt_obs, allow_pickle=True)
    
    with open(args.gene_list) as f:
        all_genes = [line.strip() for line in f]
    
    # Create mappings
    obs_to_idx = {obs: i for i, obs in enumerate(gt_obs)}
    hvg_indices = [all_genes.index(g) for g in hvg_genes]
    
    print(f"  Expression matrix: {gt_expr.shape}")
    print(f"  HVG indices: {len(hvg_indices)}")
    
    # Encode train images
    print("\n[4] Encoding train images...")
    train_img_paths = train_df['filepath'].tolist()
    train_img_embs = encode_images_batch(model, train_img_paths, preprocess, device)
    
    print("  ✓ Saving train_img_embs.npy...")
    np.save(output_dir / 'train_img_embs.npy', train_img_embs.numpy())
    
    # Prepare train expressions
    print("\n[5] Preparing train expressions...")
    train_exprs = []
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
        # Extract barcode from filepath
        barcode = extract_barcode_from_path(row['filepath'])
        
        if barcode in obs_to_idx:
            spot_idx = obs_to_idx[barcode]
            expr = gt_expr[spot_idx, hvg_indices]
            train_exprs.append(expr)
        else:
            print(f"  Warning: {barcode} not found, using zeros")
            train_exprs.append(np.zeros(len(hvg_genes)))
    
    train_exprs = np.array(train_exprs)
    print("  ✓ Saving train_exprs.npy...")
    np.save(output_dir / 'train_exprs.npy', train_exprs)
    
    # Encode val images
    print("\n[6] Encoding val images...")
    val_img_paths = val_df['filepath'].tolist()
    val_img_embs = encode_images_batch(model, val_img_paths, preprocess, device)
    
    print("  ✓ Saving val_img_embs.npy...")
    np.save(output_dir / 'val_img_embs.npy', val_img_embs.numpy())
    
    # Prepare val expressions
    print("\n[7] Preparing val expressions...")
    val_exprs = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val"):
        barcode = extract_barcode_from_path(row['filepath'])
        
        if barcode in obs_to_idx:
            spot_idx = obs_to_idx[barcode]
            expr = gt_expr[spot_idx, hvg_indices]
            val_exprs.append(expr)
        else:
            print(f"  Warning: {barcode} not found, using zeros")
            val_exprs.append(np.zeros(len(hvg_genes)))
    
    val_exprs = np.array(val_exprs)
    print("  ✓ Saving val_exprs.npy...")
    np.save(output_dir / 'val_exprs.npy', val_exprs)
    
    print("\n" + "="*70)
    print("Embeddings Generated Successfully!")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - train_img_embs.npy  {train_img_embs.shape}")
    print(f"  - train_exprs.npy     {train_exprs.shape}")
    print(f"  - val_img_embs.npy    {val_img_embs.shape}")
    print(f"  - val_exprs.npy       {val_exprs.shape}")
    print("="*70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--hvg_file", required=True)
    p.add_argument("--gt_expr", 
                   default="/project_antwerp/hbae/data/combined_expression_matrix.npy")
    p.add_argument("--gt_obs", 
                   default="/project_antwerp/hbae/data/combined_obs.npy")
    p.add_argument("--gene_list", 
                   default="/project_antwerp/hbae/data/all_shared_genes.txt")
    p.add_argument("--pretrained", 
                   default="/project_antwerp/assets/loki_ckpts/checkpoint.pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    
    args = p.parse_args()
    main(args)