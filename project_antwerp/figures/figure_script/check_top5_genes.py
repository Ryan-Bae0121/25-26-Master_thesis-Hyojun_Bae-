#!/usr/bin/env python3
"""
check_top5_genes.py
===================
TCGA slides에서 variance collapse 확인:
- 모든 슬라이드의 top-5 predicted genes가 동일한지 확인
- 실제 top-5 gene 이름 출력

Usage:
    python check_top5_genes.py \
        --emb_dir  /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --tcga_dir /project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings \
        --gene_list /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
        --n_slides 20
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
import os

def predex_slide(tile_embs, train_emb, train_exprs):
    """PredEx: weighted average over all train spots for each tile, then average tiles."""
    # tile_embs: (n_tiles, dim)
    # returns: (n_genes,) predicted expression for this slide
    sims = tile_embs @ train_emb.T  # (n_tiles, n_train)
    # linear normalization per tile
    s = sims.sum(dim=1, keepdim=True)  # (n_tiles, 1)
    w = sims / s.clamp(min=1e-12)      # (n_tiles, n_train)
    tile_preds = w @ train_exprs       # (n_tiles, n_genes)
    return tile_preds.mean(dim=0)      # (n_genes,)


def main(args):
    emb_dir   = Path(args.emb_dir)
    tcga_dir  = Path(args.tcga_dir)
    gene_list = Path(args.gene_list)

    print("Loading train embeddings...")
    train_text  = torch.tensor(np.load(emb_dir / 'train_text_embs.npy')).float()
    train_exprs = torch.tensor(np.load(emb_dir / 'train_exprs.npy')).float()
    train_norm  = F.normalize(train_text, dim=-1)

    print(f"  train spots: {train_text.shape[0]}")
    print(f"  n_genes:     {train_exprs.shape[1]}")

    # Load gene names
    with open(gene_list) as f:
        gene_names = [line.strip() for line in f if line.strip()]
    print(f"  gene names loaded: {len(gene_names)}")

    # Find TCGA tile embedding files
    # Assume structure: tcga_dir/embeddings/SLIDE_ID.npy  or similar
    # Try common paths
    possible_emb_dirs = [
        tcga_dir / 'embeddings',
        tcga_dir / 'tile_embeddings',
        tcga_dir / 'img_embs',
        Path('/project_antwerp/hbae/Loki_output') / 'tcga_embeddings',
        Path('/project_antwerp/hbae') / 'tcga_tile_embs',
    ]

    emb_search_dir = None
    for d in possible_emb_dirs:
        if d.exists():
            files = list(d.glob('*.npy'))
            if files:
                emb_search_dir = d
                print(f"\nFound TCGA embeddings at: {d}")
                print(f"  {len(files)} .npy files")
                break

    if emb_search_dir is None:
        # Try to find any npy files in tcga_dir recursively
        print(f"\nSearching for .npy files in {tcga_dir}...")
        npy_files = list(tcga_dir.rglob('*.npy'))[:5]
        print(f"  First 5 found: {[str(f) for f in npy_files]}")
        if npy_files:
            emb_search_dir = npy_files[0].parent
            print(f"  Using: {emb_search_dir}")
        else:
            print("No .npy files found. Please check the path.")
            print("\nFalling back: checking variance collapse via ST train mean...")
            # Show what the training mean top-5 genes are
            train_mean = train_exprs.mean(dim=0).numpy()
            top5_idx = np.argsort(train_mean)[::-1][:10]
            print("\nST training set mean expression - TOP 10 genes:")
            for i, idx in enumerate(top5_idx):
                gname = gene_names[idx] if idx < len(gene_names) else f"gene_{idx}"
                print(f"  {i+1}. {gname}: {train_mean[idx]:.4f}")
            return

    # Load and process slides
    slide_files = sorted(emb_search_dir.glob('*.npy'))[:args.n_slides]
    print(f"\nProcessing {len(slide_files)} slides...")

    all_top5 = []
    slide_top5 = {}

    for sf in slide_files:
        slide_id = sf.stem
        try:
            tile_embs = torch.tensor(np.load(sf)).float()
            if tile_embs.ndim == 1:
                tile_embs = tile_embs.unsqueeze(0)
            tile_norm = F.normalize(tile_embs, dim=-1)

            pred = predex_slide(tile_norm, train_norm, train_exprs)
            pred_np = pred.numpy()

            top5_idx = np.argsort(pred_np)[::-1][:5]
            top5_genes = [gene_names[i] if i < len(gene_names) else f"gene_{i}"
                         for i in top5_idx]
            all_top5.append(tuple(top5_genes))
            slide_top5[slide_id] = top5_genes

        except Exception as e:
            print(f"  [SKIP] {slide_id}: {e}")

    # Analysis
    print(f"\n{'='*60}")
    print(f"VARIANCE COLLAPSE ANALYSIS")
    print(f"{'='*60}")
    print(f"Slides processed: {len(all_top5)}")

    counter = Counter(all_top5)
    print(f"\nTop-5 gene combinations (unique): {len(counter)}")
    print(f"Most common combination (n={counter.most_common(1)[0][1]}):")
    print(f"  {list(counter.most_common(1)[0][0])}")

    if len(counter) == 1:
        print("\n✅ CONFIRMED: All slides have identical top-5 predicted genes")
        print("   → Variance collapse confirmed")
    else:
        print(f"\n⚠️  {len(counter)} unique combinations found")
        print("Top 5 most common:")
        for combo, cnt in counter.most_common(5):
            print(f"  {list(combo)}: {cnt} slides ({cnt/len(all_top5)*100:.1f}%)")

    # Show first few slides
    print(f"\nPer-slide top-5 (first 5 slides):")
    for i, (sid, genes) in enumerate(list(slide_top5.items())[:5]):
        print(f"  {sid}: {genes}")

    # Also show training set mean top-5 for comparison
    train_mean = train_exprs.mean(dim=0).numpy()
    top5_train_idx = np.argsort(train_mean)[::-1][:5]
    top5_train = [gene_names[i] if i < len(gene_names) else f"gene_{i}"
                  for i in top5_train_idx]
    print(f"\nST training set mean top-5 genes:")
    print(f"  {top5_train}")
    print(f"\nAre predicted top-5 == training mean top-5?",
          list(all_top5[0]) == top5_train if all_top5 else "N/A")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir",   required=True,
                    help="fold embedding dir (train_text_embs.npy etc)")
    ap.add_argument("--tcga_dir",  required=True,
                    help="TCGA dir containing tile embeddings")
    ap.add_argument("--gene_list", required=True,
                    help="HVG gene list txt file")
    ap.add_argument("--n_slides",  type=int, default=20,
                    help="Number of slides to check (default 20)")
    args = ap.parse_args()
    main(args)