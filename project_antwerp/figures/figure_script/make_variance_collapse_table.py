#!/usr/bin/env python3
"""
make_variance_collapse_table.py
================================
TCGA variance collapse 증거 table 생성
- 랜덤 샘플링된 10개 슬라이드의 top-5 predicted genes 테이블
- PNG 저장

Usage:
    python make_variance_collapse_table.py \
        --emb_dir   /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --tcga_dir  /project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03 \
        --gene_list /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
        --out_dir   /project_antwerp/hbae/figures
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

def predex_slide(tile_embs, train_emb, train_exprs):
    sims = tile_embs @ train_emb.T
    s = sims.sum(dim=1, keepdim=True)
    w = sims / s.clamp(min=1e-12)
    tile_preds = w @ train_exprs
    return tile_preds.mean(dim=0)

def main(args):
    emb_dir  = Path(args.emb_dir)
    tcga_dir = Path(args.tcga_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train embeddings...")
    train_text  = torch.tensor(np.load(emb_dir / 'train_text_embs.npy')).float()
    train_exprs = torch.tensor(np.load(emb_dir / 'train_exprs.npy')).float()
    train_norm  = F.normalize(train_text, dim=-1)

    with open(args.gene_list) as f:
        gene_names = [l.strip() for l in f if l.strip()]

    # Get non-coords npy files
    slide_files = [f for f in sorted(tcga_dir.glob('*.npy'))
                   if '_coords' not in f.name]
    random.seed(42)
    selected = random.sample(slide_files, min(10, len(slide_files)))
    selected = sorted(selected, key=lambda x: x.stem)

    rows = []
    for sf in selected:
        try:
            tile_embs = torch.tensor(np.load(sf)).float()
            if tile_embs.ndim == 1:
                tile_embs = tile_embs.unsqueeze(0)
            tile_norm = F.normalize(tile_embs, dim=-1)
            pred = predex_slide(tile_norm, train_norm, train_exprs)
            top5_idx = np.argsort(pred.numpy())[::-1][:5]
            top5 = [gene_names[i] for i in top5_idx]
            # shorten slide ID
            sid = sf.stem
            if len(sid) > 22:
                sid = sid[:19] + '...'
            rows.append([sid] + top5)
        except Exception as e:
            print(f"[SKIP] {sf.stem}: {e}")

    # Training mean top-5
    train_mean = train_exprs.mean(dim=0).numpy()
    top5_train = [gene_names[i] for i in np.argsort(train_mean)[::-1][:5]]
    rows.append(['ST train mean'] + top5_train)

    # ── Figure: table ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 0.45 * (len(rows) + 2)))
    ax.axis('off')

    col_labels = ['Slide ID', 'Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5']
    cell_colors = []
    for i, row in enumerate(rows):
        if i == len(rows) - 1:
            # training mean row - light yellow
            cell_colors.append(['#FFF9C4'] * 6)
        else:
            cell_colors.append(['#FFFFFF'] * 6)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Header styling
    for j in range(6):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Last row (train mean) - bold
    for j in range(6):
        table[len(rows), j].set_text_props(fontweight='bold')

    fig.suptitle(
        'Top-5 Predicted Genes per TCGA Slide\n'
        '(PredEx variance collapse: all slides predict identical genes)',
        fontsize=10, fontweight='bold', y=0.98
    )

    out_path = out_dir / 'tcga_variance_collapse_table.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'\n✅ Saved: {out_path}')
    print(f'\nAll rows:')
    for row in rows:
        print(f'  {row}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb_dir',   required=True)
    ap.add_argument('--tcga_dir',  required=True)
    ap.add_argument('--gene_list', required=True)
    ap.add_argument('--out_dir',   required=True)
    args = ap.parse_args()
    main(args)