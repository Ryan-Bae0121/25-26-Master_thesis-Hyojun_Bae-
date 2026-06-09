#!/usr/bin/env python3
"""
make_strategy_comparison.py
============================
Tile-wise PCC vs Cosine Similarity 전략 비교 figure 생성
- K값에 따른 gene-wise PCC 비교 (두 전략)

Usage:
    python make_strategy_comparison.py \
        --emb_dir   /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03 \
        --tcga_dir  /project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03 \
        --bulk_csv  /project_antwerp/TCGA-HNSC/ref_file.csv \
        --gene_list /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
        --out_dir   /project_antwerp/hbae/figures
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def predex_tile(tile_emb, train_emb, train_exprs):
    """PredEx for single tile: linear normalized weighted average."""
    sim = tile_emb @ train_emb.T        # (N_train,)
    s = sim.sum()
    w = sim / s if s.abs() > 1e-12 else torch.ones_like(sim) / sim.numel()
    return (w[:, None] * train_exprs).sum(dim=0), sim  # pred, sim


def gene_wise_pcc(preds, bulk):
    """Gene-wise PCC between (n_slides, n_genes) arrays."""
    pccs = []
    for g in range(preds.shape[1]):
        if bulk[:, g].std() > 1e-8 and preds[:, g].std() > 1e-8:
            r, _ = pearsonr(preds[:, g], bulk[:, g])
            if np.isfinite(r):
                pccs.append(r)
    return np.mean(pccs) if pccs else np.nan


def main(args):
    emb_dir  = Path(args.emb_dir)
    tcga_dir = Path(args.tcga_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    train_text  = torch.tensor(np.load(emb_dir / 'train_text_embs.npy')).float()
    train_exprs = torch.tensor(np.load(emb_dir / 'train_exprs.npy')).float()
    train_norm  = F.normalize(train_text, dim=-1)

    with open(args.gene_list) as f:
        gene_names = [l.strip() for l in f if l.strip()]

    # Load bulk RNA-seq
    print("Loading bulk RNA-seq...")
    bulk_raw = pd.read_csv(args.bulk_csv)
    print(f"  Bulk shape: {bulk_raw.shape}")
    print(f"  Columns: {list(bulk_raw.columns[:5])}")

    # Find patient_id column
    id_col = None
    for col in bulk_raw.columns:
        if 'patient' in col.lower() or 'slide' in col.lower() or 'tcga' in str(bulk_raw[col].iloc[0]).upper():
            id_col = col
            break
    if id_col is None:
        # Try first column
        id_col = bulk_raw.columns[0]
    print(f"  Using ID column: {id_col}")

    # Build slide_id -> row mapping
    # slide IDs look like TCGA-CV-6950-01Z-00-DX1
    # patient_id may look like TCGA-CV-6950
    # Try exact match first, then prefix match
    bulk_raw = bulk_raw.set_index(id_col)

    # Get slide files
    slide_files = [f for f in sorted(tcga_dir.glob('*.npy'))
                   if '_coords' not in f.name]
    print(f"  {len(slide_files)} slide embedding files found")

    # Match: try exact, then prefix (patient_id is prefix of slide_id)
    matched = []
    matched_bulk_ids = []
    for sf in slide_files:
        sid = sf.stem  # e.g. TCGA-CV-6950-01Z-00-DX1
        if sid in bulk_raw.index:
            matched.append(sf)
            matched_bulk_ids.append(sid)
        else:
            # Try matching by patient prefix (first 3 parts: TCGA-XX-XXXX)
            prefix = '-'.join(sid.split('-')[:3])  # TCGA-CV-6950
            hits = [idx for idx in bulk_raw.index if str(idx).startswith(prefix)]
            if hits:
                matched.append(sf)
                matched_bulk_ids.append(hits[0])
    print(f"  {len(matched)} slides matched to bulk RNA-seq")

    if len(matched) == 0:
        print("No matches found.")
        print(f"  Slide ID sample: {[f.stem for f in slide_files[:3]]}")
        print(f"  Bulk ID sample:  {list(bulk_raw.index[:3])}")
        return

    # Common genes
    common_genes = [g for g in gene_names if g in bulk_raw.columns]
    print(f"  Common genes: {len(common_genes)}")
    bulk_sub = bulk_raw.loc[matched_bulk_ids, common_genes].values.astype(np.float32)
    gene_idx = [gene_names.index(g) for g in common_genes]

    # Per-slide: compute tile-wise PCC scores and cosine similarity scores
    K_values = [50, 100, 200, 300, 500, 1000]
    all_tile_scores_pcc = []   # list of (n_tiles,) arrays per slide
    all_tile_scores_cos = []
    all_tile_preds      = []   # list of (n_tiles, n_genes) per slide

    print("\nProcessing slides...")
    for sf in tqdm(matched[:min(len(matched), 100)]):  # cap at 100 for speed
        sid = sf.stem
        try:
            tile_embs = torch.tensor(np.load(sf)).float()
            if tile_embs.ndim == 1:
                tile_embs = tile_embs.unsqueeze(0)
            tile_norm = F.normalize(tile_embs, dim=-1)
            n_tiles = tile_norm.shape[0]

            # Get bulk label for this slide
            slide_idx = [f.stem for f in matched].index(sid)
            bulk_label = bulk_sub[slide_idx]  # (n_common_genes,)

            tile_preds_list = []
            tile_sims_list  = []
            for i in range(n_tiles):
                pred, sim = predex_tile(tile_norm[i], train_norm, train_exprs)
                pred_np = pred.cpu().numpy()[gene_idx]
                tile_preds_list.append(pred_np)
                tile_sims_list.append(sim.cpu().numpy())

            tile_preds = np.array(tile_preds_list)  # (n_tiles, n_common_genes)

            # Strategy 1: cosine similarity score (tile pred vs bulk)
            cos_scores = np.array([
                np.dot(tile_preds[i], bulk_label) /
                (np.linalg.norm(tile_preds[i]) * np.linalg.norm(bulk_label) + 1e-12)
                for i in range(n_tiles)
            ])

            # Strategy 2: tile-wise PCC score (tile pred vs bulk)
            pcc_scores = np.array([
                pearsonr(tile_preds[i], bulk_label)[0]
                if tile_preds[i].std() > 1e-8 else 0.0
                for i in range(n_tiles)
            ])

            all_tile_scores_cos.append(cos_scores)
            all_tile_scores_pcc.append(pcc_scores)
            all_tile_preds.append(tile_preds)

        except Exception as e:
            print(f"  [SKIP] {sid}: {e}")

    if len(all_tile_preds) == 0:
        print("No slides processed.")
        return

    n_processed = len(all_tile_preds)
    bulk_used = bulk_sub[:n_processed]

    # Evaluate at each K
    results_cos = []
    results_pcc = []
    baseline_pcc = None

    for K in K_values:
        slide_preds_cos = []
        slide_preds_pcc = []
        slide_preds_all = []

        for i in range(n_processed):
            n_tiles = len(all_tile_scores_cos[i])
            k_actual = min(K, n_tiles)

            # Cosine strategy
            top_k_cos = np.argsort(all_tile_scores_cos[i])[::-1][:k_actual]
            pred_cos = all_tile_preds[i][top_k_cos].mean(axis=0)
            slide_preds_cos.append(pred_cos)

            # PCC strategy
            top_k_pcc = np.argsort(all_tile_scores_pcc[i])[::-1][:k_actual]
            pred_pcc = all_tile_preds[i][top_k_pcc].mean(axis=0)
            slide_preds_pcc.append(pred_pcc)

            # All tiles baseline
            if baseline_pcc is None:
                slide_preds_all.append(all_tile_preds[i].mean(axis=0))

        preds_cos = np.array(slide_preds_cos)
        preds_pcc_arr = np.array(slide_preds_pcc)

        results_cos.append(gene_wise_pcc(preds_cos, bulk_used[:n_processed]))
        results_pcc.append(gene_wise_pcc(preds_pcc_arr, bulk_used[:n_processed]))

        if baseline_pcc is None:
            preds_all = np.array(slide_preds_all)
            baseline_pcc = gene_wise_pcc(preds_all, bulk_used[:n_processed])

    print(f"\nBaseline (all tiles): {baseline_pcc:.4f}")
    print(f"\nK | Cosine Sim | Tile-wise PCC")
    for k, c, p in zip(K_values, results_cos, results_pcc):
        print(f"{k:5d} | {c:.4f} | {p:.4f}")

    # ── Figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(K_values, results_cos, 'o-', color='steelblue', linewidth=2,
            markersize=7, label='Strategy 1: Cosine similarity')
    ax.plot(K_values, results_pcc, 's-', color='tomato', linewidth=2,
            markersize=7, label='Strategy 2: Tile-wise PCC')
    ax.axhline(baseline_pcc, color='grey', linestyle='--', linewidth=1.5,
               label=f'All tiles baseline ({baseline_pcc:.3f})')

    # Annotate K=100 values
    k100_idx = K_values.index(100)
    ax.annotate(f'{results_cos[k100_idx]:.3f}',
                xy=(100, results_cos[k100_idx]),
                xytext=(130, results_cos[k100_idx] - 0.01),
                fontsize=9, color='steelblue')
    ax.annotate(f'{results_pcc[k100_idx]:.3f}',
                xy=(100, results_pcc[k100_idx]),
                xytext=(130, results_pcc[k100_idx] + 0.005),
                fontsize=9, color='tomato')

    ax.set_xlabel('Number of selected tiles (K)', fontsize=12)
    ax.set_ylabel('Gene-wise PCC (mean)', fontsize=12)
    ax.set_title('Tile Selection Strategy Comparison\n'
                 'Gene-wise PCC vs. Number of Selected Tiles per Slide',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 1200)

    out_path = out_dir / 'strategy_comparison_pcc_vs_cosine.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Saved: {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb_dir',   required=True)
    ap.add_argument('--tcga_dir',  required=True)
    ap.add_argument('--bulk_csv',  required=True)
    ap.add_argument('--gene_list', required=True)
    ap.add_argument('--out_dir',   required=True)
    args = ap.parse_args()
    main(args)