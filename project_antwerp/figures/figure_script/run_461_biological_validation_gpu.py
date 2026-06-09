#!/usr/bin/env python3
"""
run_461_biological_validation_gpu.py
=====================================
GPU 가속 버전 - top-500 tile score vs bulk correlation + DE analysis
461 slides, 144px tiles, K=30% sparse retrieval

Usage:
    python run_461_biological_validation_gpu.py \
        --emb_dir   /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03 \
        --tcga_dir  /project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px/fold_03 \
        --bulk_csv  /project_antwerp/TCGA-HNSC/ref_file.csv \
        --gene_list /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
        --out_dir   /project_antwerp/hbae/figures/461_validation \
        --device    cuda:0
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, ttest_ind
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(emb_dir, bulk_csv, gene_list, device):
    """Load train embeddings and bulk RNA-seq."""
    print("Loading train embeddings...")
    train_text  = torch.tensor(np.load(emb_dir / 'train_text_embs.npy')).float().to(device)
    train_exprs = torch.tensor(np.load(emb_dir / 'train_exprs.npy')).float().to(device)
    train_norm  = F.normalize(train_text, dim=-1)
    print(f"  train spots: {train_text.shape[0]}, genes: {train_exprs.shape[1]}")

    with open(gene_list) as f:
        gene_names = [l.strip() for l in f if l.strip()]

    print("Loading bulk RNA-seq...")
    bulk_raw = pd.read_csv(bulk_csv)

    # wsi_file_name -> slide_id (remove extension)
    # Extract base slide ID (first part before UUID)
    # wsi_file_name: TCGA-CV-6950-01Z-00-DX1.UUID
    # Use full string before first '.' as key
    bulk_raw['slide_id'] = bulk_raw['wsi_file_name'].apply(
        lambda x: str(x).split('.')[0]
    )
    bulk_raw = bulk_raw.set_index('slide_id')

    # Gene columns have 'rna_' prefix
    rna_cols = [c for c in bulk_raw.columns if c.startswith('rna_')]
    rna_gene_names = [c[4:] for c in rna_cols]  # remove 'rna_' prefix

    # Common genes
    common_genes = [g for g in gene_names if g in rna_gene_names]
    common_rna_cols = [f'rna_{g}' for g in common_genes]
    gene_idx = [gene_names.index(g) for g in common_genes]

    print(f"  Common genes: {len(common_genes)}")
    print(f"  Slides in bulk: {len(bulk_raw)}")

    return train_norm, train_exprs, gene_names, gene_idx, common_genes, bulk_raw, common_rna_cols


def predex_batch_gpu(tile_embs, train_norm, train_exprs, top_k_pct=0.30):
    """
    GPU batch PredEx with K=30% sparse retrieval.
    tile_embs: (n_tiles, dim) already normalized, on GPU
    Returns: (n_tiles, n_genes)
    """
    # Compute similarities: (n_tiles, n_train)
    sims = tile_embs @ train_norm.T

    k = max(1, int(sims.shape[1] * top_k_pct))

    # Top-K per tile
    topk_sims, topk_idx = sims.topk(k, dim=1)  # (n_tiles, k)

    # Linear normalization per tile
    s = topk_sims.sum(dim=1, keepdim=True).clamp(min=1e-12)
    w = topk_sims / s  # (n_tiles, k)

    # Gather top-k expressions
    # train_exprs: (n_train, n_genes)
    # topk_idx: (n_tiles, k)
    # topk_exprs: (n_tiles, k, n_genes)
    topk_exprs = train_exprs[topk_idx]  # (n_tiles, k, n_genes)

    # Weighted sum: (n_tiles, n_genes)
    preds = (w.unsqueeze(-1) * topk_exprs).sum(dim=1)

    return preds  # (n_tiles, n_genes)


def compute_tile_pcc_scores(tile_preds_np, bulk_label_np):
    """Compute tile-wise PCC scores efficiently."""
    # tile_preds_np: (n_tiles, n_genes)
    # bulk_label_np: (n_genes,)
    # Vectorized pearson r
    n = tile_preds_np.shape[1]
    tp = tile_preds_np - tile_preds_np.mean(axis=1, keepdims=True)
    bl = bulk_label_np - bulk_label_np.mean()
    num = (tp * bl).sum(axis=1)
    denom = np.sqrt((tp**2).sum(axis=1)) * np.sqrt((bl**2).sum()) + 1e-12
    return num / denom


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    emb_dir  = Path(args.emb_dir)
    tcga_dir = Path(args.tcga_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_norm, train_exprs, gene_names, gene_idx, common_genes, bulk_raw, common_rna_cols = \
        load_data(emb_dir, Path(args.bulk_csv), Path(args.gene_list), device)

    # Get slide files
    slide_files = [f for f in sorted(tcga_dir.glob('*.npy'))
                   if '_coords' not in f.name]
    print(f"\n{len(slide_files)} slide embedding files found")

    # Match to bulk
    # npy stem may have .svs appended: TCGA-XX-XXXX-01Z-00-DX1.UUID.svs
    # Extract base ID (before first '.')
    def get_slide_key(sf):
        stem = sf.stem  # removes .npy
        # Remove .svs if present
        if stem.endswith('.svs'):
            stem = stem[:-4]
        # Take part before first '.' (remove UUID)
        return stem.split('.')[0]

    matched = [(sf, get_slide_key(sf))
               for sf in slide_files
               if get_slide_key(sf) in bulk_raw.index]
    print(f"{len(matched)} slides matched to bulk RNA-seq")

    # ── Process slides ────────────────────────────────────────────────────
    slide_top500_scores = {}
    slide_top500_preds  = {}

    BATCH_TILES = 256  # process tiles in batches on GPU

    for sf, sid in tqdm(matched, desc="Processing slides"):
        try:
            tile_embs_np = np.load(sf)
            if tile_embs_np.ndim == 1:
                tile_embs_np = tile_embs_np[np.newaxis, :]

            tile_embs = torch.tensor(tile_embs_np).float().to(device)
            tile_norm = F.normalize(tile_embs, dim=-1)
            n_tiles = tile_norm.shape[0]

            # Batch predict
            all_preds = []
            for start in range(0, n_tiles, BATCH_TILES):
                batch = tile_norm[start:start+BATCH_TILES]
                with torch.no_grad():
                    preds = predex_batch_gpu(batch, train_norm, train_exprs)
                all_preds.append(preds.cpu().numpy()[:, gene_idx])
            tile_preds = np.concatenate(all_preds, axis=0)  # (n_tiles, n_common)

            # Get bulk label
            bulk_label = bulk_raw.loc[sid, common_rna_cols].values.astype(np.float32)  # sid is already the base key

            # Tile-wise PCC scores (vectorized)
            pcc_scores = compute_tile_pcc_scores(tile_preds, bulk_label)
            pcc_scores = np.nan_to_num(pcc_scores, nan=0.0)

            # Top-500 / Bottom-500
            k_sel = min(500, n_tiles)
            top500_idx = np.argsort(pcc_scores)[::-1][:k_sel]
            top500_score = pcc_scores[top500_idx].mean()

            slide_top500_scores[sid] = top500_score
            slide_top500_preds[sid] = tile_preds[top500_idx].mean(axis=0)

        except Exception as e:
            print(f"\n  [SKIP] {sid}: {e}")

    print(f"\n{len(slide_top500_scores)} slides processed successfully")

    # ── 5.4.3: Tile score vs bulk correlation ────────────────────────────
    print("\n=== 5.4.3: Top-500 Tile Score vs Bulk Correlation ===")

    valid_sids = [sid for sid in [m[1] for m in matched] if sid in slide_top500_scores]
    scores = np.array([slide_top500_scores[sid] for sid in valid_sids])
    bulk_matrix = bulk_raw.loc[valid_sids, common_rna_cols].values.astype(np.float32)

    gene_corrs = {}
    for gi, gname in enumerate(common_genes):
        if bulk_matrix[:, gi].std() > 1e-8 and scores.std() > 1e-8:
            r, _ = pearsonr(scores, bulk_matrix[:, gi])
            gene_corrs[gname] = float(r) if np.isfinite(r) else 0.0

    sorted_genes = sorted(gene_corrs.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 positively correlated genes:")
    for g, r in sorted_genes[:20]:
        print(f"  {g}: r = {r:.4f}")
    print("\nBottom 10 negatively correlated genes:")
    for g, r in sorted_genes[-10:]:
        print(f"  {g}: r = {r:.4f}")

    # Focus genes
    focus_genes = ['KRT6B', 'KRT16', 'SPRR1B', 'S100A9', 'S100A12', 'S100A8',
                   'CSTA', 'SPRR3', 'COL1A1', 'VCAN']
    focus_genes = [g for g in focus_genes if g in gene_corrs]

    print("\nFocus gene correlations:")
    for g in focus_genes:
        print(f"  {g}: r = {gene_corrs.get(g, 'N/A'):.4f}")

    # Figure: bar chart + scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Top-500 Tile Score vs Gene Expression Correlation\n'
                 f'(461 slides, 144px tiles, fold_03, K=30% sparse retrieval)',
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    vals = [gene_corrs.get(g, 0) for g in focus_genes]
    colors = ['salmon' if v > 0 else 'steelblue' for v in vals]
    ax.barh(focus_genes[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='r=0.5')
    ax.set_xlabel('Pearson r (top-500 score vs bulk expression)', fontsize=10)
    ax.set_title('Mean Correlation (fold_03)', fontsize=10)
    ax.legend(fontsize=9)
    for i, (g, v) in enumerate(zip(focus_genes[::-1], vals[::-1])):
        offset = 0.01 if v >= 0 else -0.01
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + offset, i, f'{v:.3f}', va='center', ha=ha, fontsize=8)

    ax = axes[1]
    if 'KRT6B' in gene_corrs:
        ki = common_genes.index('KRT6B')
        r_krt = gene_corrs['KRT6B']
        ax.scatter(scores, bulk_matrix[:, ki], s=8, alpha=0.4, color='steelblue')
        m, b = np.polyfit(scores, bulk_matrix[:, ki], 1)
        xs = np.linspace(scores.min(), scores.max(), 100)
        ax.plot(xs, m*xs+b, 'r-', linewidth=1.5)
        ax.set_xlabel('Top-500 tile PCC score', fontsize=10)
        ax.set_ylabel('KRT6B bulk expression', fontsize=10)
        ax.set_title(f'Top-500 Score vs KRT6B\nAll folds combined r = {r_krt:.4f}', fontsize=10)

    plt.tight_layout()
    out1 = out_dir / 'tile500_score_bulk_correlation_461.png'
    fig.savefig(out1, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Saved: {out1}")

    # ── 5.4.4: High vs Low score DE analysis ─────────────────────────────
    print("\n=== 5.4.4: High vs Low Top-500 Score DE Analysis ===")

    score_series = pd.Series({sid: slide_top500_scores[sid] for sid in valid_sids})
    q75 = score_series.quantile(0.75)
    q25 = score_series.quantile(0.25)

    high_sids = score_series[score_series >= q75].index.tolist()
    low_sids  = score_series[score_series <= q25].index.tolist()
    print(f"  High (top 25%, n={len(high_sids)}): score >= {q75:.4f}")
    print(f"  Low  (bot 25%, n={len(low_sids)}):  score <= {q25:.4f}")

    high_bulk = bulk_raw.loc[high_sids, common_rna_cols].values.astype(np.float32)
    low_bulk  = bulk_raw.loc[low_sids,  common_rna_cols].values.astype(np.float32)

    mean_diff = high_bulk.mean(axis=0) - low_bulk.mean(axis=0)
    pvals = np.array([ttest_ind(high_bulk[:, i], low_bulk[:, i]).pvalue
                      for i in range(len(common_genes))])
    neg_log_p = -np.log10(pvals + 1e-300)
    sig_mask  = (pvals < 0.01) & (np.abs(mean_diff) > 0.5)
    up_high   = sig_mask & (mean_diff > 0)
    up_low    = sig_mask & (mean_diff < 0)
    print(f"  Up in High: {up_high.sum()} | Up in Low: {up_low.sum()}")

    top_high_idx = np.argsort(mean_diff)[::-1][:10]
    top_low_idx  = np.argsort(mean_diff)[:10]
    print("\nTop 10 up in HIGH:")
    for i in top_high_idx:
        print(f"  {common_genes[i]}: diff={mean_diff[i]:.3f}, p={pvals[i]:.2e}")
    print("\nTop 10 up in LOW:")
    for i in top_low_idx:
        print(f"  {common_genes[i]}: diff={mean_diff[i]:.3f}, p={pvals[i]:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'High vs Low Top-500 Score Slides: DE Gene Analysis\n'
                 f'High (top 25%, score\u2265{q75:.4f}, n={len(high_sids)}) vs '
                 f'Low (bot 25%, score\u2264{q25:.4f}, n={len(low_sids)})\n'
                 f'461 slides, 144px tiles, fold_03, K=30%',
                 fontsize=10, fontweight='bold')

    ax = axes[0]
    ax.scatter(mean_diff[~sig_mask], neg_log_p[~sig_mask], s=4, alpha=0.3, color='grey')
    ax.scatter(mean_diff[up_high], neg_log_p[up_high], s=8, alpha=0.6,
               color='salmon', label=f'Up in High ({up_high.sum()})')
    ax.scatter(mean_diff[up_low], neg_log_p[up_low], s=8, alpha=0.6,
               color='steelblue', label=f'Up in Low ({up_low.sum()})')
    ax.axhline(-np.log10(0.01), color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(0.5,  color='red',  linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(-0.5, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
    for i in top_high_idx[:5]:
        ax.annotate(common_genes[i], (mean_diff[i], neg_log_p[i]), fontsize=7, color='darkred')
    for i in top_low_idx[:5]:
        ax.annotate(common_genes[i], (mean_diff[i], neg_log_p[i]), fontsize=7, color='darkblue')
    ax.set_xlabel('Mean difference (High - Low)', fontsize=10)
    ax.set_ylabel('-log10(p-value)', fontsize=10)
    ax.set_title('Volcano Plot', fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    top10h = [common_genes[i] for i in top_high_idx]
    top10l = [common_genes[i] for i in top_low_idx]
    all_n  = top10h + top10l
    all_v  = [mean_diff[i] for i in top_high_idx] + [mean_diff[i] for i in top_low_idx]
    cols_b = ['salmon']*10 + ['steelblue']*10
    ax.barh(all_n[::-1], all_v[::-1], color=cols_b[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean diff (High - Low)', fontsize=10)
    ax.set_title('Top-10 DE Genes', fontsize=10)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    out2 = out_dir / 'DE_analysis_high_low_top500_461.png'
    fig.savefig(out2, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Saved: {out2}")

    # Save CSVs
    pd.DataFrame({'gene': list(gene_corrs.keys()),
                  'pearson_r': list(gene_corrs.values())}) \
      .sort_values('pearson_r', ascending=False) \
      .to_csv(out_dir / 'tile500_score_correlation_461.csv', index=False)

    pd.DataFrame({'gene': common_genes, 'mean_diff': mean_diff,
                  'pval': pvals, 'up_in_high': up_high, 'up_in_low': up_low}) \
      .sort_values('mean_diff', ascending=False) \
      .to_csv(out_dir / 'DE_results_top500_461.csv', index=False)

    print(f"\n✅ All outputs saved to {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb_dir',   required=True)
    ap.add_argument('--tcga_dir',  required=True)
    ap.add_argument('--bulk_csv',  required=True)
    ap.add_argument('--gene_list', required=True)
    ap.add_argument('--out_dir',   required=True)
    ap.add_argument('--device',    default='cuda:0')
    args = ap.parse_args()
    main(args)