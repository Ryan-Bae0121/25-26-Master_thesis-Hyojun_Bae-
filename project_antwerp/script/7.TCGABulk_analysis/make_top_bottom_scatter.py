#!/usr/bin/env python3
"""
make_top_bottom_scatter.py
==========================
10-fold ST validation에서 gene-wise PCC 기준 TOP/BOTTOM sample scatter plot 생성

Usage:
    python make_top_bottom_scatter.py \
        --emb_base /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new \
        --csv_base /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold \
        --out_dir  /project_antwerp/hbae/figures
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
import matplotlib.gridspec as gridspec


# ─── helpers ──────────────────────────────────────────────────────────────────

def extract_sample_id(img_path: str) -> str:
    parts = Path(img_path).parts
    if 'Processed_Data' in parts:
        idx = parts.index('Processed_Data')
        return parts[idx + 2]
    return Path(img_path).parent.parent.name


def predex(val_emb, train_emb, train_exprs):
    """Loki-style linear-normalized weighted average (no top-k)."""
    sim = val_emb @ train_emb.T          # (N_train,)
    s = sim.sum()
    w = sim / s if s.abs() > 1e-12 else torch.ones_like(sim) / sim.numel()
    return (w[:, None] * train_exprs).sum(dim=0)


def calc_gene_pcc(preds, exprs):
    """Gene-wise PCC across spots. Returns array of per-gene PCC."""
    corrs = []
    for g in range(preds.shape[1]):
        if exprs[:, g].std() > 1e-8:
            r, _ = pearsonr(preds[:, g], exprs[:, g])
            if np.isfinite(r):
                corrs.append(r)
            else:
                corrs.append(np.nan)
        else:
            corrs.append(np.nan)
    return np.array(corrs)


def slide_gene_pcc(preds_top300, exprs_top300):
    """Mean gene-wise PCC for one sample (scalar)."""
    corrs = calc_gene_pcc(preds_top300, exprs_top300)
    return np.nanmean(corrs)


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    emb_base = Path(args.emb_base)
    csv_base = Path(args.csv_base)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: collect per-sample predictions across all folds ──────────────
    # dict: sample_id -> {'preds': [...], 'exprs': [...]}
    sample_data = {}

    for fold_i in range(1, 11):
        fold_str = f"fold_{fold_i:02d}"
        emb_dir  = emb_base / fold_str
        csv_path = csv_base / f"{fold_str}_val.csv"

        if not emb_dir.exists():
            print(f"[SKIP] {fold_str} emb_dir not found: {emb_dir}")
            continue
        if not csv_path.exists():
            print(f"[SKIP] {fold_str} csv not found: {csv_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing {fold_str} ...")

        # load
        train_text = torch.tensor(
            np.load(emb_dir / 'train_text_embs.npy')).float()
        train_exprs = torch.tensor(
            np.load(emb_dir / 'train_exprs.npy')).float()
        val_img   = torch.tensor(
            np.load(emb_dir / 'val_img_embs.npy')).float()
        val_exprs = np.load(emb_dir / 'val_exprs.npy')

        # normalize
        train_norm = F.normalize(train_text, dim=-1)
        val_norm   = F.normalize(val_img, dim=-1)

        # top 300 expressed genes in this fold's val set
        mean_expr   = val_exprs.mean(axis=0)
        top300_idx  = np.argsort(mean_expr)[::-1][:300]

        # sample map
        df = pd.read_csv(csv_path)
        df['sample_id'] = df['img_path'].apply(extract_sample_id)
        df = df.reset_index(drop=True)

        # predict all spots
        preds_list = []
        for i in tqdm(range(len(val_img)), desc=f"  Predicting {fold_str}",
                      leave=False):
            pred = predex(val_norm[i], train_norm, train_exprs)
            preds_list.append(pred.cpu().numpy())
        preds_all = np.array(preds_list)          # (n_spots, n_genes)

        preds_top300 = preds_all[:, top300_idx]
        exprs_top300 = val_exprs[:, top300_idx]

        # store per sample
        for sid, group in df.groupby('sample_id'):
            idx = group.index.tolist()
            if sid not in sample_data:
                sample_data[sid] = {'preds': [], 'exprs': []}
            sample_data[sid]['preds'].append(preds_top300[idx])
            sample_data[sid]['exprs'].append(exprs_top300[idx])

    if not sample_data:
        print("No data collected. Check paths.")
        return

    # ── Step 2: compute slide-level gene-wise PCC per sample ─────────────────
    sample_pcc = {}
    sample_preds_mean = {}
    sample_exprs_mean = {}

    for sid, d in sample_data.items():
        preds = np.concatenate(d['preds'], axis=0)   # (all_spots, 300)
        exprs = np.concatenate(d['exprs'], axis=0)

        pcc = slide_gene_pcc(preds, exprs)
        sample_pcc[sid] = pcc

        # slide-level mean expression per gene (for scatter)
        sample_preds_mean[sid] = preds.mean(axis=0)  # (300,)
        sample_exprs_mean[sid] = exprs.mean(axis=0)

    # sort by gene-wise PCC
    sorted_samples = sorted(sample_pcc.items(), key=lambda x: x[1], reverse=True)
    print("\n\nAll samples ranked by gene-wise PCC:")
    for sid, pcc in sorted_samples:
        print(f"  {sid:40s}  PCC = {pcc:.4f}")

    # GSE220978 + 19h1257 제외 후 TOP/BOTTOM 선택
    exclude = {'Patient1', 'Patient2', 'Patient3', 'Patient4', '19h1257'}
    filtered = [(s, p) for s, p in sorted_samples if s not in exclude]

    n_show = args.n_show   # default 3
    top_samples    = filtered[:n_show]
    bottom_samples = filtered[-n_show:]

    print(f"\nTOP {n_show} samples (domain-matched):")
    for s, p in top_samples:
        print(f"  {s}: {p:.4f}")
    print(f"\nBOTTOM {n_show} samples (domain-matched):")
    for s, p in bottom_samples:
        print(f"  {s}: {p:.4f}")

    # ── Step 3: scatter plots ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(5 * n_show, 11))
    gs  = gridspec.GridSpec(2, n_show, hspace=0.45, wspace=0.35)

    def plot_scatter(ax, sid, pcc, row_label):
        x = sample_exprs_mean[sid]
        y = sample_preds_mean[sid]

        ax.scatter(x, y, s=12, alpha=0.55, color='steelblue', linewidths=0)

        # identity line range
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2, alpha=0.7, label='y = x')

        # regression line
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, m * xs + b, 'k-', lw=1.2, alpha=0.8)

        short_id = sid if len(sid) <= 22 else sid[:19] + '...'
        ax.set_title(f"{short_id}\nGene-wise PCC = {pcc:.3f}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel("Ground truth (mean expr)", fontsize=8)
        ax.set_ylabel("Predicted (mean expr)", fontsize=8)
        ax.tick_params(labelsize=7)

        # row label on leftmost
        if ax.get_subplotspec().colspan.start == 0:
            ax.set_ylabel(f"{row_label}\nPredicted (mean expr)", fontsize=8)

    for col, (sid, pcc) in enumerate(top_samples):
        ax = fig.add_subplot(gs[0, col])
        plot_scatter(ax, sid, pcc, "TOP")

    for col, (sid, pcc) in enumerate(bottom_samples):
        ax = fig.add_subplot(gs[1, col])
        plot_scatter(ax, sid, pcc, "BOTTOM")

    # row labels
    fig.text(0.01, 0.75, f"TOP {n_show}", va='center', rotation='vertical',
             fontsize=11, fontweight='bold', color='#1a6e3d')
    fig.text(0.01, 0.27, f"BOTTOM {n_show}", va='center', rotation='vertical',
             fontsize=11, fontweight='bold', color='#b22222')

    fig.suptitle(
        "Predicted vs. Ground-Truth Gene Expression\n"
        "TOP vs. BOTTOM Samples by Gene-wise PCC (domain-matched, top 300 HVGs)",
        fontsize=11, fontweight='bold', y=1.01
    )

    out_path = out_dir / f"top{n_show}_bottom{n_show}_scatter.png"
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Saved: {out_path}")

    # ── Step 4: also save PCC summary CSV ────────────────────────────────────
    pcc_df = pd.DataFrame(
        [(s, p) for s, p in sorted_samples],
        columns=['sample_id', 'gene_wise_pcc']
    )
    pcc_df['excluded'] = pcc_df['sample_id'].isin(exclude)
    pcc_csv = out_dir / "sample_gene_pcc_summary.csv"
    pcc_df.to_csv(pcc_csv, index=False)
    print(f"✅ Saved PCC summary: {pcc_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_base", required=True,
                    help="Base dir containing fold_01 .. fold_10")
    ap.add_argument("--csv_base", required=True,
                    help="Base dir containing fold_01_val.csv ..")
    ap.add_argument("--out_dir",  required=True,
                    help="Output directory for figures")
    ap.add_argument("--n_show",   type=int, default=3,
                    help="Number of TOP/BOTTOM samples to show (default 3)")
    args = ap.parse_args()
    main(args)