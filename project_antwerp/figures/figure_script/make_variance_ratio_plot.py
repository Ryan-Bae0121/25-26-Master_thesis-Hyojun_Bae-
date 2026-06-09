#!/usr/bin/env python3
"""
make_variance_ratio_plot.py
===========================
ST validation에서 각 sample의 pred_var / bulk_var 분포를 시각화

Usage:
    python make_variance_ratio_plot.py \
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


def extract_sample_id(img_path: str) -> str:
    parts = Path(img_path).parts
    if 'Processed_Data' in parts:
        idx = parts.index('Processed_Data')
        return parts[idx + 2]
    return Path(img_path).parent.parent.name


def predex(val_emb, train_emb, train_exprs):
    sim = val_emb @ train_emb.T
    s = sim.sum()
    w = sim / s if s.abs() > 1e-12 else torch.ones_like(sim) / sim.numel()
    return (w[:, None] * train_exprs).sum(dim=0)


def main(args):
    emb_base = Path(args.emb_base)
    csv_base = Path(args.csv_base)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # domain shift samples to exclude
    exclude = {'Patient1', 'Patient2', 'Patient3', 'Patient4', '19h1257'}

    sample_results = {}  # sid -> {'var_ratio': float, 'gene_pcc': float}

    for fold_i in range(1, 11):
        fold_str = f"fold_{fold_i:02d}"
        emb_dir  = emb_base / fold_str
        csv_path = csv_base / f"{fold_str}_val.csv"

        if not emb_dir.exists() or not csv_path.exists():
            print(f"[SKIP] {fold_str}")
            continue

        print(f"Processing {fold_str}...")

        train_text  = torch.tensor(np.load(emb_dir / 'train_text_embs.npy')).float()
        train_exprs = torch.tensor(np.load(emb_dir / 'train_exprs.npy')).float()
        val_img     = torch.tensor(np.load(emb_dir / 'val_img_embs.npy')).float()
        val_exprs   = np.load(emb_dir / 'val_exprs.npy')

        train_norm = F.normalize(train_text, dim=-1)
        val_norm   = F.normalize(val_img, dim=-1)

        mean_expr  = val_exprs.mean(axis=0)
        top300_idx = np.argsort(mean_expr)[::-1][:300]

        df = pd.read_csv(csv_path)
        df['sample_id'] = df['img_path'].apply(extract_sample_id)
        df = df.reset_index(drop=True)

        preds_list = []
        for i in tqdm(range(len(val_img)), desc=f"  {fold_str}", leave=False):
            pred = predex(val_norm[i], train_norm, train_exprs)
            preds_list.append(pred.cpu().numpy())
        preds_all = np.array(preds_list)

        preds_top300 = preds_all[:, top300_idx]
        exprs_top300 = val_exprs[:, top300_idx]

        for sid, group in df.groupby('sample_id'):
            if sid in exclude:
                continue
            idx = group.index.tolist()
            p = preds_top300[idx]  # (n_spots, 300)
            v = exprs_top300[idx]

            # variance ratio: mean over genes
            pred_var = p.var(axis=0)   # per-gene variance across spots
            bulk_var = v.var(axis=0)
            # only genes with non-zero bulk variance
            mask = bulk_var > 1e-8
            var_ratio = (pred_var[mask] / bulk_var[mask]).mean()

            # gene-wise PCC
            gene_pccs = []
            for g in range(p.shape[1]):
                if v[:, g].std() > 1e-8:
                    r, _ = pearsonr(p[:, g], v[:, g])
                    if np.isfinite(r):
                        gene_pccs.append(r)
            gene_pcc = np.mean(gene_pccs) if gene_pccs else np.nan

            if sid not in sample_results:
                sample_results[sid] = {'var_ratios': [], 'gene_pccs': []}
            sample_results[sid]['var_ratios'].append(var_ratio)
            sample_results[sid]['gene_pccs'].append(gene_pcc)

    # average across folds
    sids = sorted(sample_results.keys())
    var_ratios = [np.mean(sample_results[s]['var_ratios']) for s in sids]
    gene_pccs  = [np.mean(sample_results[s]['gene_pccs'])  for s in sids]

    print(f"\n{len(sids)} domain-matched samples")
    print(f"Mean var ratio: {np.mean(var_ratios):.3f} ({np.mean(var_ratios)*100:.1f}%)")
    print(f"Mean gene PCC:  {np.mean(gene_pccs):.3f}")
    r_corr, p_corr = pearsonr(var_ratios, gene_pccs)
    print(f"Var ratio vs gene PCC: r={r_corr:.3f}, p={p_corr:.4f}")

    # ── Figure: 2-panel ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Predicted Expression Variance Compression\n"
        "ST Validation — 31 Domain-Matched Samples (Top 300 HVGs)",
        fontsize=12, fontweight='bold'
    )

    # Panel 1: Histogram of var ratio across samples
    ax = axes[0]
    ax.hist(var_ratios, bins=15, color='steelblue', edgecolor='white', linewidth=0.8)
    ax.axvline(np.mean(var_ratios), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(var_ratios)*100:.1f}%')
    ax.axvline(1.0, color='grey', linestyle=':', linewidth=1.2, label='Perfect recovery (100%)')
    ax.set_xlabel("Pred variance / Ground-truth variance", fontsize=11)
    ax.set_ylabel("Number of samples", fontsize=11)
    ax.set_title("Distribution of variance ratio\nacross 31 samples", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(max(var_ratios) * 1.1, 0.5))

    # Panel 2: Scatter var ratio vs gene PCC
    ax = axes[1]
    ax.scatter(var_ratios, gene_pccs, s=40, alpha=0.75, color='steelblue', linewidths=0)
    # regression line
    m, b = np.polyfit(var_ratios, gene_pccs, 1)
    xs = np.linspace(min(var_ratios), max(var_ratios), 100)
    ax.plot(xs, m * xs + b, 'r-', linewidth=1.5,
            label=f'r = {r_corr:.3f}, p = {p_corr:.4f}')
    ax.set_xlabel("Pred variance / Ground-truth variance", fontsize=11)
    ax.set_ylabel("Gene-wise PCC", fontsize=11)
    ax.set_title("Variance ratio vs. gene-wise PCC\nacross 31 samples", fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    out_path = out_dir / "variance_ratio_distribution.png"
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Saved: {out_path}")

    # Save CSV
    df_out = pd.DataFrame({
        'sample_id': sids,
        'var_ratio_mean': var_ratios,
        'gene_pcc_mean': gene_pccs
    })
    df_out.to_csv(out_dir / "variance_ratio_per_sample.csv", index=False)
    print(f"✅ Saved CSV: {out_dir / 'variance_ratio_per_sample.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_base", required=True)
    ap.add_argument("--csv_base", required=True)
    ap.add_argument("--out_dir",  required=True)
    args = ap.parse_args()
    main(args)