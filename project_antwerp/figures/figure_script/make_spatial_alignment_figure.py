#!/usr/bin/env python3
"""
make_spatial_alignment_figure.py
==================================
Figure 4.2: Spatial alignment verification using KRT14 and CD3E
Uses GSM7998257 (GSE252265, tongue SCC) as representative sample

Usage:
    python make_spatial_alignment_figure.py \
        --data_dir  /project_antwerp/hbae/data/0317_HVG_NEW \
        --out_dir   /project_antwerp/hbae/figures \
        --sample_id GSM7998257
"""

import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')


def find_sample_h5(data_dir, sample_id):
    """Find h5ad file for given sample."""
    data_dir = Path(data_dir)
    # Try common patterns
    patterns = [
        f"{sample_id}*.h5ad",
        f"*{sample_id}*.h5ad",
        f"{sample_id}/{sample_id}*.h5ad",
    ]
    for pattern in patterns:
        files = list(data_dir.rglob(pattern))
        if files:
            return files[0]
    return None


def plot_spatial_gene(ax, adata, gene, title, cmap='viridis', spot_size=1.5):
    """Plot spatial expression of a single gene."""
    if gene not in adata.var_names:
        print(f"  Warning: {gene} not in var_names")
        # Try raw counts
        expr = np.zeros(adata.n_obs)
    else:
        expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, 'toarray') \
               else adata[:, gene].X.flatten()

    # Get spatial coordinates
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
        x = coords[:, 0]
        y = coords[:, 1]
    elif 'array_col' in adata.obs.columns and 'array_row' in adata.obs.columns:
        x = adata.obs['array_col'].values
        y = adata.obs['array_row'].values
    else:
        # Try pxl_col_in_fullres
        x = adata.obs.get('pxl_col_in_fullres', np.arange(adata.n_obs))
        y = adata.obs.get('pxl_row_in_fullres', np.arange(adata.n_obs))

    # Plot tissue background (if image available)
    if 'spatial' in adata.uns and len(adata.uns['spatial']) > 0:
        library_id = list(adata.uns['spatial'].keys())[0]
        if 'images' in adata.uns['spatial'][library_id]:
            img = adata.uns['spatial'][library_id]['images'].get('hires')
            if img is not None:
                scalef = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
                ax.imshow(img, origin='upper', alpha=0.5)
                x_plot = x * scalef
                y_plot = y * scalef
            else:
                x_plot, y_plot = x, -y
        else:
            x_plot, y_plot = x, -y
    else:
        x_plot, y_plot = x, -y

    sc_plot = ax.scatter(
        x_plot, y_plot,
        c=expr,
        cmap=cmap,
        s=spot_size,
        alpha=0.85,
        linewidths=0,
        vmin=0,
        vmax=np.percentile(expr[expr > 0], 95) if (expr > 0).any() else 1
    )

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('spatial1', fontsize=9)
    ax.set_ylabel('spatial2', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal')

    cbar = plt.colorbar(sc_plot, ax=ax, shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    return expr


def main(args):
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Looking for sample: {args.sample_id}")

    # Find h5ad file
    h5_file = find_sample_h5(data_dir, args.sample_id)

    if h5_file is None:
        # Try to find any h5ad in data_dir
        all_h5 = list(data_dir.rglob('*.h5ad'))
        print(f"Sample not found. Available h5ad files ({len(all_h5)}):")
        for f in all_h5[:10]:
            print(f"  {f.name}")
        return

    print(f"Loading: {h5_file}")
    adata = sc.read_h5ad(h5_file)
    print(f"  Spots: {adata.n_obs}, Genes: {adata.n_vars}")
    print(f"  obs keys: {list(adata.obs.columns[:5])}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")

    # Check genes
    for gene in ['KRT14', 'CD3E']:
        found = gene in adata.var_names
        print(f"  {gene}: {'found' if found else 'NOT FOUND'}")
        if not found:
            # Try case-insensitive
            matches = [v for v in adata.var_names if v.upper() == gene.upper()]
            if matches:
                print(f"    -> Found as: {matches[0]}")

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        f'Spatial Alignment Verification — {args.sample_id}\n'
        f'KRT14 (Epithelial Marker) and CD3E (T Cell Marker)',
        fontsize=11, fontweight='bold', y=1.02
    )

    # KRT14
    expr_krt14 = plot_spatial_gene(
        axes[0], adata, 'KRT14', 'KRT14',
        cmap='viridis', spot_size=args.spot_size
    )
    print(f"  KRT14 expression: mean={expr_krt14.mean():.3f}, "
          f"max={expr_krt14.max():.3f}, "
          f"frac>0={(expr_krt14>0).mean():.2%}")

    # CD3E
    expr_cd3e = plot_spatial_gene(
        axes[1], adata, 'CD3E', 'CD3E',
        cmap='viridis', spot_size=args.spot_size
    )
    print(f"  CD3E expression:  mean={expr_cd3e.mean():.3f}, "
          f"max={expr_cd3e.max():.3f}, "
          f"frac>0={(expr_cd3e>0).mean():.2%}")

    plt.tight_layout()

    out_path = out_dir / f'figure4_2_spatial_alignment_{args.sample_id}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ Saved: {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',  required=True,
                    help='Directory containing h5ad files')
    ap.add_argument('--out_dir',   required=True)
    ap.add_argument('--sample_id', default='GSM7998257')
    ap.add_argument('--spot_size', type=float, default=2.0,
                    help='Scatter point size for spots')
    args = ap.parse_args()
    main(args)