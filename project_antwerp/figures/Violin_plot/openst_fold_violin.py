#!/usr/bin/env python3
"""
OpenST fold-wise violin - 4가지 방법 × 10 folds
2행 2열 subplot
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/project_antwerp/hbae/Loki_output')

METHODS = [
    {
        'label': 'All genes (1,948)',
        'dir':   BASE / 'openst_validation_agg_v2',
        'file':  'openst_allgene_pcc.npy',
        'color': '#AED6F1',
    },
    {
        'label': 'HEG top-300 (Loki paper)',
        'dir':   BASE / 'openst_validation_agg_v2',
        'file':  'openst_genewise_pcc.npy',
        'color': '#A9DFBF',
    },
    {
        'label': 'Scanpy HVG top-300',
        'dir':   BASE / 'openst_validation_hvg300',
        'file':  'openst_genewise_pcc.npy',
        'color': '#F5CBA7',
    },
    {
        'label': 'Top PCC top-300 (oracle)',
        'dir':   BASE / 'openst_validation_agg_v2',
        'file':  'openst_genewise_pcc_toppcc.npy',
        'color': '#D2B4DE',
    },
]

# fold별 데이터 로드
for m in METHODS:
    m['folds'] = []
    for fold_dir in sorted(m['dir'].glob('fold_*')):
        npy = fold_dir / m['file']
        if npy.exists():
            m['folds'].append(np.load(npy))
    print(f"{m['label']}: {len(m['folds'])} folds loaded")

# ── 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, m in zip(axes, METHODS):
    n = len(m['folds'])
    pos = list(range(1, n + 1))

    vp = ax.violinplot(m['folds'], positions=pos,
                       showmeans=True, showmedians=True,
                       showextrema=True, widths=0.7)

    for body in vp['bodies']:
        body.set_facecolor(m['color'])
        body.set_alpha(0.78)
        body.set_edgecolor('black')
        body.set_linewidth(0.6)

    for k in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if k in vp:
            vp[k].set_color('black')
            vp[k].set_linewidth(1.6 if k == 'cmeans' else 1.0)
            if k == 'cmeans':
                vp[k].set_linestyle('--')

    ax.set_ylim(-0.4, 1.0)
    ymax = ax.get_ylim()[1]

    # mean 값 상단 표시
    for p, arr in zip(pos, m['folds']):
        ax.text(p, ymax, f"{arr.mean():.3f}",
                ha='center', va='bottom', fontsize=7,
                color='#111', fontweight='bold')

    # 전체 mean 표시
    all_mean = np.concatenate(m['folds']).mean()
    ax.axhline(all_mean, color=m['color'], linestyle='-',
               linewidth=1.5, alpha=0.8,
               label=f'overall mean={all_mean:.3f}')

    ax.set_xticks(pos)
    ax.set_xticklabels([f'f{i:02d}' for i in range(1, n+1)], fontsize=8)
    ax.set_xlim(0.3, n + 0.7)
    ax.set_ylabel('Gene-wise PCC', fontsize=10)
    ax.axhline(0, color='red', linestyle=':', linewidth=0.9, alpha=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_title(m['label'], fontsize=11, fontweight='bold', pad=8)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

fig.suptitle('Open-ST HNSCC External Validation\nGene-wise PCC per Fold — All Evaluation Schemes',
             fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
out = Path('/project_antwerp/hbae/script/0208_start/Violin_plot/openst_fold_violin.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")