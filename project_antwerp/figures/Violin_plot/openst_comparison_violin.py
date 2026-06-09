#!/usr/bin/env python3
"""
OpenST 4-method comparison violin plot
fold_01~10 concat → 방법별 단일 violin 4개
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path('/project_antwerp/hbae/Loki_output')

METHODS = [
    {
        'label': 'All genes\n(1,948)',
        'dir':   BASE / 'openst_validation_agg_v2',
        'file':  'openst_allgene_pcc.npy',
        'color': '#AED6F1',
    },
    {
        'label': 'HEG top-300\n(Loki paper)',
        'dir':   BASE / 'openst_validation_agg_v2',
        'file':  'openst_genewise_pcc.npy',
        'color': '#A9DFBF',
    },
    {
        'label': 'Scanpy HVG\ntop-300',
        'dir':   BASE / 'openst_validation_hvg300',
        'file':  'openst_genewise_pcc.npy',
        'color': '#F5CBA7',
    },
    {
        'label': 'Top PCC\ntop-300 (oracle)',
        'dir':   BASE / 'openst_validation_agg_v2',
        'file':  'openst_genewise_pcc_toppcc.npy',
        'color': '#A9DFBF',
    },
]
# 마지막 색 구분
METHODS[3]['color'] = '#D2B4DE'

# ── fold concat
for m in METHODS:
    arrays = []
    for fold_dir in sorted(m['dir'].glob('fold_*')):
        npy = fold_dir / m['file']
        if npy.exists():
            arrays.append(np.load(npy))
    m['data'] = np.concatenate(arrays) if arrays else np.array([])
    print(f"{m['label'].replace(chr(10),' ')}: n={len(m['data'])}, mean={m['data'].mean():.4f}")

# ── Plot
fig, ax = plt.subplots(figsize=(8, 6))

pos    = list(range(1, len(METHODS) + 1))
vdata  = [m['data'] for m in METHODS]
colors = [m['color'] for m in METHODS]

vp = ax.violinplot(vdata, positions=pos,
                   showmeans=True, showmedians=True,
                   showextrema=True, widths=0.65)

for body, c in zip(vp['bodies'], colors):
    body.set_facecolor(c)
    body.set_alpha(0.80)
    body.set_edgecolor('black')
    body.set_linewidth(0.8)

for k in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
    if k in vp:
        vp[k].set_color('black')
        vp[k].set_linewidth(1.6 if k == 'cmeans' else 1.0)
        if k == 'cmeans':
            vp[k].set_linestyle('--')

ax.set_ylim(-0.4, 1.0)
ymax = ax.get_ylim()[1]

# mean 값 상단 표시
for p, m in zip(pos, METHODS):
    ax.text(p, ymax, f"mean={m['data'].mean():.3f}",
            ha='center', va='bottom', fontsize=8,
            color='#111', fontweight='bold')

ax.set_xticks(pos)
ax.set_xticklabels([m['label'] for m in METHODS], fontsize=10)
ax.set_xlim(0.3, len(METHODS) + 0.7)
ax.set_ylabel('Gene-wise PCC', fontsize=12)
ax.axhline(0, color='red', linestyle=':', linewidth=0.9, alpha=0.6)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.set_title('Open-ST HNSCC External Validation\nGene-wise PCC — All Evaluation Schemes\n(10 folds merged)',
             fontsize=12, fontweight='bold', pad=20)

# 범례
handles = [mpatches.Patch(facecolor=m['color'], label=m['label'].replace('\n',' '),
                           alpha=0.85, edgecolor='black', linewidth=0.5)
           for m in METHODS]
ax.legend(handles=handles, loc='lower right', fontsize=9,
          framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
out = Path('/project_antwerp/hbae/script/0208_start/Violin_plot/openst_comparison_violin.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")