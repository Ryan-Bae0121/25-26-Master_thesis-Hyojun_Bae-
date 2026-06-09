import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

GENE_LIST   = '/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt'
TRAIN_EXPRS = '/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01/train_exprs.npy'
VAL_EXPRS   = '/project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/tile_exprs.npy'
OUT_DIR     = '/project_antwerp/hbae/script/0208_start/Visium_HD/plots/'

C1, C2 = '#2E75B6', '#ED7D31'

gene_names  = open(GENE_LIST).read().strip().split('\n')
train_exprs = np.load(TRAIN_EXPRS)
val_exprs   = np.load(VAL_EXPRS)

train_mean = train_exprs.mean(axis=0)
hd_mean    = val_exprs.mean(axis=0)

expressed_hd = [(gene_names[i], float(hd_mean[i]))
                for i in range(len(gene_names)) if hd_mean[i] > 0.5]
expressed_hd.sort(key=lambda x: -x[1])
n_hnscc = int((train_mean > 0.5).sum())
n_hd    = len(expressed_hd)

# ── fig2b: 가로 bar chart로 유전자 발현값 표시
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_facecolor('#F8F9FA')

genes = [g for g, v in expressed_hd]
vals  = [v for g, v in expressed_hd]
y_pos = range(len(genes))

bars = ax.barh(list(y_pos), vals, color=C2, edgecolor='white', height=0.6, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(list(y_pos))
ax.set_yticklabels(genes, fontsize=12)
ax.set_xlabel('Mean Expression (log-normalized)', fontsize=11)
ax.set_title('Visium HD: Genes with mean expr > 0.5\n(from HNSCC HVG 2000)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlim(0, max(vals) * 1.25)
ax.invert_yaxis()
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=2)
ax.grid(axis='x', alpha=0.3, zorder=1)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR + 'fig2b_expressed_table.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Done:', OUT_DIR + 'fig2b_expressed_table.png')