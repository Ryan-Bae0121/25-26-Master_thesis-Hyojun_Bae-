import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

GENE_LIST   = '/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt'
TRAIN_EXPRS = '/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01/train_exprs.npy'
VAL_EXPRS   = '/project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/tile_exprs.npy'
OUT_DIR     = '/project_antwerp/hbae/script/0208_start/Visium_HD/plots/'

C1, C2 = '#2E75B6', '#ED7D31'
GRAY = '#AAAAAA'
AX_BG = '#F8F9FA'

gene_names  = open(GENE_LIST).read().strip().split('\n')
train_exprs = np.load(TRAIN_EXPRS)
val_exprs   = np.load(VAL_EXPRS)

train_mean = train_exprs.mean(axis=0)
hd_mean    = val_exprs.mean(axis=0)

# 발현값 > 0.5 유전자
expressed_hd = [(gene_names[i], float(hd_mean[i]))
                for i in range(len(gene_names)) if hd_mean[i] > 0.5]
expressed_hd.sort(key=lambda x: -x[1])
n_hnscc = int((train_mean > 0.5).sum())
n_hd    = len(expressed_hd)

print(f'HNSCC: {n_hnscc}, Visium HD: {n_hd}')
print('Visium HD expressed:', expressed_hd)

# ── 두 개의 figure로 나눠서 저장

# Figure 2a: bar chart
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_facecolor(AX_BG)
bars = ax.bar(['HNSCC\ntrain', 'Visium HD\nTonsil'],
              [n_hnscc, n_hd],
              color=[C1, C2], edgecolor='white', width=0.45, zorder=3)
for bar, v in zip(bars, [n_hnscc, n_hd]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 3,
            str(v), ha='center', va='bottom',
            fontsize=16, fontweight='bold')
ax.set_ylabel('# Genes (mean expr > 0.5)', fontsize=12)
ax.set_title('Meaningfully Expressed Genes\nin HNSCC HVG 2000\n(mean expr > 0.5)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_ylim(0, 380)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR + 'fig2a_expressed_bar.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved fig2a')

# Figure 2b: gene table
fig, ax = plt.subplots(figsize=(5, 5))
ax.axis('off')

table_data = [[f'{i+1}', g, f'{v:.3f}']
              for i, (g, v) in enumerate(expressed_hd)]
table = ax.table(
    cellText=table_data,
    colLabels=['#', 'Gene', 'Mean Expr'],
    cellLoc='center',
    loc='center',
    bbox=[0.0, 0.0, 1.0, 1.0]
)
table.auto_set_font_size(False)
table.set_fontsize(13)
for (r, c), cell in table.get_celld().items():
    cell.set_height(0.11)
    if r == 0:
        cell.set_facecolor(C1)
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#EEF4FB' if r % 2 == 0 else 'white')
    cell.set_edgecolor('#DDDDDD')

ax.set_title('Visium HD: Genes with mean expr > 0.5\n(from HNSCC HVG 2000)',
             fontsize=13, fontweight='bold', pad=12)
fig.tight_layout()
fig.savefig(OUT_DIR + 'fig2b_expressed_table.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved fig2b')