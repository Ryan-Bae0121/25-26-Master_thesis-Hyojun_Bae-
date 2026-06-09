"""
논문용 Table: K vs Gene-wise PCC (Tile-wise PCC ranking, 144px)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── 데이터 ────────────────────────────────────────────────
data = [
    ['50',   '0.4207', '0.4422', '0.7353'],
    ['100',  '0.4145', '0.4355', '0.7345'],
    ['200',  '0.4057', '0.4258', '0.7335'],
    ['300',  '0.3995', '0.4199', '0.7329'],
    ['500',  '0.3898', '0.4077', '0.7321'],
    ['1000', '0.3725', '0.3885', '0.7309'],
    ['All',  '0.0572', '0.0646', '0.7217'],
]
cols = ['K', 'Gene-wise PCC\n(Mean)', 'Gene-wise PCC\n(Median)', 'Slide-wise PCC\n(Mean)']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   10,
    'figure.dpi':  300,
})

fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.axis('off')

table = ax.table(
    cellText=data,
    colLabels=cols,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.7)

# 헤더 스타일
for j in range(len(cols)):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
    cell.set_edgecolor('white')

# 행 스타일
row_colors = ['#f8f9fa', '#ffffff']
best_row = 1  # K=50 (1-indexed, header=0)

for i in range(1, len(data) + 1):
    is_best = (i == best_row)
    is_all  = (i == len(data))
    for j in range(len(cols)):
        cell = table[i, j]
        if is_best:
            cell.set_facecolor('#fdecea')
            cell.set_text_props(fontweight='bold', color='#c0392b')
        elif is_all:
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(color='#666666', fontstyle='italic')
        else:
            cell.set_facecolor(row_colors[i % 2])
            cell.set_text_props(color='#2c3e50')
        cell.set_edgecolor('#dddddd')

# 제목
ax.set_title(
    'Table. Gene expression prediction performance by tile selection threshold (K)\n'
    'Tile-wise PCC ranking, 144 px tiles, fold_03',
    fontsize=10, pad=12, color='#2c3e50', loc='left'
)

# 각주
fig.text(0.01, 0.01,
         '† Bold (red): best performing K.  Italic: no tile selection baseline.',
         fontsize=8, color='#666666')

plt.tight_layout(pad=1.5)

out     = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/table_topk_gene_pcc.png'
out_pdf = out.replace('.png', '.pdf')
plt.savefig(out,     bbox_inches='tight', dpi=300)
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f'Saved:\n  {out}\n  {out_pdf}')