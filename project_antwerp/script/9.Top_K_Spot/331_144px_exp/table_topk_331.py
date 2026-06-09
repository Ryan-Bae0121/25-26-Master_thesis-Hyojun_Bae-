"""
논문용 Table: K vs Gene-wise PCC (331 slides, tile-wise PCC ranking)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = [
    ['50',   '0.4582', '0.4804', '0.7570'],
    ['100',  '0.4483', '0.4699', '0.7552'],
    ['200',  '0.4374', '0.4585', '0.7532'],
    ['300',  '0.4308', '0.4516', '0.7520'],
    ['500',  '0.4177', '0.4379', '0.7499'],
    ['1000', '0.3973', '0.4165', '0.7468'],
    ['All',  '0.0373', '0.0451', '0.7156'],
]
cols = ['K', 'Gene-wise PCC\n(Mean)', 'Gene-wise PCC\n(Median)', 'Slide-wise PCC\n(Mean)']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   10,
    'figure.dpi':  300,
})
fig, ax = plt.subplots(figsize=(7.5, 3.5))
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

for j in range(len(cols)):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
    cell.set_edgecolor('white')

row_colors = ['#f8f9fa', '#ffffff']
best_row   = 1  # K=50
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

ax.set_title(
    'Table. Gene expression prediction performance by tile selection threshold (K)\n'
    'Tile-wise PCC ranking, 144 px tiles, fold_03, 331 patient-level slides',
    fontsize=10, pad=12, color='#2c3e50', loc='left'
)
fig.text(0.01, 0.01,
         '† Bold (red): best performing K.  Italic: no tile selection baseline.',
         fontsize=8, color='#666666')
plt.tight_layout(pad=1.5)
out     = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/table_topk_gene_pcc_331.png'
out_pdf = out.replace('.png', '.pdf')
plt.savefig(out,     bbox_inches='tight', dpi=300)
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f'Saved:\n  {out}\n  {out_pdf}')