"""
논문용 Figure: K vs Gene-wise PCC (331 slides, tile-wise PCC ranking)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

K_vals    = [50, 100, 200, 300, 500, 1000]
gene_mean = [0.4582, 0.4483, 0.4374, 0.4308, 0.4177, 0.3973]
gene_med  = [0.4804, 0.4699, 0.4585, 0.4516, 0.4379, 0.4165]
all_mean  = 0.0373

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.size':  4,
    'ytick.major.size':  4,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'figure.dpi':        300,
})
fig, ax = plt.subplots(figsize=(5.5, 4.2))

ax.axhline(all_mean, color='#888888', lw=1.2, ls='--', zorder=1,
           label='All tiles (no selection)')
ax.plot(K_vals, gene_med, color='#2c7bb6', lw=1.5, ls='--',
        marker='s', markersize=5, markerfacecolor='white',
        markeredgewidth=1.5, zorder=3, label='Median gene-wise PCC')
ax.plot(K_vals, gene_mean, color='#d7191c', lw=2.0,
        marker='o', markersize=6, markerfacecolor='white',
        markeredgewidth=2.0, zorder=4, label='Mean gene-wise PCC')

ax.scatter([50], [0.4582], s=80, color='#d7191c', zorder=5)
ax.annotate('K = 50\n(mean = 0.458)', xy=(50, 0.4582),
            xytext=(150, 0.470),
            fontsize=9, color='#d7191c',
            arrowprops=dict(arrowstyle='->', color='#d7191c', lw=1.2),
            ha='left')
ax.text(1050, all_mean + 0.005, 'All tiles\n(0.037)',
        fontsize=8.5, color='#666666', va='bottom', ha='right')

ax.set_xlabel('Number of selected tiles (K)', fontsize=11)
ax.set_ylabel('Gene-wise Pearson Correlation', fontsize=11)
ax.set_xlim(0, 1100)
ax.set_ylim(0.00, 0.52)
ax.set_xticks(K_vals)
ax.set_xticklabels([str(k) for k in K_vals], fontsize=9)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.legend(fontsize=8.5, frameon=False, loc='upper right')
ax.axvspan(0, 200, alpha=0.04, color='#d7191c', zorder=0)

plt.tight_layout(pad=1.2)
out     = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/fig_topk_gene_pcc_331.pdf'
out_png = out.replace('.pdf', '.png')
plt.savefig(out,     bbox_inches='tight')
plt.savefig(out_png, bbox_inches='tight', dpi=300)
plt.close()
print(f'Saved:\n  {out}\n  {out_png}')