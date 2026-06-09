import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

# ── 경로 설정
GENE_LIST   = '/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt'
TRAIN_EXPRS = '/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01/train_exprs.npy'
TRAIN_EMBS  = '/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01/train_text_embs.npy'
VAL_EXPRS   = '/project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/tile_exprs.npy'
VAL_EMBS    = '/project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/tile_img_embs.npy'
COORDS      = '/project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01/tile_coords.npy'
POSITIONS   = '/project_antwerp/hbae/data/visium_hd_tonsil/binned_outputs/square_008um/spatial/tissue_positions.parquet'
OUT_DIR     = '/project_antwerp/hbae/script/0208_start/Visium_HD/plots/'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 스타일
C1, C2, C3 = '#2E75B6', '#ED7D31', '#70AD47'
GRAY = '#AAAAAA'
AX_BG = '#F8F9FA'
TITLE_KW = dict(fontsize=14, fontweight='bold', pad=12)

def save(fig, name):
    path = OUT_DIR + name
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {path}')

# ── 데이터 로드
print('Loading data...')
gene_names  = open(GENE_LIST).read().strip().split('\n')
train_exprs = np.load(TRAIN_EXPRS)
train_embs  = np.load(TRAIN_EMBS)
val_exprs   = np.load(VAL_EXPRS)
val_embs    = np.load(VAL_EMBS)
coords      = np.load(COORDS)

import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

train_t = torch.tensor(train_embs).float()
val_t   = torch.tensor(val_embs).float()
train_n = F.normalize(train_t, dim=-1)
val_n   = F.normalize(val_t, dim=-1)
train_e = torch.tensor(train_exprs).float()

print('Computing predictions...')
preds = []
for i in range(len(val_n)):
    sim = val_n[i] @ train_n.T
    idx = sim.topk(500).indices
    w   = sim[idx] / sim[idx].sum()
    preds.append((w[:, None] * train_e[idx]).sum(0).numpy())
preds = np.array(preds)

train_mean = train_exprs.mean(axis=0)
hd_mean    = val_exprs.mean(axis=0)

print('Computing gene-wise PCC...')
mean_expr  = val_exprs.mean(axis=0)
top300_idx = np.argsort(mean_expr)[::-1][:300]
gene_pccs  = []
for idx in top300_idx:
    if val_exprs[:, idx].std() > 1e-8:
        r, _ = pearsonr(preds[:, idx], val_exprs[:, idx])
        if np.isfinite(r):
            gene_pccs.append({
                'gene': gene_names[idx], 'pcc': r,
                'gt_mean': mean_expr[idx], 'pred_mean': preds[:, idx].mean(),
                'gt_std': val_exprs[:, idx].std(), 'pred_std': preds[:, idx].std(),
            })
gene_pccs = sorted(gene_pccs, key=lambda x: -x['pcc'])

print('Computing bin counts...')
df = pd.read_parquet(POSITIONS, engine='pyarrow')
df = df[df['in_tissue'] == 1].reset_index(drop=True)
scalef = 0.09202595
df['hires_col'] = df['pxl_col_in_fullres'] * scalef
df['hires_row'] = df['pxl_row_in_fullres'] * scalef
bin_col, bin_row = df['hires_col'].values, df['hires_row'].values
tile_size = 57
bin_counts = []
for col_start, row_start in coords:
    in_tile = (
        (bin_col >= col_start) & (bin_col < col_start + tile_size) &
        (bin_row >= row_start) & (bin_row < row_start + tile_size)
    )
    bin_counts.append(in_tile.sum())
bin_counts = np.array(bin_counts)

igkc_idx  = gene_names.index('IGKC')
igkc_gt   = val_exprs[:, igkc_idx]
igkc_pred = preds[:, igkc_idx]

col_uniq = np.unique(coords[:, 0])
row_uniq = np.unique(coords[:, 1])
col2i = {c: i for i, c in enumerate(col_uniq)}
row2i = {r: i for i, r in enumerate(row_uniq)}
grid_gt   = np.full((len(row_uniq), len(col_uniq)), np.nan)
grid_pred = np.full((len(row_uniq), len(col_uniq)), np.nan)
grid_bins = np.full((len(row_uniq), len(col_uniq)), np.nan)
for i, (col, row) in enumerate(coords):
    ci, ri = col2i[col], row2i[row]
    grid_gt[ri, ci]   = igkc_gt[i]
    grid_pred[ri, ci] = igkc_pred[i]
    grid_bins[ri, ci] = bin_counts[i]

# ════════════════════════════════════════════
# Figure 1: Top-K 유전자 겹침
# ════════════════════════════════════════════
print('\n[1] Top-K overlap...')
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor(AX_BG)
ks = [20, 50, 100, 200, 500, 1000]
overlaps, pcts = [], []
for k in ks:
    tk = set([gene_names[i] for i in np.argsort(train_mean)[::-1][:k]])
    hk = set([gene_names[i] for i in np.argsort(hd_mean)[::-1][:k]])
    ov = len(tk & hk)
    overlaps.append(ov)
    pcts.append(ov / k * 100)
bars = ax.bar([str(k) for k in ks], pcts, color=C1, edgecolor='white', width=0.6, zorder=3)
for bar, ov, pct in zip(bars, overlaps, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{ov} genes\n({pct:.0f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xlabel('Top-K Genes', fontsize=12)
ax.set_ylabel('Overlap (%)', fontsize=12)
ax.set_title('Top-K Gene Overlap: HNSCC train vs Visium HD Tonsil', **TITLE_KW)
ax.set_ylim(0, 80)
ax.axhline(50, color=GRAY, linestyle='--', linewidth=1, alpha=0.7)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'fig1_topk_overlap.png')

# ════════════════════════════════════════════
# Figure 2: 발현값 > 0.5 유전자 수 비교
# ════════════════════════════════════════════
print('[2] Expressed genes...')
expressed_hd = [(gene_names[i], hd_mean[i]) for i in range(len(gene_names)) if hd_mean[i] > 0.5]
expressed_hd.sort(key=lambda x: -x[1])
n_hnscc = int((train_mean > 0.5).sum())
n_hd    = len(expressed_hd)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6),
                                          gridspec_kw={'width_ratios': [1, 1.5]})
ax_left.set_facecolor(AX_BG)
bars2 = ax_left.bar(['HNSCC\ntrain', 'Visium HD\nTonsil'], [n_hnscc, n_hd],
                     color=[C1, C2], edgecolor='white', width=0.5, zorder=3)
for bar, v in zip(bars2, [n_hnscc, n_hd]):
    ax_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                 str(v), ha='center', va='bottom', fontsize=15, fontweight='bold')
ax_left.set_ylabel('# Genes (mean expr > 0.5)', fontsize=12)
ax_left.set_title('Meaningfully Expressed Genes\nin HNSCC HVG 2000', **TITLE_KW)
ax_left.set_ylim(0, 380)
ax_left.grid(axis='y', alpha=0.3)
ax_left.spines[['top','right']].set_visible(False)

ax_right.axis('off')
table_data = [[f'{i+1}', g, f'{v:.3f}'] for i, (g, v) in enumerate(expressed_hd)]
table = ax_right.table(
    cellText=table_data,
    colLabels=['#', 'Gene', 'Mean Expr'],
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(C1)
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#EEF4FB' if r % 2 == 0 else 'white')
    cell.set_edgecolor('white')
ax_right.set_title('Visium HD: Genes with mean expr > 0.5\n(from HNSCC HVG 2000)', **TITLE_KW)
fig.suptitle('Expressed Gene Comparison: HNSCC HVG in Visium HD Tonsil', fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'fig2_expressed_genes.png')

# ════════════════════════════════════════════
# Figure 3: 전체 발현 프로파일 상관관계
# ════════════════════════════════════════════
print('[3] Expression profile correlation...')
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_facecolor(AX_BG)
ax.scatter(train_mean, hd_mean, alpha=0.3, s=10, color=C1, zorder=3, label='All 2000 HVGs')
highlight = ['IGKC', 'S100A8', 'S100A9', 'KRT16', 'CD74', 'COL1A1', 'VIM', 'IGHG1', 'PTGDS', 'ZAP70']
for g in highlight:
    if g in gene_names:
        idx = gene_names.index(g)
        ax.scatter(train_mean[idx], hd_mean[idx], s=80, color=C2, zorder=5, edgecolors='white', linewidth=0.8)
        ax.annotate(g, (train_mean[idx], hd_mean[idx]),
                    textcoords='offset points', xytext=(6, 3), fontsize=9, color='#333333', fontweight='bold')
r_val, _ = pearsonr(train_mean, hd_mean)
ax.set_xlabel('HNSCC train — mean expression', fontsize=12)
ax.set_ylabel('Visium HD Tonsil — mean expression', fontsize=12)
ax.set_title(f'Gene Expression Profile Correlation\nPearson r = {r_val:.3f}  (2000 HVGs)', **TITLE_KW)
ax.text(0.05, 0.92, f'Pearson r = {r_val:.3f}\nSpearman r = 0.401',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY, alpha=0.9))
ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'fig3_expression_correlation.png')

# ════════════════════════════════════════════
# Figure 4: IGKC 공간 맵 (GT + Pred + scatter)
# ════════════════════════════════════════════
print('[4] IGKC spatial maps...')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
r_igkc, _ = pearsonr(igkc_pred, igkc_gt)

# GT 맵
im0 = axes[0].imshow(grid_gt, cmap='RdYlBu_r', aspect='auto',
                      vmin=np.nanmin(grid_gt), vmax=np.nanmax(grid_gt))
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Expression')
axes[0].set_title('IGKC: Ground Truth\n(Spatial map per tile)', **TITLE_KW)
axes[0].set_xlabel('Tile column', fontsize=11)
axes[0].set_ylabel('Tile row', fontsize=11)
axes[0].set_facecolor('black')

# Pred 맵
im1 = axes[1].imshow(grid_pred, cmap='RdYlBu_r', aspect='auto',
                      vmin=np.nanmin(grid_pred), vmax=np.nanmax(grid_pred))
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Expression')
axes[1].set_title('IGKC: Predicted\n(Spatial map per tile)', **TITLE_KW)
axes[1].set_xlabel('Tile column', fontsize=11)
axes[1].set_ylabel('Tile row', fontsize=11)
axes[1].set_facecolor('black')

# Scatter
axes[2].set_facecolor(AX_BG)
axes[2].scatter(igkc_gt, igkc_pred, alpha=0.4, s=12, color=C1, zorder=3)
lim = [min(igkc_gt.min(), igkc_pred.min())-0.2, max(igkc_gt.max(), igkc_pred.max())+0.2]
axes[2].plot(lim, lim, 'k--', linewidth=1, alpha=0.5)
axes[2].set_xlabel('GT expression', fontsize=12)
axes[2].set_ylabel('Predicted expression', fontsize=12)
axes[2].set_title(f'IGKC: GT vs Predicted\nGene-wise PCC = {r_igkc:.4f}', **TITLE_KW)
axes[2].text(0.05, 0.88,
             f'GT: mean={igkc_gt.mean():.2f}, std={igkc_gt.std():.2f}\n'
             f'Pred: mean={igkc_pred.mean():.2f}, std={igkc_pred.std():.2f}\n'
             f'GT zeros: {(igkc_gt==0).mean()*100:.1f}%\nGT >3.0: {(igkc_gt>3.0).mean()*100:.1f}%',
             transform=axes[2].transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY, alpha=0.9))
axes[2].grid(alpha=0.3)
axes[2].spines[['top','right']].set_visible(False)

fig.suptitle('IGKC Gene Expression: Ground Truth vs Predicted', fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'fig4_igkc_spatial.png')

# ════════════════════════════════════════════
# Figure 5: Gene-wise PCC 분포 + Top-20 bar
# ════════════════════════════════════════════
print('[5] Gene-wise PCC...')
fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(18, 7))
all_pccs = [x['pcc'] for x in gene_pccs]
n_pos = sum(1 for p in all_pccs if p > 0.0)
n_01  = sum(1 for p in all_pccs if p > 0.1)
n_02  = sum(1 for p in all_pccs if p > 0.2)

# 히스토그램
ax_hist.set_facecolor(AX_BG)
ax_hist.hist(all_pccs, bins=40, color=C1, edgecolor='white', linewidth=0.5, zorder=3)
ax_hist.axvline(0,   color='red', linestyle='--', linewidth=1.5, label='PCC = 0', zorder=4)
ax_hist.axvline(0.1, color=C2,   linestyle='--', linewidth=1.5, label='PCC = 0.1', zorder=4)
ax_hist.set_xlabel('Gene-wise PCC', fontsize=12)
ax_hist.set_ylabel('# Genes', fontsize=12)
ax_hist.set_title('Gene-wise PCC Distribution\n(top 300 expressed genes in Visium HD)', **TITLE_KW)
ax_hist.legend(fontsize=11)
ax_hist.text(0.60, 0.85,
             f'PCC > 0.1:  {n_01:3d} genes ({n_01/len(all_pccs)*100:.1f}%)\n'
             f'PCC > 0.0:  {n_pos:3d} genes ({n_pos/len(all_pccs)*100:.1f}%)\n'
             f'PCC ≤ 0.0:  {len(all_pccs)-n_pos:3d} genes ({(len(all_pccs)-n_pos)/len(all_pccs)*100:.1f}%)\n'
             f'Total:      {len(all_pccs):3d} genes',
             transform=ax_hist.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY, alpha=0.9))
ax_hist.grid(axis='y', alpha=0.3)
ax_hist.spines[['top','right']].set_visible(False)

# Top-20 bar
ax_bar.set_facecolor(AX_BG)
top20 = gene_pccs[:20]
genes_top = [x['gene'] for x in top20]
pccs_top  = [x['pcc'] for x in top20]
colors_bar = [C3 if p > 0.1 else C1 for p in pccs_top]
bars_h = ax_bar.barh(range(len(genes_top)), pccs_top, color=colors_bar,
                      edgecolor='white', linewidth=0.5, zorder=3)
ax_bar.set_yticks(range(len(genes_top)))
ax_bar.set_yticklabels(genes_top, fontsize=10)
ax_bar.set_xlabel('Gene-wise PCC', fontsize=12)
ax_bar.set_title('Top 20 Genes by Gene-wise PCC\n(green = PCC > 0.1)', **TITLE_KW)
ax_bar.axvline(0.1, color=C2,   linestyle='--', linewidth=1.2, alpha=0.8, zorder=2, label='PCC=0.1')
ax_bar.axvline(0.0, color=GRAY, linestyle='-',  linewidth=0.8, alpha=0.5, zorder=2)
for bar, pcc in zip(bars_h, pccs_top):
    ax_bar.text(pcc + 0.002, bar.get_y() + bar.get_height()/2,
                f'{pcc:.3f}', va='center', fontsize=9)
ax_bar.set_xlim(-0.01, 0.22)
ax_bar.invert_yaxis()
ax_bar.legend(fontsize=10)
ax_bar.grid(axis='x', alpha=0.3)
ax_bar.spines[['top','right']].set_visible(False)

fig.suptitle('Gene-wise PCC Analysis — Visium HD External Validation', fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'fig5_gene_pcc.png')

# ════════════════════════════════════════════
# Figure 6: 타일당 bin 수
# ════════════════════════════════════════════
print('[6] Bin counts...')
fig, (ax_hist2, ax_map) = plt.subplots(1, 2, figsize=(14, 6))

# 히스토그램
ax_hist2.set_facecolor(AX_BG)
ax_hist2.hist(bin_counts, bins=30, color=C1, edgecolor='white', zorder=3)
ax_hist2.axvline(bin_counts.mean(), color=C2,  linestyle='--', linewidth=1.5,
                  label=f'Mean = {bin_counts.mean():.1f}', zorder=4)
ax_hist2.axvline(bin_counts.min(), color='red', linestyle=':',  linewidth=1.5,
                  label=f'Min = {bin_counts.min()}', zorder=4)
ax_hist2.set_xlabel('# Bins per tile', fontsize=12)
ax_hist2.set_ylabel('# Tiles', fontsize=12)
ax_hist2.set_title(f'Bins per Tile Distribution\n(n = {len(bin_counts):,} tiles total)', **TITLE_KW)
ax_hist2.legend(fontsize=11)
ax_hist2.text(0.03, 0.80,
              f'Mean:   {bin_counts.mean():.1f}\nMedian: {np.median(bin_counts):.1f}\n'
              f'Min:    {bin_counts.min()}\nMax:    {bin_counts.max()}\nStd:    {bin_counts.std():.1f}',
              transform=ax_hist2.transAxes, fontsize=11,
              bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY, alpha=0.9))
ax_hist2.grid(axis='y', alpha=0.3)
ax_hist2.spines[['top','right']].set_visible(False)

# 공간 맵
ax_map.set_facecolor('black')
im_b = ax_map.imshow(grid_bins, cmap='viridis', aspect='auto')
plt.colorbar(im_b, ax=ax_map, fraction=0.046, pad=0.04, label='# Bins')
ax_map.set_title('Bins per Tile — Spatial Map\n', **TITLE_KW)
ax_map.set_xlabel('Tile column', fontsize=11)
ax_map.set_ylabel('Tile row', fontsize=11)

fig.suptitle('Tile Quality Check: Bins per Tile', fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'fig6_bin_counts.png')

# ════════════════════════════════════════════
# Figure 7: IGKC 발현 분포 비교
# ════════════════════════════════════════════
print('[7] IGKC distribution...')
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor(AX_BG)
igkc_train = train_exprs[:, igkc_idx]
ax.hist(igkc_train, bins=50, alpha=0.6, color=C1, label='HNSCC train spots', density=True, zorder=3)
ax.hist(igkc_gt,    bins=30, alpha=0.6, color=C2, label='Visium HD tiles (GT)', density=True, zorder=3)
ax.set_xlabel('IGKC expression (log-normalized)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('IGKC Expression Distribution\nHNSCC train vs Visium HD Tonsil', **TITLE_KW)
ax.legend(fontsize=11)
ax.text(0.55, 0.75,
        f'HNSCC:    zeros={((igkc_train==0).mean()*100):.1f}%\n'
        f'           >1.0={((igkc_train>1.0).mean()*100):.1f}%\n'
        f'           >3.0={((igkc_train>3.0).mean()*100):.1f}%\n\n'
        f'Visium HD: zeros={(igkc_gt==0).mean()*100:.1f}%\n'
        f'           >1.0={(igkc_gt>1.0).mean()*100:.1f}%\n'
        f'           >3.0={(igkc_gt>3.0).mean()*100:.1f}%',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY, alpha=0.9))
ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'fig7_igkc_distribution.png')

print('\nAll figures saved to:', OUT_DIR)