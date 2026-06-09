import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import os

OUT_DIR = '/project_antwerp/hbae/script/0208_start/visium_hd_array_embeddings/plots/' 
os.makedirs(OUT_DIR, exist_ok=True)

GENE_LIST   = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
TRAIN_EMBS  = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01/train_text_embs.npy'
TRAIN_EXPRS = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01/train_exprs.npy'
VAL_EMBS    = '/project_antwerp/hbae/Loki_output/visium_hd_array_embeddings/fold_01/tile_img_embs.npy'
VAL_EXPRS   = '/project_antwerp/hbae/Loki_output/visium_hd_array_embeddings/fold_01/tile_exprs.npy'

C1, C2, C3 = '#2E75B6', '#ED7D31', '#70AD47'
GRAY = '#AAAAAA'
AX_BG = '#F8F9FA'

print('Loading...')
gene_names  = open(GENE_LIST).read().strip().split('\n')
train_embs  = torch.tensor(np.load(TRAIN_EMBS)).float()
train_exprs = torch.tensor(np.load(TRAIN_EXPRS)).float()
val_embs    = torch.tensor(np.load(VAL_EMBS)).float()
val_exprs   = np.load(VAL_EXPRS)

train_norm = F.normalize(train_embs, dim=-1)
val_norm   = F.normalize(val_embs, dim=-1)

print('Computing predictions...')
preds = []
for i in range(len(val_norm)):
    sim = val_norm[i] @ train_norm.T
    idx = sim.topk(500).indices
    w   = sim[idx] / sim[idx].sum()
    preds.append((w[:, None] * train_exprs[idx]).sum(0).numpy())
preds = np.array(preds)

print('Computing gene-wise PCC...')
gene_pcc_list = []
for g in range(preds.shape[1]):
    if val_exprs[:, g].std() > 1e-8 and preds[:, g].std() > 1e-8:
        r, _ = pearsonr(preds[:, g], val_exprs[:, g])
        if np.isfinite(r):
            gene_pcc_list.append({
                'gene': gene_names[g],
                'pcc': r,
                'gt_mean': val_exprs[:, g].mean(),
                'gt_std': val_exprs[:, g].std(),
                'pred_std': preds[:, g].std(),
            })

gene_pcc_list.sort(key=lambda x: -x['pcc'])
all_pccs = [x['pcc'] for x in gene_pcc_list]
print(f'평가된 유전자: {len(all_pccs)}개')

n_pos  = sum(1 for p in all_pccs if p > 0)
n_01   = sum(1 for p in all_pccs if p > 0.1)
n_neg  = sum(1 for p in all_pccs if p < 0)

# ── Figure 1: PCC 분포 히스토그램
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor(AX_BG)
ax.hist(all_pccs, bins=60, color=C1, edgecolor='white', linewidth=0.5, zorder=3)
ax.axvline(0,   color='red', linestyle='--', linewidth=1.5, label='PCC=0', zorder=4)
ax.axvline(0.1, color=C2,   linestyle='--', linewidth=1.5, label='PCC=0.1', zorder=4)
ax.set_xlabel('Gene-wise PCC', fontsize=13)
ax.set_ylabel('# Genes', fontsize=13)
ax.set_title('Gene-wise PCC Distribution\n(HVG 2000개 전체 평가, array 9×9 기반)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.text(0.62, 0.85,
        f'PCC > 0.1:  {n_01:4d} genes ({n_01/len(all_pccs)*100:.1f}%)\n'
        f'PCC > 0.0:  {n_pos:4d} genes ({n_pos/len(all_pccs)*100:.1f}%)\n'
        f'PCC < 0.0:  {n_neg:4d} genes ({n_neg/len(all_pccs)*100:.1f}%)\n'
        f'Mean PCC:   {np.mean(all_pccs):.4f}\n'
        f'Total:      {len(all_pccs):4d} genes',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY, alpha=0.9))
ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR + 'fig_all2000_pcc_dist.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved: fig_all2000_pcc_dist.png')

# ── Figure 2: Top 30 gene PCC bar chart
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor(AX_BG)
top30 = gene_pcc_list[:30]
genes = [x['gene'] for x in top30]
pccs  = [x['pcc'] for x in top30]
means = [x['gt_mean'] for x in top30]
colors = [C3 if p > 0.1 else C1 for p in pccs]
bars = ax.barh(range(len(genes)), pccs, color=colors, edgecolor='white', height=0.7, zorder=3)
ax.set_yticks(range(len(genes)))
ax.set_yticklabels(genes, fontsize=10)
ax.set_xlabel('Gene-wise PCC', fontsize=12)
ax.set_title('Top 30 Genes by Gene-wise PCC\n(HVG 2000개 전체 평가, green=PCC>0.1)', fontsize=13, fontweight='bold')
ax.axvline(0.1, color=C2, linestyle='--', linewidth=1.2, alpha=0.8, label='PCC=0.1', zorder=2)
for bar, pcc, m in zip(bars, pccs, means):
    ax.text(pcc + 0.001, bar.get_y() + bar.get_height()/2,
            f'{pcc:.3f} (mean={m:.2f})', va='center', fontsize=8.5)
ax.set_xlim(-0.01, max(pccs) * 1.35)
ax.invert_yaxis()
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR + 'fig_all2000_top30.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved: fig_all2000_top30.png')

# ── Figure 3: PCC vs mean expression scatter
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_facecolor(AX_BG)
gt_means = [x['gt_mean'] for x in gene_pcc_list]
ax.scatter(gt_means, all_pccs, alpha=0.3, s=8, color=C1, zorder=3)
# 상위 10개 하이라이트
for x in gene_pcc_list[:10]:
    ax.scatter(x['gt_mean'], x['pcc'], s=60, color=C2, zorder=5, edgecolors='white')
    ax.annotate(x['gene'], (x['gt_mean'], x['pcc']),
                textcoords='offset points', xytext=(5, 3), fontsize=8, fontweight='bold')
ax.axhline(0,   color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(0.1, color=C2,   linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Visium HD mean expression', fontsize=12)
ax.set_ylabel('Gene-wise PCC', fontsize=12)
ax.set_title('Gene-wise PCC vs Mean Expression\n(HVG 2000개)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR + 'fig_all2000_pcc_vs_expr.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved: fig_all2000_pcc_vs_expr.png')

# ── Figure 4: Bottom 20 (가장 안 되는 유전자)
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor(AX_BG)
bot20 = gene_pcc_list[-20:]
genes_b = [x['gene'] for x in bot20]
pccs_b  = [x['pcc'] for x in bot20]
means_b = [x['gt_mean'] for x in bot20]
bars_b = ax.barh(range(len(genes_b)), pccs_b,
                  color=['#FF6B6B' if p < 0 else C1 for p in pccs_b],
                  edgecolor='white', height=0.7, zorder=3)
ax.set_yticks(range(len(genes_b)))
ax.set_yticklabels(genes_b, fontsize=10)
ax.set_xlabel('Gene-wise PCC', fontsize=12)
ax.set_title('Bottom 20 Genes by Gene-wise PCC\n(가장 예측 안 되는 유전자)', fontsize=13, fontweight='bold')
ax.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
for bar, pcc, m in zip(bars_b, pccs_b, means_b):
    ax.text(pcc - 0.001 if pcc < 0 else pcc + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{pcc:.3f} (mean={m:.2f})',
            va='center', ha='right' if pcc < 0 else 'left', fontsize=8.5)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR + 'fig_all2000_bottom20.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved: fig_all2000_bottom20.png')

print(f'\n모든 plot 저장 완료: {OUT_DIR}')