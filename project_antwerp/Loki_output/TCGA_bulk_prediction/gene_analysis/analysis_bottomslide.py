python3 << 'EOF'
import numpy as np, pandas as pd
from scipy.stats import pearsonr, ttest_ind
import matplotlib.pyplot as plt
import os

GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/gene_analysis'

with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df['slide_id'] = ref_df['wsi_file_name'].apply(lambda x: x.split('.')[0])
rna_cols     = [c for c in ref_df.columns if c.startswith('rna_')]
ref_genes    = [c.replace('rna_', '') for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
bulk_cols    = ['rna_' + g for g in common_genes]
G = len(common_genes)

import torch, torch.nn.functional as F
device = 'cuda'
TCGA_EMB = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
FT_EMB   = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
common_idx = [gene_list.index(g) for g in common_genes]

train_embs = F.normalize(torch.tensor(
    np.load(f'{FT_EMB}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
train_expr = torch.tensor(
    np.load(f'{FT_EMB}/train_exprs.npy'), dtype=torch.float32, device=device)

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/{row["slide_id"]}.npy')]

K = 3
top3_scores, bulk_list, slide_ids = [], [], []

for sid, bulk in matched:
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim   = torch.clamp(embs @ train_embs.T, min=0)
        w     = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tp    = (w @ train_expr).cpu().numpy()[:, common_idx]
    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
    scores = np.where(denom > 1e-8, num/denom, -999)
    valid  = scores > -999
    v_idx  = np.where(valid)[0]
    top3   = v_idx[np.argsort(scores[valid])[::-1][:K]]
    top3_scores.append(scores[top3].mean())
    bulk_list.append(bulk)
    slide_ids.append(sid)
    del embs

torch.cuda.empty_cache()
top3_scores = np.array(top3_scores)
bulk_arr    = np.array(bulk_list)

p75 = np.percentile(top3_scores, 75)
p25 = np.percentile(top3_scores, 25)
high_idx = np.where(top3_scores >= p75)[0]
low_idx  = np.where(top3_scores <= p25)[0]
high_bulk = bulk_arr[high_idx]
low_bulk  = bulk_arr[low_idx]

from scipy.stats import ttest_ind
t_stats, p_vals, mean_diffs = [], [], []
for j in range(G):
    t, p = ttest_ind(high_bulk[:,j], low_bulk[:,j])
    t_stats.append(t); p_vals.append(p)
    mean_diffs.append(high_bulk[:,j].mean() - low_bulk[:,j].mean())
t_stats = np.array(t_stats); p_vals = np.array(p_vals); mean_diffs = np.array(mean_diffs)

top20_up   = np.argsort(mean_diffs)[::-1][:20]
top20_down = np.argsort(mean_diffs)[:20]

# ── Figure 1: Volcano + DE bar ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('High vs Low Top-3 Score Slides: DE Gene Analysis\n'
             f'High (top 25%, score≥{p75:.4f}, n={len(high_idx)}) vs '
             f'Low (bot 25%, score≤{p25:.4f}, n={len(low_idx)})', fontsize=12)

# Volcano
ax1 = axes[0]
neg_log_p = -np.log10(p_vals + 1e-300)
colors_v  = ['#e74c3c' if (p<0.01 and d>0.5) else
             '#3498db' if (p<0.01 and d<-0.5) else 'lightgray'
             for p, d in zip(p_vals, mean_diffs)]
ax1.scatter(mean_diffs, neg_log_p, c=colors_v, s=5, alpha=0.6)
ax1.axhline(-np.log10(0.01), color='black', linestyle='--', alpha=0.5, label='p=0.01')
ax1.axvline(0.5,  color='red',  linestyle='--', alpha=0.4, label='|diff|=0.5')
ax1.axvline(-0.5, color='blue', linestyle='--', alpha=0.4)
for i in list(top20_up[:8]) + list(top20_down[:8]):
    if p_vals[i] < 0.001 and abs(mean_diffs[i]) > 1.0:
        ax1.annotate(common_genes[i], (mean_diffs[i], neg_log_p[i]),
                    fontsize=7, alpha=0.9,
                    color='red' if mean_diffs[i]>0 else 'blue',
                    xytext=(3,3), textcoords='offset points')
ax1.set_xlabel('Mean difference (High - Low)', fontsize=11)
ax1.set_ylabel('-log10(p-value)', fontsize=11)
ax1.set_title(f'Volcano Plot\nUp in High: {(mean_diffs>0.5).sum()} | Down: {(mean_diffs<-0.5).sum()}', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# DE bar
ax2 = axes[1]
top10_up_names   = [common_genes[i] for i in top20_up[:10]]
top10_up_diff    = [mean_diffs[i]   for i in top20_up[:10]]
top10_down_names = [common_genes[i] for i in top20_down[:10]]
top10_down_diff  = [mean_diffs[i]   for i in top20_down[:10]]
all_names = top10_up_names + top10_down_names
all_diffs = top10_up_diff  + top10_down_diff
bar_c     = ['#e74c3c']*10 + ['#3498db']*10
ax2.barh(all_names[::-1], all_diffs[::-1], color=bar_c[::-1], alpha=0.8)
ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
ax2.set_title('Top-10 DE Genes\nred=High>Low | blue=Low>High', fontsize=10)
ax2.set_xlabel('Mean diff (High - Low)')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/de_genes_high_vs_low.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: de_genes_high_vs_low.png')

# ── Figure 2: KRT6B vs COL1A1 scatter ────────────────────
krt6b_i  = next(i for i, g in enumerate(common_genes) if g=='KRT6B')
col1a1_i = next(i for i, g in enumerate(common_genes) if g=='COL1A1')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('High vs Low Score Slides: KRT6B, COL1A1, Score Distribution', fontsize=12)

# Score distribution
ax0 = axes[0]
ax0.hist(top3_scores[low_idx],  bins=25, color='#3498db', alpha=0.7, label=f'Low (n={len(low_idx)})', density=True)
ax0.hist(top3_scores[high_idx], bins=25, color='#e74c3c', alpha=0.7, label=f'High (n={len(high_idx)})', density=True)
ax0.axvline(p25, color='blue', linestyle='--', alpha=0.7)
ax0.axvline(p75, color='red',  linestyle='--', alpha=0.7)
ax0.set_title('Top-3 Score Distribution', fontsize=10)
ax0.set_xlabel('Top-3 PCC score'); ax0.set_ylabel('Density')
ax0.legend(fontsize=9); ax0.grid(alpha=0.3)

# KRT6B vs COL1A1
ax1 = axes[1]
ax1.scatter(bulk_arr[high_idx, krt6b_i], bulk_arr[high_idx, col1a1_i],
           c='#e74c3c', s=20, alpha=0.6, label=f'High score (n={len(high_idx)})')
ax1.scatter(bulk_arr[low_idx,  krt6b_i], bulk_arr[low_idx,  col1a1_i],
           c='#3498db', s=20, alpha=0.6, label=f'Low score (n={len(low_idx)})')
ax1.set_xlabel('KRT6B (bulk)', fontsize=10); ax1.set_ylabel('COL1A1 (bulk)', fontsize=10)
ax1.set_title('KRT6B vs COL1A1\nHigh vs Low score slides', fontsize=10)
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
r, _ = pearsonr(bulk_arr[:,krt6b_i], bulk_arr[:,col1a1_i])
ax1.text(0.05, 0.95, f'r(KRT6B,COL1A1)={r:.3f}', transform=ax1.transAxes, fontsize=8)

# Top-3 score vs KRT6B and COL1A1
ax2 = axes[2]
ax2.scatter(top3_scores, bulk_arr[:,krt6b_i],  c='#e74c3c', s=8, alpha=0.5, label='KRT6B')
ax2.scatter(top3_scores, bulk_arr[:,col1a1_i], c='#3498db', s=8, alpha=0.5, label='COL1A1')
r_krt, _ = pearsonr(top3_scores, bulk_arr[:,krt6b_i])
r_col, _ = pearsonr(top3_scores, bulk_arr[:,col1a1_i])
ax2.set_xlabel('Top-3 PCC score', fontsize=10)
ax2.set_ylabel('Bulk expression', fontsize=10)
ax2.set_title(f'Top-3 Score vs KRT6B/COL1A1\nr(KRT6B)={r_krt:.3f}  r(COL1A1)={r_col:.3f}', fontsize=10)
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/krt6b_col1a1_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: krt6b_col1a1_scatter.png')

# ── Figure 3: High vs Low bulk expression profile ────────
fig, ax = plt.subplots(figsize=(16, 6))
top15_up_idx   = top20_up[:15]
top10_down_idx = top20_down[:10]
all_show = list(top15_up_idx) + list(top10_down_idx)
names    = [common_genes[i] for i in all_show]
x        = np.arange(len(all_show))
w        = 0.35
ax.bar(x - w/2, [high_bulk[:,i].mean() for i in all_show], w,
       label=f'High score (n={len(high_idx)})', color='#e74c3c', alpha=0.8)
ax.bar(x + w/2, [low_bulk[:,i].mean()  for i in all_show], w,
       label=f'Low  score (n={len(low_idx)})',  color='#3498db', alpha=0.8)
ax.axvline(14.5, color='black', linestyle='--', alpha=0.5)
ax.text(7, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else 8,
        'Higher in HIGH →', ha='center', fontsize=9, color='#e74c3c')
ax.text(18, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else 8,
        '← Higher in LOW', ha='center', fontsize=9, color='#3498db')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_title('Bulk Gene Expression: High vs Low Top-3 Score Slides', fontsize=12)
ax.set_ylabel('Mean bulk expression')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/high_vs_low_bulk_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: high_vs_low_bulk_profile.png')
print('\nDone!')
EOF