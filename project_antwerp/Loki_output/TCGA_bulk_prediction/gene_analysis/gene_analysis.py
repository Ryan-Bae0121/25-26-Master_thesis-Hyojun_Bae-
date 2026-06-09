python3 << 'EOF'
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr, ttest_ind, rankdata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/gene_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df['slide_id'] = ref_df['wsi_file_name'].apply(lambda x: x.split('.')[0])
rna_cols     = [c for c in ref_df.columns if c.startswith('rna_')]
ref_genes    = [c.replace('rna_', '') for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
common_idx   = [gene_list.index(g) for g in common_genes]
bulk_cols    = ['rna_' + g for g in common_genes]
G = len(common_genes)

train_embs = F.normalize(torch.tensor(
    np.load(f'{FT_EMB}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
train_expr = torch.tensor(
    np.load(f'{FT_EMB}/train_exprs.npy'), dtype=torch.float32, device=device)

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/{row["slide_id"]}.npy')]
print(f'Slides: {len(matched)}, Genes: {G}')

K = 3

# 전체 331 슬라이드에 대해 top-3, bot-3 tile pred 수집
print('Computing top-3/bot-3 gene expressions for all slides...')

top3_preds_all  = []  # (331, 1968) top-3 tile sum
bot3_preds_all  = []  # (331, 1968) bot-3 tile sum
all_preds_all   = []  # (331, 1968) all tile mean
bulk_all        = []  # (331, 1968) actual bulk
top3_scores_all = []  # (331,) top-3 mean PCC score
slide_ids       = []

for sid, bulk in matched:
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim        = torch.clamp(embs @ train_embs.T, min=0)
        weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()
    tp = tile_preds[:, common_idx]
    T  = len(tp)

    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
    scores = np.where(denom > 1e-8, num/denom, -999)

    valid  = scores > -999
    v_idx  = np.where(valid)[0]
    sorted_ = v_idx[np.argsort(scores[valid])[::-1]]
    top3   = sorted_[:K]
    bot3   = sorted_[-K:]

    top3_preds_all.append(tp[top3].sum(axis=0))
    bot3_preds_all.append(tp[bot3].sum(axis=0))
    all_preds_all.append(tp.mean(axis=0))
    bulk_all.append(bulk)
    top3_scores_all.append(scores[top3].mean())
    slide_ids.append(sid)
    del embs

torch.cuda.empty_cache()

top3_arr  = np.array(top3_preds_all)   # (331, 1968)
bot3_arr  = np.array(bot3_preds_all)   # (331, 1968)
all_arr   = np.array(all_preds_all)    # (331, 1968)
bulk_arr  = np.array(bulk_all)         # (331, 1968)
top3_scores = np.array(top3_scores_all)  # (331,)

print(f'Done. top3_arr: {top3_arr.shape}')

# ── 분석 1: Top-3 pred와 bulk의 gene-wise 상관관계 ────────
print('\n[Analysis 1] Gene-wise correlation between top-3 pred and bulk...')

gene_pcc_top3 = np.array([pearsonr(top3_arr[:,j], bulk_arr[:,j])[0]
                           for j in range(G)
                           if top3_arr[:,j].std()>1e-8 and bulk_arr[:,j].std()>1e-8])
gene_pcc_all  = np.array([pearsonr(all_arr[:,j],  bulk_arr[:,j])[0]
                           for j in range(G)
                           if all_arr[:,j].std()>1e-8  and bulk_arr[:,j].std()>1e-8])

# top/bottom 20 gene by PCC improvement
pcc_diff = gene_pcc_top3 - gene_pcc_all
top20_improved = np.argsort(pcc_diff)[::-1][:20]
top20_degraded = np.argsort(pcc_diff)[:20]

print(f'Gene-wise PCC (top-3): mean={gene_pcc_top3.mean():.4f}')
print(f'Gene-wise PCC (all):   mean={gene_pcc_all.mean():.4f}')
print(f'Top-20 most improved genes:')
for i in top20_improved[:10]:
    print(f'  {common_genes[i]:15s}: top3={gene_pcc_top3[i]:.4f}  all={gene_pcc_all[i]:.4f}  diff={pcc_diff[i]:+.4f}')

# ── 분석 2: Top-3 vs Bot-3 differential expression ────────
print('\n[Analysis 2] Differential expression: top-3 vs bot-3...')

# t-test: 각 gene에 대해 top-3 pred가 bot-3 pred보다 높은가
t_stats, p_vals = [], []
mean_diff = []
for j in range(G):
    t, p = ttest_ind(top3_arr[:,j], bot3_arr[:,j])
    t_stats.append(t)
    p_vals.append(p)
    mean_diff.append(top3_arr[:,j].mean() - bot3_arr[:,j].mean())

t_stats  = np.array(t_stats)
p_vals   = np.array(p_vals)
mean_diff = np.array(mean_diff)

# top-3에서 유의하게 높은 gene
sig_up   = np.where((p_vals < 0.05) & (mean_diff > 0))[0]
sig_down = np.where((p_vals < 0.05) & (mean_diff < 0))[0]
print(f'Significantly higher in top-3: {len(sig_up)} genes')
print(f'Significantly lower  in top-3: {len(sig_down)} genes')

# top-20 differentially expressed genes
top20_de = np.argsort(mean_diff)[::-1][:20]
print(f'\nTop-20 genes higher in Top-3 vs Bot-3:')
for i in top20_de[:15]:
    print(f'  {common_genes[i]:15s}: top3_mean={top3_arr[:,i].mean():.4f}  bot3_mean={bot3_arr[:,i].mean():.4f}  diff={mean_diff[i]:+.4f}  p={p_vals[i]:.2e}')

# ── 분석 3: Top-3 score vs gene expression 상관관계 ───────
print('\n[Analysis 3] Top-3 score vs gene expression correlation...')

# 슬라이드의 top-3 PCC score와 각 gene의 bulk expression 상관관계
score_gene_pcc = np.array([pearsonr(top3_scores, bulk_arr[:,j])[0]
                            for j in range(G)
                            if bulk_arr[:,j].std()>1e-8])

top20_score_corr = np.argsort(np.abs(score_gene_pcc))[::-1][:20]
print(f'Top-20 genes correlated with top-3 score:')
for i in top20_score_corr[:10]:
    print(f'  {common_genes[i]:15s}: score-gene PCC={score_gene_pcc[i]:.4f}')

# ── 시각화 ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 20))
fig.suptitle('Gene Expression Analysis: Top-3 Tile Characteristics (331 slides, fold_03)',
             fontsize=14)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Gene-wise PCC 분포 비교 (top-3 vs all)
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(gene_pcc_all,  bins=60, alpha=0.6, color='gray',  density=True, label=f'All tiles (mean={gene_pcc_all.mean():.4f})')
ax1.hist(gene_pcc_top3, bins=60, alpha=0.6, color='red',   density=True, label=f'Top-3 tiles (mean={gene_pcc_top3.mean():.4f})')
ax1.axvline(gene_pcc_all.mean(),  color='gray', linestyle='--')
ax1.axvline(gene_pcc_top3.mean(), color='red',  linestyle='--')
ax1.set_title('Gene-wise PCC Distribution: All tiles vs Top-3 tiles', fontsize=11)
ax1.set_xlabel('Gene-wise PCC')
ax1.set_ylabel('Density')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 2. PCC improvement scatter (top-3 - all)
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(gene_pcc_all, pcc_diff, alpha=0.3, s=5, color='steelblue')
ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('All tiles PCC')
ax2.set_ylabel('PCC improvement (top3 - all)')
ax2.set_title('PCC Improvement by Gene', fontsize=11)
ax2.grid(alpha=0.3)

# 상위 10개 gene 이름 표시
for i in top20_improved[:8]:
    ax2.annotate(common_genes[i], (gene_pcc_all[i], pcc_diff[i]),
                fontsize=6, color='red', alpha=0.8)

# 3. Top-20 DE genes (top-3 vs bot-3)
ax3 = fig.add_subplot(gs[1, :])
top15_genes = [common_genes[i] for i in top20_de[:15]]
top15_diff  = [mean_diff[i] for i in top20_de[:15]]
top15_top3  = [top3_arr[:,i].mean() for i in top20_de[:15]]
top15_bot3  = [bot3_arr[:,i].mean() for i in top20_de[:15]]
top15_bulk  = [bulk_arr[:,i].mean()  for i in top20_de[:15]]

x = np.arange(15)
w = 0.25
ax3.bar(x - w, top15_bulk, w, label='Bulk (mean)', color='black', alpha=0.8)
ax3.bar(x,     top15_top3, w, label='Top-3 pred', color='red',   alpha=0.7)
ax3.bar(x + w, top15_bot3, w, label='Bot-3 pred', color='blue',  alpha=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels(top15_genes, rotation=45, ha='right', fontsize=9)
ax3.set_title('Top-15 DE Genes: Higher in Top-3 vs Bot-3', fontsize=11)
ax3.set_ylabel('Mean expression (across 331 slides)')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# 4. Top-3 score 분포
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(top3_scores, bins=40, color='#e74c3c', alpha=0.8)
ax4.axvline(top3_scores.mean(), color='black', linestyle='--',
            label=f'mean={top3_scores.mean():.4f}')
ax4.set_title(f'Top-3 PCC Score Distribution\n(331 slides)', fontsize=10)
ax4.set_xlabel('Mean top-3 PCC score')
ax4.set_ylabel('Count')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# 5. Top-3 score vs bulk expression (대표 gene)
ax5 = fig.add_subplot(gs[2, 1])
best_corr_gene = top20_score_corr[0]
ax5.scatter(top3_scores, bulk_arr[:, best_corr_gene], alpha=0.4, s=10, color='purple')
r, _ = pearsonr(top3_scores, bulk_arr[:, best_corr_gene])
ax5.set_title(f'Top-3 Score vs {common_genes[best_corr_gene]}\nr={r:.4f}', fontsize=10)
ax5.set_xlabel('Top-3 PCC score')
ax5.set_ylabel(f'{common_genes[best_corr_gene]} expression (bulk)')
ax5.grid(alpha=0.3)

# 6. Gene PCC improvement top-20 bar
ax6 = fig.add_subplot(gs[2, 2])
top10_names = [common_genes[i] for i in top20_improved[:10]]
top10_diff  = [pcc_diff[i]     for i in top20_improved[:10]]
colors_bar  = ['#e74c3c' if d > 0 else '#3498db' for d in top10_diff]
ax6.barh(top10_names[::-1], top10_diff[::-1], color=colors_bar[::-1], alpha=0.8)
ax6.axvline(0, color='black', linestyle='--', alpha=0.5)
ax6.set_title('Top-10 PCC Improved Genes\n(top-3 vs all tiles)', fontsize=10)
ax6.set_xlabel('PCC improvement')
ax6.grid(axis='x', alpha=0.3)

plt.savefig(f'{OUT_DIR}/gene_expression_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: gene_expression_analysis.png')

# ── 결과 저장 ──────────────────────────────────────────────
pd.DataFrame({
    'gene':         common_genes,
    'pcc_top3':     gene_pcc_top3,
    'pcc_all':      gene_pcc_all,
    'pcc_diff':     pcc_diff,
    'mean_diff_top3_bot3': mean_diff,
    'tstat':        t_stats,
    'pval':         p_vals,
    'score_gene_corr': score_gene_pcc,
}).sort_values('pcc_diff', ascending=False).to_csv(
    f'{OUT_DIR}/gene_analysis_results.csv', index=False)

print(f'Saved: gene_analysis_results.csv')
print('Done!')
EOF