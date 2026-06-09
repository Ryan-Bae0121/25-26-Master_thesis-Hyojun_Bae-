python3 << 'EOF'
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings'
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

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/fold_01/{row["slide_id"]}.npy')]
print(f'Slides: {len(matched)}, Genes: {G}')

# bulk 전체 로드
bulk_arr = np.array([b for _, b in matched])  # (331, 1968)

FOLDS = [f'fold_{i:02d}' for i in range(1, 11)]
K = 3

# 분석할 gene들: fold_03에서 상관관계 높았던 것들
target_genes = ['KRT6B', 'KRT16', 'SPRR1B', 'S100A9', 'S100A8',
                'CSTA', 'SPRR3', 'S100A12', 'COL1A1', 'VCAN']
target_idx   = {g: i for i, g in enumerate(common_genes) if g in target_genes}
print(f'Target genes found: {list(target_idx.keys())}')

# fold별 top-3 score와 각 gene의 상관관계
results = {}  # {fold: {gene: r}}
fold_top3_scores = {}  # {fold: (331,) top-3 mean PCC score}

for fold in FOLDS:
    print(f'\n[{fold}]')
    train_embs = F.normalize(torch.tensor(
        np.load(f'{FT_EMB}/{fold}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
    train_expr = torch.tensor(
        np.load(f'{FT_EMB}/{fold}/train_exprs.npy'), dtype=torch.float32, device=device)

    top3_scores_fold = []

    for sid, bulk in matched:
        embs = F.normalize(torch.tensor(
            np.load(f'{TCGA_EMB}/{fold}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
        with torch.no_grad():
            sim        = torch.clamp(embs @ train_embs.T, min=0)
            weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
            tile_preds = (weights @ train_expr).cpu().numpy()
        tp = tile_preds[:, common_idx]

        bulk_c = bulk - bulk.mean()
        tile_c = tp - tp.mean(axis=1, keepdims=True)
        num    = (tile_c * bulk_c).sum(axis=1)
        denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
        scores = np.where(denom > 1e-8, num/denom, -999)

        valid  = scores > -999
        v_idx  = np.where(valid)[0]
        top3   = v_idx[np.argsort(scores[valid])[::-1][:K]]
        top3_scores_fold.append(scores[top3].mean())
        del embs

    del train_embs, train_expr
    torch.cuda.empty_cache()

    top3_scores_arr = np.array(top3_scores_fold)
    fold_top3_scores[fold] = top3_scores_arr

    # 각 gene과 top-3 score의 상관관계
    fold_results = {}
    for gene, g_idx in target_idx.items():
        r, p = pearsonr(top3_scores_arr, bulk_arr[:, g_idx])
        fold_results[gene] = (r, p)
        print(f'  {gene:12s}: r={r:.4f}  p={p:.2e}')
    results[fold] = fold_results

# ── 요약 표 ────────────────────────────────────────────────
print('\n' + '='*70)
print('Fold별 Top-3 Score vs Gene Correlation 요약')
print('='*70)

for gene in target_genes:
    if gene not in target_idx:
        continue
    rs = [results[f][gene][0] for f in FOLDS if gene in results[f]]
    print(f'{gene:12s}: ' + '  '.join([f'{r:.3f}' for r in rs]) +
          f'  →  mean={np.mean(rs):.4f}  std={np.std(rs):.4f}')

# ── 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Top-3 PCC Score vs Gene Expression Correlation\n(All 10 Folds)', fontsize=13)

# 1. Gene별 fold-across correlation (heatmap 스타일)
ax1 = axes[0, 0]
corr_matrix = []
gene_names  = [g for g in target_genes if g in target_idx]
for gene in gene_names:
    rs = [results[f][gene][0] for f in FOLDS if gene in results[f]]
    corr_matrix.append(rs)
corr_matrix = np.array(corr_matrix)  # (n_genes, 10)

im = ax1.imshow(corr_matrix, cmap='RdYlGn', vmin=-0.5, vmax=0.9, aspect='auto')
plt.colorbar(im, ax=ax1)
ax1.set_xticks(range(10))
ax1.set_xticklabels([f'f{i:02d}' for i in range(1,11)], fontsize=8)
ax1.set_yticks(range(len(gene_names)))
ax1.set_yticklabels(gene_names, fontsize=9)
ax1.set_title('Correlation Heatmap\n(Top-3 score vs bulk gene expression)', fontsize=10)
for i in range(len(gene_names)):
    for j in range(10):
        ax1.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7)

# 2. Gene별 평균 상관관계 bar
ax2 = axes[0, 1]
mean_rs = corr_matrix.mean(axis=1)
std_rs  = corr_matrix.std(axis=1)
sorted_idx = np.argsort(mean_rs)[::-1]
colors_bar = ['#e74c3c' if r > 0.5 else '#3498db' if r > 0.3 else 'gray'
              for r in mean_rs[sorted_idx]]
ax2.barh([gene_names[i] for i in sorted_idx][::-1],
         mean_rs[sorted_idx][::-1],
         xerr=std_rs[sorted_idx][::-1],
         color=colors_bar[::-1], alpha=0.8)
ax2.axvline(0, color='black', linestyle='--', alpha=0.3)
ax2.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='r=0.5')
ax2.set_title('Mean Correlation across 10 folds\n(Top-3 score vs bulk)', fontsize=10)
ax2.set_xlabel('Pearson r (mean ± std)')
ax2.legend(fontsize=8)
ax2.grid(axis='x', alpha=0.3)

# 3. KRT6B scatter (all folds)
ax3 = axes[1, 0]
krt6b_idx = target_idx.get('KRT6B')
if krt6b_idx is not None:
    colors_fold = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, fold in enumerate(FOLDS):
        ax3.scatter(fold_top3_scores[fold], bulk_arr[:, krt6b_idx],
                   alpha=0.2, s=5, color=colors_fold[i], label=fold)
    # 전체 평균
    all_scores_concat = np.concatenate([fold_top3_scores[f] for f in FOLDS])
    all_krt6b = np.tile(bulk_arr[:, krt6b_idx], 10)
    r_all, _  = pearsonr(all_scores_concat, all_krt6b)
    ax3.set_title(f'Top-3 Score vs KRT6B\nAll folds combined r={r_all:.4f}', fontsize=10)
    ax3.set_xlabel('Top-3 PCC score')
    ax3.set_ylabel('KRT6B bulk expression')
    ax3.grid(alpha=0.3)

# 4. fold별 KRT6B correlation line plot
ax4 = axes[1, 1]
krt6b_rs = [results[f].get('KRT6B', (0,0))[0] for f in FOLDS]
sprr1b_rs = [results[f].get('SPRR1B', (0,0))[0] for f in FOLDS]
s100a9_rs = [results[f].get('S100A9', (0,0))[0] for f in FOLDS]
csta_rs   = [results[f].get('CSTA', (0,0))[0] for f in FOLDS]

ax4.plot(range(1,11), krt6b_rs,  '-o', label=f'KRT6B (mean={np.mean(krt6b_rs):.3f})',  linewidth=2)
ax4.plot(range(1,11), sprr1b_rs, '-s', label=f'SPRR1B (mean={np.mean(sprr1b_rs):.3f})', linewidth=2)
ax4.plot(range(1,11), s100a9_rs, '-^', label=f'S100A9 (mean={np.mean(s100a9_rs):.3f})', linewidth=2)
ax4.plot(range(1,11), csta_rs,   '-d', label=f'CSTA (mean={np.mean(csta_rs):.3f})',    linewidth=2)
ax4.axhline(0, color='black', linestyle='--', alpha=0.3)
ax4.axhline(0.5, color='red', linestyle=':', alpha=0.5)
ax4.set_xticks(range(1,11))
ax4.set_xticklabels([f'f{i:02d}' for i in range(1,11)], fontsize=8)
ax4.set_title('Top-3 Score vs Gene Correlation\nby Fold', fontsize=10)
ax4.set_ylabel('Pearson r')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/krt6b_correlation_all_folds.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: krt6b_correlation_all_folds.png')
print('Done!')
EOF