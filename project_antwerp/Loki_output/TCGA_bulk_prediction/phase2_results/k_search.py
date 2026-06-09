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
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/phase2_results'
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

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/fold_01/{row["slide_id"]}.npy')]
print(f'Slides: {len(matched)}')

FOLDS = [f'fold_{i:02d}' for i in range(1, 11)]
Ks_violin = [50, 100, 500, 'all']
Ks_fine   = list(range(10, 110, 10))  # K=10~100

fold_results = {}  # {fold: {K: gene_pccs array}}

for fold in FOLDS:
    print(f'\n[{fold}]')
    train_embs = F.normalize(torch.tensor(
        np.load(f'{FT_EMB}/{fold}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
    train_expr = torch.tensor(
        np.load(f'{FT_EMB}/{fold}/train_exprs.npy'), dtype=torch.float32, device=device)

    all_Ks = list(set(Ks_violin + Ks_fine) - {'all'}) + ['all']
    K_preds = {K: [] for K in all_Ks}
    K_bulks = {K: [] for K in all_Ks}

    for sid, bulk in matched:
        embs = F.normalize(torch.tensor(
            np.load(f'{TCGA_EMB}/{fold}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
        with torch.no_grad():
            sim        = torch.clamp(embs @ train_embs.T, min=0)
            weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
            tile_preds = (weights @ train_expr).cpu().numpy()
        tp = tile_preds[:, common_idx]
        T  = len(tp)

        # tile-wise PCC score
        bulk_c = bulk - bulk.mean()
        tile_c = tp - tp.mean(axis=1, keepdims=True)
        num    = (tile_c * bulk_c).sum(axis=1)
        denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
        scores = np.where(denom > 1e-8, num / denom, -999)
        sorted_idx = np.argsort(scores)[::-1]

        for K in all_Ks:
            if K == 'all':
                selected = tp
            else:
                k = min(K, T)
                valid   = scores > -999
                v_idx   = np.where(valid)[0]
                top_idx = v_idx[np.argsort(scores[valid])[::-1][:k]]
                selected = tp[top_idx]
            K_preds[K].append(selected.sum(axis=0))
            K_bulks[K].append(bulk)
        del embs

    del train_embs, train_expr
    torch.cuda.empty_cache()

    # PCC 계산
    fold_results[fold] = {}
    for K in all_Ks:
        p_arr = np.array(K_preds[K])
        b_arr = np.array(K_bulks[K])
        gene_pccs = [pearsonr(p_arr[:,j], b_arr[:,j])[0]
                     for j in range(p_arr.shape[1])
                     if p_arr[:,j].std()>1e-8 and b_arr[:,j].std()>1e-8]
        fold_results[fold][K] = np.array(gene_pccs)
    print(f'  K=50: {fold_results[fold][50].mean():.4f}  K=100: {fold_results[fold][100].mean():.4f}')

# ── Figure 1: Violin plot (K=50, 100, 500, all) ───────────
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Gene-wise PCC Distribution: All Folds\n(Tile PCC Sum Aggregation)', fontsize=14)
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for ax_i, K in enumerate(Ks_violin):
    ax   = axes[ax_i//2, ax_i%2]
    data = [fold_results[fold][K] for fold in FOLDS]
    parts = ax.violinplot(data, positions=range(1,11), showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    means = [d.mean() for d in data]
    ax.plot(range(1,11), means, 'k^', markersize=7, label='mean')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xticks(range(1,11))
    ax.set_xticklabels([f'f{i:02d}' for i in range(1,11)], fontsize=9)
    ax.set_title(f'K={K}: Gene-wise PCC (mean={np.mean(means):.4f})', fontsize=11)
    ax.set_ylabel('Gene-wise PCC')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/violin_all_folds.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: violin_all_folds.png')

# ── Figure 2: K 정밀 탐색 (K=10~100) ─────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
colors2 = plt.cm.tab10(np.linspace(0, 1, 10))

for i, fold in enumerate(FOLDS):
    means = [fold_results[fold][K].mean() for K in Ks_fine]
    ax.plot(Ks_fine, means, '-o', color=colors2[i], label=fold,
            linewidth=1.5, markersize=4, alpha=0.8)

# 전체 평균
overall_means = [np.mean([fold_results[f][K].mean() for f in FOLDS]) for K in Ks_fine]
ax.plot(Ks_fine, overall_means, 'k-o', linewidth=3, markersize=8, label='Mean (all folds)', zorder=5)

best_K = Ks_fine[np.argmax(overall_means)]
best_v = max(overall_means)
ax.axvline(best_K, color='red', linestyle='--', alpha=0.7)
ax.scatter([best_K], [best_v], color='red', s=150, zorder=6,
           label=f'Best K={best_K} (mean={best_v:.4f})')

ax.set_xlabel('K (top tile count)', fontsize=12)
ax.set_ylabel('Gene-wise PCC (mean)', fontsize=12)
ax.set_title('Optimization K (K=10~100): All Folds', fontsize=13)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.set_xticks(Ks_fine)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/K_fine_search.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: K_fine_search.png')

# ── 요약 표 ───────────────────────────────────────────────
print('\n=== K 정밀 탐색 요약 ===')
print(f'{"K":>5} | {"mean":>8} | {"std":>8} | {"best fold":>10}')
print('-'*38)
for K in Ks_fine:
    vals      = [fold_results[f][K].mean() for f in FOLDS]
    best_fold = FOLDS[np.argmax(vals)]
    print(f'{K:>5} | {np.mean(vals):>8.4f} | {np.std(vals):>8.4f} | {best_fold:>10}')

print(f'\n optimal K: {best_K}  mean={best_v:.4f}')
print('Done!')
EOF