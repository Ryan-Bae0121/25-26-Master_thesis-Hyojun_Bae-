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
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/optimal_K'
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
Ks    = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]

# {fold: {K: gene_wise_mean}}
fold_K_results = {fold: {} for fold in FOLDS}

for fold in FOLDS:
    print(f'\n[{fold}]')
    train_embs = F.normalize(torch.tensor(
        np.load(f'{FT_EMB}/{fold}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
    train_expr = torch.tensor(
        np.load(f'{FT_EMB}/{fold}/train_exprs.npy'), dtype=torch.float32, device=device)

    # 모든 tile pred 미리 계산
    all_tps, all_bulks, all_scores = [], [], []
    for sid, bulk in matched:
        embs = F.normalize(torch.tensor(
            np.load(f'{TCGA_EMB}/{fold}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
        with torch.no_grad():
            sim   = torch.clamp(embs @ train_embs.T, min=0)
            w     = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
            tp    = (w @ train_expr).cpu().numpy()[:, common_idx]
        # tile-wise PCC score
        bulk_c = bulk - bulk.mean()
        tile_c = tp - tp.mean(axis=1, keepdims=True)
        num    = (tile_c * bulk_c).sum(axis=1)
        denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
        scores = np.where(denom > 1e-8, num/denom, -999)
        all_tps.append(tp)
        all_bulks.append(bulk)
        all_scores.append(scores)
        del embs

    del train_embs, train_expr
    torch.cuda.empty_cache()

    # K별 PCC 계산
    for K in Ks:
        preds = []
        for tp, bulk, scores in zip(all_tps, all_bulks, all_scores):
            T      = len(tp)
            k      = min(K, T)
            valid  = scores > -999
            v_idx  = np.where(valid)[0]
            top_idx = v_idx[np.argsort(scores[valid])[::-1][:k]]
            preds.append(tp[top_idx].sum(axis=0))

        p_arr = np.array(preds)
        b_arr = np.array(all_bulks)
        g     = [pearsonr(p_arr[:,j], b_arr[:,j])[0]
                 for j in range(p_arr.shape[1])
                 if p_arr[:,j].std()>1e-8 and b_arr[:,j].std()>1e-8]
        fold_K_results[fold][K] = np.array(g).mean()

    # fold별 최적 K
    best_K = max(Ks, key=lambda k: fold_K_results[fold][k])
    best_v = fold_K_results[fold][best_K]
    print(f'  Best K={best_K}  PCC={best_v:.4f}')
    for K in Ks:
        print(f'  K={K:>3}: {fold_K_results[fold][K]:.4f}')

# ── 최종 비교 표 ──────────────────────────────────────────
print('\n' + '='*70)
print('전체 Fold 최적 K 비교')
print('='*70)
header = f'{"fold":>8} | ' + ' | '.join([f'K={k:>3}' for k in Ks]) + ' | best_K'
print(header)
print('-' * len(header))

best_Ks = []
for fold in FOLDS:
    row_vals = [fold_K_results[fold][K] for K in Ks]
    best_K   = Ks[np.argmax(row_vals)]
    best_Ks.append(best_K)
    row_str  = f'{fold:>8} | ' + ' | '.join([f'{v:>5.3f}' for v in row_vals]) + f' | K={best_K}'
    print(row_str)

# 전체 평균
mean_vals = [np.mean([fold_K_results[f][K] for f in FOLDS]) for K in Ks]
best_K_overall = Ks[np.argmax(mean_vals)]
print('-' * len(header))
print(f'{"mean":>8} | ' + ' | '.join([f'{v:>5.3f}' for v in mean_vals]) + f' | K={best_K_overall}')
print(f'\n전체 최적 K: {best_K_overall}  mean={max(mean_vals):.4f}')
print(f'fold별 최적 K: {dict(zip(FOLDS, best_Ks))}')

# ── 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Optimal K Search: All Folds (Tile PCC Sum Aggregation)', fontsize=13)

colors = plt.cm.tab10(np.linspace(0, 1, 10))

# 1. fold별 K curve
for i, fold in enumerate(FOLDS):
    vals = [fold_K_results[fold][K] for K in Ks]
    axes[0].plot(Ks, vals, '-o', color=colors[i], label=fold,
                 linewidth=1.5, markersize=4, alpha=0.8)

axes[0].plot(Ks, mean_vals, 'k-o', linewidth=3, markersize=8,
             label=f'Mean (best K={best_K_overall})', zorder=5)
axes[0].axvline(best_K_overall, color='red', linestyle='--', alpha=0.7)
axes[0].scatter([best_K_overall], [max(mean_vals)], color='red', s=150, zorder=6)
axes[0].set_xlabel('K', fontsize=11)
axes[0].set_ylabel('Gene-wise PCC (mean)', fontsize=11)
axes[0].set_title('Gene-wise PCC of each K (all folds)', fontsize=11)
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(alpha=0.3)
axes[0].set_xticks(Ks)

# 2. fold별 최적 K 분포
best_K_counts = {K: best_Ks.count(K) for K in Ks if best_Ks.count(K) > 0}
axes[1].bar([str(k) for k in best_K_counts.keys()],
            best_K_counts.values(), color='#3498db', alpha=0.8)
axes[1].set_xlabel('Best K', fontsize=11)
axes[1].set_ylabel('Fold count', fontsize=11)
axes[1].set_title('Optimization K of each fold', fontsize=11)
axes[1].grid(axis='y', alpha=0.3)
for k, v in best_K_counts.items():
    axes[1].text(str(k), v+0.05, str(v), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/optimal_K_all_folds.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: {OUT_DIR}/optimal_K_all_folds.png')
print('Done!')
EOF