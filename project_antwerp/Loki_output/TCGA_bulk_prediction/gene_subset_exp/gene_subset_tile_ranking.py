python3 << 'EOF'
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
"""
Slides: 331, Genes: 1968
HNSCC markers found: 24 / 37
  Found: ['CD4', 'CD8A', 'CDH1', 'CXCL1', 'CXCL8', 'EGFR', 'FN1', 'FOXP3', 'HIF1A', 'IL6', 'KRT16', 'KRT6B', 'MMP1', 'MMP10', 'MMP3', 'MMP9', 'PDCD1', 'S100A8', 'S100A9', 'SNAI1', 'TP63', 'VEGFA', 'VIM', 'ZEB1']
Bulk HVG top-50/100/300 selected

Computing tile predictions...
Done.

======================================================================
Gene Subset 기반 Tile Ranking 비교
======================================================================

[All 1968 (baseline)] (genes: 1968)
     K | Gene-wise mean | Gene-wise median |  score_std
----------------------------------------------------
    50 |         0.3795 |           0.4025 |           
   100 |         0.3654 |           0.3867 |           
   300 |         0.1986 |           0.2013 |           
   all |        -0.0048 |          -0.0076 |           

[HNSCC markers] (genes: 24)
     K | Gene-wise mean | Gene-wise median |  score_std
----------------------------------------------------
    50 |         0.2984 |           0.2971 |           
   100 |         0.2867 |           0.2851 |           
   300 |         0.1476 |           0.1457 |           
   all |        -0.0048 |          -0.0076 |           

[Bulk HVG top-50] (genes: 50)
     K | Gene-wise mean | Gene-wise median |  score_std
----------------------------------------------------
    50 |         0.2634 |           0.2595 |           
   100 |         0.2532 |           0.2485 |           
   300 |         0.1336 |           0.1225 |           
   all |        -0.0048 |          -0.0076 |           

[Bulk HVG top-100] (genes: 100)
     K | Gene-wise mean | Gene-wise median |  score_std
----------------------------------------------------
    50 |         0.3045 |           0.3056 |           
   100 |         0.2911 |           0.2913 |           
   300 |         0.1557 |           0.1465 |           
   all |        -0.0048 |          -0.0076 |           

[Bulk HVG top-300] (genes: 300)
     K | Gene-wise mean | Gene-wise median |  score_std
----------------------------------------------------
    50 |         0.3524 |           0.3668 |           
   100 |         0.3402 |           0.3540 |           
   300 |         0.1865 |           0.1863 |           
   all |        -0.0048 |          -0.0076 |           

============================================================
최종 비교 (K=50 기준)
============================================================
Gene subset            |  genes |  K=50 gene |  K=100 gene
-------------------------------------------------------
All 1968 (baseline)    |   1968 |     0.3795 |      0.3654
HNSCC markers          |     24 |     0.2984 |      0.2867
Bulk HVG top-50        |     50 |     0.2634 |      0.2532
Bulk HVG top-100       |    100 |     0.3045 |      0.2911
Bulk HVG top-300       |    300 |     0.3524 |      0.3402

Saved: gene_subset_comparison.png
Done!
"""
device = 'cuda'
GENE_LIST =  '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/gene_subset_exp'
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

# ── Gene subset 정의 ──────────────────────────────────────
# 1. HNSCC marker genes
hnscc_markers = [
    # Squamous cell markers
    'TP63', 'KRT5', 'KRT14', 'KRT6A', 'KRT6B', 'KRT16', 'KRT17',
    # Tumor markers
    'EGFR', 'CDKN2A', 'CCND1', 'TP53', 'NOTCH1', 'FAT1', 'CASP8',
    # EMT markers
    'CDH1', 'VIM', 'FN1', 'SNAI1', 'TWIST1', 'ZEB1',
    # Immune markers
    'CD274', 'PDCD1', 'CD8A', 'CD4', 'FOXP3', 'CD68',
    # HNSCC specific
    'S100A8', 'S100A9', 'MMP1', 'MMP3', 'MMP9', 'MMP10',
    'CXCL1', 'CXCL8', 'IL6', 'VEGFA', 'HIF1A',
]
# common_genes에 있는 것만 필터링
hnscc_idx = [i for i, g in enumerate(common_genes) if g in hnscc_markers]
hnscc_genes_found = [common_genes[i] for i in hnscc_idx]
print(f'HNSCC markers found: {len(hnscc_idx)} / {len(hnscc_markers)}')
print(f'  Found: {hnscc_genes_found}')

# 2. Bulk HVG top-50
bulk_arr_all = np.array([row[bulk_cols].values.astype(float)
                         for _, row in ref_df.iterrows()
                         if any(row['wsi_file_name'].split('.')[0] == s for s, _ in matched)])
bulk_var = np.array([ref_df[ref_df['slide_id']==sid].iloc[0][bulk_cols].values.astype(float)
                     for sid, _ in matched]).var(axis=0)
hvg50_idx  = np.argsort(bulk_var)[::-1][:50].tolist()
hvg100_idx = np.argsort(bulk_var)[::-1][:100].tolist()
hvg300_idx = np.argsort(bulk_var)[::-1][:300].tolist()
print(f'Bulk HVG top-50/100/300 selected')

# 3. Gene subsets 정의
gene_subsets = {
    'All 1968 (baseline)': list(range(G)),
    'HNSCC markers':        hnscc_idx,
    'Bulk HVG top-50':      hvg50_idx,
    'Bulk HVG top-100':     hvg100_idx,
    'Bulk HVG top-300':     hvg300_idx,
}

# ── tile predictions 미리 계산 ────────────────────────────
print('\nComputing tile predictions...')
slide_tile_preds, slide_bulks = [], []
for sid, bulk in matched:
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim        = torch.clamp(embs @ train_embs.T, min=0)
        weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()
    slide_tile_preds.append(tile_preds[:, common_idx])
    slide_bulks.append(bulk)
    del embs
torch.cuda.empty_cache()
bulk_mat = np.array(slide_bulks)
print(f'Done.')

def calc_pcc_arrays(pred_arr, bulk_arr):
    g = [pearsonr(pred_arr[:,j], bulk_arr[:,j])[0]
         for j in range(pred_arr.shape[1])
         if pred_arr[:,j].std()>1e-8 and bulk_arr[:,j].std()>1e-8]
    s = [pearsonr(pred_arr[i], bulk_arr[i])[0]
         for i in range(pred_arr.shape[0])
         if pred_arr[i].std()>1e-8 and bulk_arr[i].std()>1e-8]
    return np.array(g), np.array(s)

Ks = [50, 100, 300, 'all']

print('\n' + '='*70)
print('Gene Subset 기반 Tile Ranking 비교')
print('='*70)

all_results = {}

for subset_name, subset_idx in gene_subsets.items():
    print(f'\n[{subset_name}] (genes: {len(subset_idx)})')
    print(f'{"K":>6} | {"Gene-wise mean":>14} | {"Gene-wise median":>16} | {"score_std":>10}')
    print('-'*52)

    subset_results = []
    score_stds     = []

    for K in Ks:
        preds = []
        for tp, bulk in zip(slide_tile_preds, slide_bulks):
            T = len(tp)

            # subset gene으로만 PCC score 계산
            tp_sub   = tp[:, subset_idx]      # (T, subset_size)
            bulk_sub = bulk[subset_idx]        # (subset_size,)

            bulk_c = bulk_sub - bulk_sub.mean()
            tile_c = tp_sub - tp_sub.mean(axis=1, keepdims=True)
            num    = (tile_c * bulk_c).sum(axis=1)
            denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
            scores = np.where(denom > 1e-8, num / denom, -999)

            if K == 'all':
                selected = tp  # aggregation은 전체 1968 gene
            else:
                k       = min(K, T)
                valid   = scores > -999
                v_idx   = np.where(valid)[0]
                top_idx = v_idx[np.argsort(scores[valid])[::-1][:k]]
                selected = tp[top_idx]

            # pseudo-bulk: 전체 1968 gene으로 합산
            preds.append(selected.sum(axis=0))

        if K == Ks[0]:
            score_stds.append(np.mean([
                (np.where(
                    (tp[:, subset_idx] - tp[:, subset_idx].mean(axis=1, keepdims=True)) is not None,
                    ((tp[:, subset_idx]-tp[:, subset_idx].mean(axis=1, keepdims=True)) *
                     (bulk[subset_idx]-bulk[subset_idx].mean())).sum(axis=1) /
                    (np.sqrt(((tp[:, subset_idx]-tp[:, subset_idx].mean(axis=1, keepdims=True))**2).sum(axis=1)) *
                     np.sqrt(((bulk[subset_idx]-bulk[subset_idx].mean())**2).sum()) + 1e-8),
                    -999
                )).std()
                for tp, bulk in zip(slide_tile_preds[:5], slide_bulks[:5])
            ]))

        pred_arr = np.array(preds)
        g, s = calc_pcc_arrays(pred_arr, bulk_mat)
        subset_results.append((K, g.mean(), np.median(g), s.mean()))
        print(f'{str(K):>6} | {g.mean():>14.4f} | {np.median(g):>16.4f} | {"":>10}')

    all_results[subset_name] = subset_results

# ── 최종 비교 표 ──────────────────────────────────────────
print('\n' + '='*60)
print('최종 비교 (K=50 기준)')
print('='*60)
print(f'{"Gene subset":<22} | {"genes":>6} | {"K=50 gene":>10} | {"K=100 gene":>11}')
print('-'*55)
for subset_name, subset_idx in gene_subsets.items():
    r  = {k: (g, gm, s) for k, g, gm, s in all_results[subset_name]}
    print(f'{subset_name:<22} | {len(subset_idx):>6} | {r[50][0]:>10.4f} | {r[100][0]:>11.4f}')

# ── 시각화 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
colors_sub = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
K_nums = [50, 100, 300]

for i, (subset_name, _) in enumerate(gene_subsets.items()):
    r      = {k: (g, gm, s) for k, g, gm, s in all_results[subset_name]}
    g_vals = [r[k][0] for k in K_nums]
    ax.plot(K_nums, g_vals, '-o', color=colors_sub[i],
            label=subset_name, linewidth=2, markersize=7)

ax.set_xlabel('K (top tile count)', fontsize=11)
ax.set_ylabel('Gene-wise PCC (mean)', fontsize=11)
ax.set_title('Based on Gene Subset Tile Ranking: Gene-wise PCC of each K (fold_03)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/gene_subset_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: gene_subset_comparison.png')
print('Done!')
EOF