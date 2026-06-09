python3 << 'EOF'
import numpy as np, torch, torch.nn.functional as F
import pandas as pd, os
from sklearn.decomposition import PCA
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/leiden_analysis'
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

train_embs = F.normalize(torch.tensor(
    np.load(f'{FT_EMB}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
train_expr = torch.tensor(
    np.load(f'{FT_EMB}/train_exprs.npy'), dtype=torch.float32, device=device)

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/{row["slide_id"]}.npy')]
print(f'Slides: {len(matched)}')

K = 3
all_tile_preds = []
all_labels     = []
all_scores     = []
all_slide_ids  = []

print('Collecting top-3 / bot-3 tiles only...')
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
    sorted_ = v_idx[np.argsort(scores[valid])[::-1]]

    top3_idx = sorted_[:K]
    bot3_idx = sorted_[-K:]

    for t in top3_idx:
        all_tile_preds.append(tp[t])
        all_labels.append('top3')
        all_scores.append(scores[t])
        all_slide_ids.append(sid)

    for t in bot3_idx:
        all_tile_preds.append(tp[t])
        all_labels.append('bot3')
        all_scores.append(scores[t])
        all_slide_ids.append(sid)

    del embs

torch.cuda.empty_cache()

tile_pred_matrix = np.array(all_tile_preds)  # (331×6, 1968)
labels           = np.array(all_labels)
scores_arr       = np.array(all_scores)
print(f'Total tiles: {len(tile_pred_matrix)} (top3: {(labels=="top3").sum()}, bot3: {(labels=="bot3").sum()})')

# ── PCA ──────────────────────────────────────────────────
print('Running PCA...')
pca = PCA(n_components=50, random_state=42)
tile_pca = pca.fit_transform(tile_pred_matrix)
print(f'Explained variance (top-10 PC): {pca.explained_variance_ratio_[:10].sum():.3f}')
print(f'Per PC: {pca.explained_variance_ratio_[:5].round(3)}')

# ── Leiden ───────────────────────────────────────────────
print('Running Leiden clustering...')
adata = sc.AnnData(tile_pca)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
sc.tl.leiden(adata, resolution=0.5, random_state=42)
sc.tl.umap(adata, random_state=42)

clusters   = adata.obs['leiden'].values.astype(str)
n_clusters = len(np.unique(clusters))
umap_coords = adata.obsm['X_umap']
print(f'Clusters: {n_clusters}')

# ── 통계 출력 ─────────────────────────────────────────────
print('\n=== Cluster별 top3/bot3 비율 ===')
print(f'{"Cluster":>8} | {"n":>6} | {"top3":>6} | {"bot3":>6} | {"top3%":>7} | {"mean_score":>11}')
print('-'*55)
for c in sorted(np.unique(clusters)):
    mask      = clusters == c
    n         = mask.sum()
    n_top     = ((labels[mask]) == 'top3').sum()
    n_bot     = ((labels[mask]) == 'bot3').sum()
    top_pct   = n_top / n * 100
    mean_sc   = scores_arr[mask].mean()
    print(f'{c:>8} | {n:>6} | {n_top:>6} | {n_bot:>6} | {top_pct:>6.1f}% | {mean_sc:>11.4f}')

print('\n=== Top-3 tile cluster 분포 ===')
print(pd.Series(clusters[labels=='top3']).value_counts().to_dict())
print('\n=== Bot-3 tile cluster 분포 ===')
print(pd.Series(clusters[labels=='bot3']).value_counts().to_dict())

print('\n=== 각 cluster 대표 gene (top-5) ===')
for c in sorted(np.unique(clusters)):
    mask   = clusters == c
    c_mean = tile_pred_matrix[mask].mean(axis=0)
    top5   = np.argsort(c_mean)[::-1][:5]
    print(f'  C{c}: {[common_genes[i] for i in top5]}')

# ── 시각화 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Tile pred PCA + Leiden: Top-3 vs Bot-3 Only (fold_03)\n'
             f'Total tiles: {len(tile_pred_matrix)} (top3: {(labels=="top3").sum()}, bot3: {(labels=="bot3").sum()})',
             fontsize=12)

# 1. Leiden clusters
colors_c = plt.cm.tab10(np.linspace(0, 1, n_clusters))
for i, c in enumerate(sorted(np.unique(clusters))):
    mask = clusters == c
    axes[0].scatter(umap_coords[mask,0], umap_coords[mask,1],
                   c=[colors_c[i]], s=15, alpha=0.6, label=f'C{c} (n={mask.sum()})')
axes[0].set_title('Leiden Clusters', fontsize=11)
axes[0].legend(fontsize=7, markerscale=2)
axes[0].set_xlabel('UMAP1'); axes[0].set_ylabel('UMAP2')

# 2. Top3 vs Bot3
for group, color, size in [('bot3','#3498db', 30), ('top3','#e74c3c', 30)]:
    mask = labels == group
    axes[1].scatter(umap_coords[mask,0], umap_coords[mask,1],
                   c=color, s=size, alpha=0.7,
                   label=f'{group} (n={mask.sum()})')
axes[1].set_title('Top-3 vs Bot-3 Tiles', fontsize=11)
axes[1].legend(fontsize=9, markerscale=2)
axes[1].set_xlabel('UMAP1'); axes[1].set_ylabel('UMAP2')

# 3. PCC score heatmap
sc_plot = axes[2].scatter(umap_coords[:,0], umap_coords[:,1],
                          c=scores_arr, cmap='RdYlGn', s=15, alpha=0.7,
                          vmin=scores_arr.min(), vmax=scores_arr.max())
plt.colorbar(sc_plot, ax=axes[2])
axes[2].set_title('Tile-wise PCC Score', fontsize=11)
axes[2].set_xlabel('UMAP1'); axes[2].set_ylabel('UMAP2')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/leiden_top3_bot3_only.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: leiden_top3_bot3_only.png')
print('Done!')
EOF