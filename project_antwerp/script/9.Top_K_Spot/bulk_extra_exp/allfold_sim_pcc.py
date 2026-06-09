
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'#'/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/'
# /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tile_pcc_results'
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

Ks = [100, 300, 500, 'all']
FOLDS = [f'fold_{i:02d}' for i in range(1, 11)]

# fold별 결과 저장
# {fold: {K: {'gene_pccs': [...], 'slide_pccs': [...]}}}
fold_results = {}

for fold in FOLDS:
    print(f'\n[{fold}]')
    ft_dir   = f'{FT_EMB}/{fold}'
    tcga_dir = f'{TCGA_EMB}/{fold}'

    train_embs = F.normalize(torch.tensor(
        np.load(f'{ft_dir}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
    train_expr = torch.tensor(
        np.load(f'{ft_dir}/train_exprs.npy'), dtype=torch.float32, device=device)

    results = {K: {'preds': [], 'bulks': []} for K in Ks}

    for sid, bulk in matched:
        emb_path = f'{tcga_dir}/{sid}.npy'
        if not os.path.exists(emb_path):
            continue

        embs = F.normalize(torch.tensor(
            np.load(emb_path), dtype=torch.float32, device=device), dim=-1)

        with torch.no_grad():
            sim        = torch.clamp(embs @ train_embs.T, min=0)
            weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
            tile_preds = (weights @ train_expr).cpu().numpy()

        tile_preds_common = tile_preds[:, common_idx]
        T = len(tile_preds_common)

        # tile-wise PCC로 ranking
        tile_pccs = []
        for i in range(T):
            p = tile_preds_common[i]
            if p.std() < 1e-8:
                tile_pccs.append(-999)
                continue
            r, _ = pearsonr(p, bulk)
            tile_pccs.append(r)
        tile_pccs   = np.array(tile_pccs)
        sorted_idx  = np.argsort(tile_pccs)[::-1]

        for K in Ks:
            if K == 'all':
                selected = tile_preds_common
            else:
                k = min(K, T)
                selected = tile_preds_common[sorted_idx[:k]]
            results[K]['preds'].append(selected.sum(axis=0))
            results[K]['bulks'].append(bulk)

        del embs
    del train_embs, train_expr
    torch.cuda.empty_cache()

    # PCC 계산
    fold_results[fold] = {}
    for K in Ks:
        p_arr = np.array(results[K]['preds'])
        b_arr = np.array(results[K]['bulks'])

        gene_pccs = [pearsonr(p_arr[:,i], b_arr[:,i])[0]
                     for i in range(p_arr.shape[1])
                     if p_arr[:,i].std() > 1e-8 and b_arr[:,i].std() > 1e-8]
        slide_pccs = [pearsonr(p_arr[i], b_arr[i])[0]
                      for i in range(p_arr.shape[0])
                      if p_arr[i].std() > 1e-8 and b_arr[i].std() > 1e-8]

        fold_results[fold][K] = {
            'gene_pccs':  np.array(gene_pccs),
            'slide_pccs': np.array(slide_pccs)
        }
        print(f'  K={K}: gene_mean={np.array(gene_pccs).mean():.4f} slide_mean={np.array(slide_pccs).mean():.4f}')

# ── 요약 표 출력 ──────────────────────────────────────────
print('\n=== 전체 요약 ===')
print(f'{"fold":>8} | {"K=100 gene":>10} | {"K=300 gene":>10} | {"K=500 gene":>10} | {"K=all gene":>10}')
print('-' * 55)
for fold in FOLDS:
    r = fold_results[fold]
    print(f'{fold:>8} | {r[100]["gene_pccs"].mean():>10.4f} | {r[300]["gene_pccs"].mean():>10.4f} | {r[500]["gene_pccs"].mean():>10.4f} | {r["all"]["gene_pccs"].mean():>10.4f}')

means = {K: np.mean([fold_results[f][K]['gene_pccs'].mean() for f in FOLDS]) for K in Ks}
print(f'{"mean":>8} | {means[100]:>10.4f} | {means[300]:>10.4f} | {means[500]:>10.4f} | {means["all"]:>10.4f}')

# ── Violin plot ───────────────────────────────────────────
# 1. fold별 K=100 gene-wise PCC violin
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Tile-wise PCC Top-K Sum: Gene-wise PCC Distribution\n(fold_01 ~ fold_10)', fontsize=14)

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for ax_idx, K in enumerate(Ks):
    ax = axes[ax_idx // 2, ax_idx % 2]
    data = [fold_results[fold][K]['gene_pccs'] for fold in FOLDS]
    parts = ax.violinplot(data, positions=range(1, 11), showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f'f{i:02d}' for i in range(1, 11)], fontsize=9)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4)
    ax.set_title(f'K={K}: Gene-wise PCC per fold')
    ax.set_ylabel('Gene-wise PCC')
    ax.set_xlabel('Fold')
    means_k = [d.mean() for d in data]
    ax.plot(range(1, 11), means_k, 'k^', markersize=6, label='mean')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/violin_gene_pcc_by_fold.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: violin_gene_pcc_by_fold.png')

# 2. K별 비교 violin (fold 평균)
fig, ax = plt.subplots(figsize=(12, 6))
K_labels = ['K=100', 'K=300', 'K=500', 'K=all']
K_colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

all_data = []
positions = []
pos = 1
for i, K in enumerate(Ks):
    combined = np.concatenate([fold_results[fold][K]['gene_pccs'] for fold in FOLDS])
    all_data.append(combined)
    positions.append(pos)
    pos += 2

parts = ax.violinplot(all_data, positions=positions, showmedians=True, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(K_colors[i])
    pc.set_alpha(0.8)

ax.set_xticks(positions)
ax.set_xticklabels(K_labels, fontsize=12)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.4)
ax.set_title('Gene-wise PCC Distribution by K (all folds combined)', fontsize=13)
ax.set_ylabel('Gene-wise PCC')

for i, (pos, data) in enumerate(zip(positions, all_data)):
    ax.text(pos, data.mean() + 0.01, f'μ={data.mean():.3f}', ha='center', fontsize=9, color=K_colors[i])

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/violin_gene_pcc_by_K.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: violin_gene_pcc_by_K.png')

# 3. Slide-wise PCC violin
fig, ax = plt.subplots(figsize=(12, 6))
slide_data = []
for K in Ks:
    combined = np.concatenate([fold_results[fold][K]['slide_pccs'] for fold in FOLDS])
    slide_data.append(combined)

parts = ax.violinplot(slide_data, positions=positions, showmedians=True, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(K_colors[i])
    pc.set_alpha(0.8)

ax.set_xticks(positions)
ax.set_xticklabels(K_labels, fontsize=12)
ax.set_title('Slide-wise PCC Distribution by K (all folds combined)', fontsize=13)
ax.set_ylabel('Slide-wise PCC')

for i, (pos, data) in enumerate(zip(positions, slide_data)):
    ax.text(pos, data.mean() + 0.005, f'μ={data.mean():.3f}', ha='center', fontsize=9, color=K_colors[i])

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/violin_slide_pcc_by_K.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: violin_slide_pcc_by_K.png')

print('\nAll done!')
