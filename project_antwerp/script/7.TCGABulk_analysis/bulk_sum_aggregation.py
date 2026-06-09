
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'

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

# bulk를 L2 정규화 → cosine similarity 계산용
bulk_tensor = torch.tensor(
    np.array([b for _, b in matched]), dtype=torch.float32, device=device)
bulk_norm = F.normalize(bulk_tensor, dim=-1)  # (S, 1968)

Ks = [100, 300, 500, 1000, 3000, 'all']

# K별 결과 저장
results = {K: {'preds': [], 'bulks': []} for K in Ks}

for i, (sid, bulk) in enumerate(matched):
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)

    with torch.no_grad():
        sim        = torch.clamp(embs @ train_embs.T, min=0)
        weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()  # (T, 2000)

    tile_preds_common = tile_preds[:, common_idx]  # (T, 1968)
    T = len(tile_preds_common)

    # 방법 1: 각 tile 예측값과 bulk의 cosine similarity로 ranking
    tile_pred_norm = F.normalize(
        torch.tensor(tile_preds_common, dtype=torch.float32, device=device), dim=-1)
    bulk_norm_i = F.normalize(
        torch.tensor(bulk, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
    cos_sim = (tile_pred_norm @ bulk_norm_i.T).squeeze().cpu().numpy()  # (T,)

    # K별 sum aggregation
    sorted_idx = np.argsort(cos_sim)[::-1]  # 높은 순 정렬

    for K in Ks:
        if K == 'all':
            selected = tile_preds_common
        else:
            k = min(K, T)
            selected = tile_preds_common[sorted_idx[:k]]

        # Sum → pseudo-bulk
        pseudo_bulk = selected.sum(axis=0)  # (1968,)
        results[K]['preds'].append(pseudo_bulk)
        results[K]['bulks'].append(bulk)

# PCC 계산 (gene-wise + slide-wise)
print('\n=== Sum Aggregation 결과 (fold_03) ===')
print(f'{"K":>8} | {"Gene-wise mean":>14} | {"Gene-wise median":>16} | {"Slide-wise mean":>15}')
print('-' * 65)

for K in Ks:
    pred_arr = np.array(results[K]['preds'])  # (S, 1968)
    bulk_arr = np.array(results[K]['bulks'])  # (S, 1968)

    # Gene-wise PCC
    gene_pccs = []
    for j in range(pred_arr.shape[1]):
        p, b = pred_arr[:, j], bulk_arr[:, j]
        if p.std() < 1e-8 or b.std() < 1e-8:
            continue
        r, _ = pearsonr(p, b)
        gene_pccs.append(r)
    gene_pccs = np.array(gene_pccs)

    # Slide-wise PCC
    slide_pccs = []
    for j in range(pred_arr.shape[0]):
        p, b = pred_arr[j], bulk_arr[j]
        if p.std() < 1e-8 or b.std() < 1e-8:
            slide_pccs.append(np.nan)
            continue
        r, _ = pearsonr(p, b)
        slide_pccs.append(r)
    slide_pccs = np.array(slide_pccs)
    valid = ~np.isnan(slide_pccs)

    print(f'{str(K):>8} | {gene_pccs.mean():>14.4f} | {np.median(gene_pccs):>16.4f} | {slide_pccs[valid].mean():>15.4f}')

