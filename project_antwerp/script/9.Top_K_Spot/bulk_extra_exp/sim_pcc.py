
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

Ks = [100, 300, 500, 1000, 'all']
results_pcc = {K: {'preds': [], 'bulks': []} for K in Ks}
results_cos = {K: {'preds': [], 'bulks': []} for K in Ks}

for sid, bulk in matched:
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)

    with torch.no_grad():
        sim        = torch.clamp(embs @ train_embs.T, min=0)
        weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()  # (T, 2000)

    tile_preds_common = tile_preds[:, common_idx]  # (T, 1968)
    T = len(tile_preds_common)

    # 방법 1: tile-wise PCC로 ranking
    tile_pccs = []
    for i in range(T):
        p = tile_preds_common[i]
        if p.std() < 1e-8:
            tile_pccs.append(-999)
            continue
        r, _ = pearsonr(p, bulk)
        tile_pccs.append(r)
    tile_pccs = np.array(tile_pccs)
    sorted_idx_pcc = np.argsort(tile_pccs)[::-1]

    # 방법 2: cosine similarity로 ranking (기존 OLD)
    tile_norm = F.normalize(
        torch.tensor(tile_preds_common, dtype=torch.float32, device=device), dim=-1)
    bulk_norm = F.normalize(
        torch.tensor(bulk, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
    cos_sim = (tile_norm @ bulk_norm.T).squeeze().cpu().numpy()
    sorted_idx_cos = np.argsort(cos_sim)[::-1]

    for K in Ks:
        if K == 'all':
            sel_pcc = tile_preds_common
            sel_cos = tile_preds_common
        else:
            k = min(K, T)
            sel_pcc = tile_preds_common[sorted_idx_pcc[:k]]
            sel_cos = tile_preds_common[sorted_idx_cos[:k]]

        results_pcc[K]['preds'].append(sel_pcc.sum(axis=0))
        results_pcc[K]['bulks'].append(bulk)
        results_cos[K]['preds'].append(sel_cos.sum(axis=0))
        results_cos[K]['bulks'].append(bulk)

def summarize(results, label):
    print(f'\n=== {label} ===')
    print(f'{"K":>6} | {"Gene-wise mean":>14} | {"Gene-wise median":>16} | {"Slide-wise mean":>15}')
    print('-' * 60)
    for K in Ks:
        p = np.array(results[K]['preds'])
        b = np.array(results[K]['bulks'])
        gpccs = [pearsonr(p[:,i], b[:,i])[0]
                 for i in range(p.shape[1])
                 if p[:,i].std() > 1e-8 and b[:,i].std() > 1e-8]
        spccs = [pearsonr(p[i], b[i])[0]
                 for i in range(p.shape[0])
                 if p[i].std() > 1e-8 and b[i].std() > 1e-8]
        g, s = np.array(gpccs), np.array(spccs)
        print(f'{str(K):>6} | {g.mean():>14.4f} | {np.median(g):>16.4f} | {s.mean():>15.4f}')

summarize(results_pcc, "Tile-wise PCC 기준 Top-K sum")
summarize(results_cos, "Cosine Similarity 기준 Top-K sum (기존 OLD)")
