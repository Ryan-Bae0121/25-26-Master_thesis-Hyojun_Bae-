import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px/fold_03'

with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]
ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df['slide_id'] = ref_df['wsi_file_name'] + '.svs'
rna_cols = [c for c in ref_df.columns if c.startswith('rna_')]
ref_genes = [c.replace('rna_', '') for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
common_idx = [gene_list.index(g) for g in common_genes]
bulk_cols = ['rna_' + g for g in common_genes]

train_embs = F.normalize(torch.tensor(np.load(f'{FT_EMB}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
train_expr = torch.tensor(np.load(f'{FT_EMB}/train_exprs.npy'), dtype=torch.float32, device=device)

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/{row["slide_id"]}.npy')]

Ks = [5, 10, 20, 30, 50]
results = {K: {'preds': [], 'bulks': []} for K in Ks}

for sid, bulk in matched:
    embs = F.normalize(torch.tensor(np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim = torch.clamp(embs @ train_embs.T, min=0)
        weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()
    tile_preds_common = tile_preds[:, common_idx]
    P = torch.tensor(tile_preds_common, dtype=torch.float32, device=device)
    b = torch.tensor(bulk, dtype=torch.float32, device=device)
    P_c = P - P.mean(dim=1, keepdim=True)
    b_c = b - b.mean()
    tile_pccs = ((P_c * b_c).sum(dim=1) / (P_c.norm(dim=1) * b_c.norm() + 1e-8)).cpu().numpy()
    sorted_idx = np.argsort(tile_pccs)[::-1]
    for K in Ks:
        k = min(K, len(tile_preds_common))
        results[K]['preds'].append(tile_preds_common[sorted_idx[:k]].mean(axis=0))
        results[K]['bulks'].append(bulk)
    del embs, sim, weights

print(f'{"K":>5} | {"Gene mean":>10} | {"Gene median":>12}')
print('-'*35)
for K in Ks:
    p = np.array(results[K]['preds'])
    b = np.array(results[K]['bulks'])
    gpccs = [pearsonr(p[:,i],b[:,i])[0] for i in range(p.shape[1]) if p[:,i].std()>1e-8 and b[:,i].std()>1e-8]
    g = np.array(gpccs)
    print(f'{K:>5} | {g.mean():>10.4f} | {np.median(g):>12.4f}')