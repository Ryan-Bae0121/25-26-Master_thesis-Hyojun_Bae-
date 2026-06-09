
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
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

sids = ['TCGA-CV-6950-01Z-00-DX1', 'TCGA-D6-6515-01Z-00-DX1', 'TCGA-CQ-7068-01Z-00-DX1']

print(f'{"슬라이드":<30} | {"방법":<20} | {"min":>6} | {"max":>6} | {"mean":>6} | {"std":>6} | {"range":>6}')
print('-' * 95)

for sid in sids:
    row = ref_df[ref_df['slide_id'] == sid]
    if len(row) == 0: continue
    bulk = row.iloc[0][bulk_cols].values.astype(float)

    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)

    with torch.no_grad():
        sim        = torch.clamp(embs @ train_embs.T, min=0)
        weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()

    tile_preds_common = tile_preds[:, common_idx]
    T = len(tile_preds_common)

    # 방법 1: Cosine similarity (OLD)
    tile_norm = F.normalize(
        torch.tensor(tile_preds_common, dtype=torch.float32, device=device), dim=-1)
    bulk_norm = F.normalize(
        torch.tensor(bulk, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
    cos_sim = (tile_norm @ bulk_norm.T).squeeze().cpu().numpy()

    # 방법 2: Tile-wise PCC (NEW)
    tile_pccs = []
    for i in range(T):
        p = tile_preds_common[i]
        if p.std() < 1e-8:
            tile_pccs.append(np.nan)
            continue
        r, _ = pearsonr(p, bulk)
        tile_pccs.append(r)
    tile_pccs = np.array(tile_pccs)
    valid = ~np.isnan(tile_pccs)

    # 방법 3: ST max similarity (NEW2)
    st_sim = (embs @ train_embs.T).max(dim=1).values.cpu().numpy()

    short_sid = sid.split('-')[1] + '-' + sid.split('-')[2]

    for method, vals in [
        ('Cosine sim (OLD)', cos_sim),
        ('Tile PCC (NEW)',   tile_pccs[valid]),
        ('ST max sim',       st_sim)
    ]:
        print(f'{short_sid:<30} | {method:<20} | {vals.min():>6.4f} | {vals.max():>6.4f} | {vals.mean():>6.4f} | {vals.std():>6.4f} | {vals.max()-vals.min():>6.4f}')
    print()
