
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

# 슬라이드 3개만 샘플링해서 분포 확인
all_sims = []
for sid, bulk in matched[:5]:
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)

    with torch.no_grad():
        sim        = torch.clamp(embs @ train_embs.T, min=0)
        weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()

    tile_preds_common = tile_preds[:, common_idx]

    # tile vs bulk cosine similarity
    tile_norm = F.normalize(
        torch.tensor(tile_preds_common, dtype=torch.float32, device=device), dim=-1)
    bulk_norm = F.normalize(
        torch.tensor(bulk, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
    cos_sim = (tile_norm @ bulk_norm.T).squeeze().cpu().numpy()

    all_sims.append(cos_sim)
    print(f'{sid}:')
    print(f'  tiles={len(cos_sim)}')
    print(f'  sim: min={cos_sim.min():.4f} max={cos_sim.max():.4f} mean={cos_sim.mean():.4f} std={cos_sim.std():.4f}')
    print(f'  percentile 10%={np.percentile(cos_sim,10):.4f} 25%={np.percentile(cos_sim,25):.4f} 50%={np.percentile(cos_sim,50):.4f} 75%={np.percentile(cos_sim,75):.4f} 90%={np.percentile(cos_sim,90):.4f}')
    print()
