"""
=== Fold-wise Sum Aggregation (K=100, 500, all) ===
    fold | K=100 gene | K=500 gene | K=all gene | K=100 slide | K=all slide
---------------------------------------------------------------------------
 fold_01 |     0.2886 |     0.0763 |    -0.0068 |      0.7223 |      0.7171
 fold_02 |     0.2548 |     0.0660 |    -0.0058 |      0.7264 |      0.7214
 fold_03 |     0.2897 |     0.0800 |    -0.0058 |      0.7242 |      0.7193
 fold_04 |     0.2667 |     0.0695 |    -0.0064 |      0.7220 |      0.7166
 fold_05 |     0.2390 |     0.0609 |    -0.0058 |      0.7229 |      0.7170
 fold_06 |     0.2500 |     0.0602 |    -0.0074 |      0.7221 |      0.7174
 fold_07 |     0.2486 |     0.0688 |    -0.0075 |      0.7234 |      0.7180
 fold_08 |     0.2907 |     0.0876 |    -0.0073 |      0.7227 |      0.7161
 fold_09 |     0.2880 |     0.0810 |    -0.0065 |      0.7192 |      0.7139
 fold_10 |     0.2183 |     0.0420 |    -0.0093 |      0.7189 |      0.7149
---------------------------------------------------------------------------
    mean |     0.2635 |     0.0692 |    -0.0068 |      0.7224 |      0.7172
"""
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import os
from tqdm import tqdm

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings'

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

Ks = [100, 500, 'all']

def compute_pcc_sum_aggregation(fold, Ks, matched):
    ft_emb_dir  = f'{FT_EMB}/{fold}'
    tcga_emb_dir = f'{TCGA_EMB}/{fold}'

    train_embs = F.normalize(torch.tensor(
        np.load(f'{ft_emb_dir}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
    train_expr = torch.tensor(
        np.load(f'{ft_emb_dir}/train_exprs.npy'), dtype=torch.float32, device=device)

    results = {K: {'preds': [], 'bulks': []} for K in Ks}

    for sid, bulk in matched:
        emb_path = f'{tcga_emb_dir}/{sid}.npy'
        if not os.path.exists(emb_path):
            continue

        embs = F.normalize(torch.tensor(
            np.load(emb_path), dtype=torch.float32, device=device), dim=-1)

        with torch.no_grad():
            sim        = torch.clamp(embs @ train_embs.T, min=0)
            weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
            tile_preds = (weights @ train_expr).cpu().numpy()  # (T, 2000)

        tile_preds_common = tile_preds[:, common_idx]  # (T, 1968)
        T = len(tile_preds_common)

        # tile ranking: 각 tile 예측값과 bulk의 cosine similarity
        tile_pred_norm = F.normalize(
            torch.tensor(tile_preds_common, dtype=torch.float32, device=device), dim=-1)
        bulk_norm = F.normalize(
            torch.tensor(bulk, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
        cos_sim    = (tile_pred_norm @ bulk_norm.T).squeeze().cpu().numpy()
        sorted_idx = np.argsort(cos_sim)[::-1]

        for K in Ks:
            if K == 'all':
                selected = tile_preds_common
            else:
                k = min(K, T)
                selected = tile_preds_common[sorted_idx[:k]]
            pseudo_bulk = selected.sum(axis=0)
            results[K]['preds'].append(pseudo_bulk)
            results[K]['bulks'].append(bulk)

    del train_embs, train_expr
    torch.cuda.empty_cache()

    # PCC 계산
    fold_result = {}
    for K in Ks:
        pred_arr = np.array(results[K]['preds'])
        bulk_arr = np.array(results[K]['bulks'])

        gene_pccs = [pearsonr(pred_arr[:,j], bulk_arr[:,j])[0]
                     for j in range(pred_arr.shape[1])
                     if pred_arr[:,j].std() > 1e-8 and bulk_arr[:,j].std() > 1e-8]
        slide_pccs = [pearsonr(pred_arr[j], bulk_arr[j])[0]
                      for j in range(pred_arr.shape[0])
                      if pred_arr[j].std() > 1e-8 and bulk_arr[j].std() > 1e-8]

        gene_pccs  = np.array(gene_pccs)
        slide_pccs = np.array(slide_pccs)
        fold_result[K] = {
            'gene_mean':  gene_pccs.mean(),
            'gene_med':   np.median(gene_pccs),
            'slide_mean': slide_pccs.mean()
        }
    return fold_result

# 전체 fold 실행
print('\n=== Fold-wise Sum Aggregation (K=100, 500, all) ===')
print(f'{"fold":>8} | {"K=100 gene":>10} | {"K=500 gene":>10} | {"K=all gene":>10} | {"K=100 slide":>11} | {"K=all slide":>11}')
print('-' * 75)

all_results = {}
for fold_num in range(1, 11):
    fold = f'fold_{fold_num:02d}'
    res  = compute_pcc_sum_aggregation(fold, Ks, matched)
    all_results[fold] = res
    print(f'{fold:>8} | {res[100]["gene_mean"]:>10.4f} | {res[500]["gene_mean"]:>10.4f} | {res["all"]["gene_mean"]:>10.4f} | {res[100]["slide_mean"]:>11.4f} | {res["all"]["slide_mean"]:>11.4f}')

# 평균
gene_100  = np.mean([all_results[f'fold_{i:02d}'][100]['gene_mean']  for i in range(1,11)])
gene_500  = np.mean([all_results[f'fold_{i:02d}'][500]['gene_mean']  for i in range(1,11)])
gene_all  = np.mean([all_results[f'fold_{i:02d}']['all']['gene_mean'] for i in range(1,11)])
slide_100 = np.mean([all_results[f'fold_{i:02d}'][100]['slide_mean'] for i in range(1,11)])
slide_all = np.mean([all_results[f'fold_{i:02d}']['all']['slide_mean'] for i in range(1,11)])

print('-' * 75)
print(f'{"mean":>8} | {gene_100:>10.4f} | {gene_500:>10.4f} | {gene_all:>10.4f} | {slide_100:>11.4f} | {slide_all:>11.4f}')
