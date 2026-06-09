
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/ensemble_ranking'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/tile_lists', exist_ok=True)

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

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/fold_01/{row["slide_id"]}.npy')]
print(f'Slides: {len(matched)}, Genes: {G}')

FOLDS = [f'fold_{i:02d}' for i in range(1, 11)]
Ks    = [50, 100, 300, 500]

# ── Step 1: 슬라이드별 10 fold 앙상블 tile_pred 계산 ──────
print('\nComputing ensemble tile predictions...')

# 결과 저장용
ensemble_results = {K: {'preds': [], 'bulks': []} for K in Ks}
# TIDO용 tile 리스트 저장
tile_list_records = []  # {sid, rank, coord_y, coord_x, score, fold_scores...}

for s_idx, (sid, bulk) in enumerate(matched):
    if s_idx % 50 == 0:
        print(f'  [{s_idx+1}/{len(matched)}] {sid}')

    # 10개 fold의 tile_pred 수집
    fold_preds = []  # list of (T, G) arrays
    coords = np.load(f'{TCGA_EMB}/fold_01/{sid}_coords.npy')  # coords는 fold 공통

    for fold in FOLDS:
        embs = F.normalize(torch.tensor(
            np.load(f'{TCGA_EMB}/{fold}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
        train_embs = F.normalize(torch.tensor(
            np.load(f'{FT_EMB}/{fold}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
        train_expr = torch.tensor(
            np.load(f'{FT_EMB}/{fold}/train_exprs.npy'), dtype=torch.float32, device=device)

        with torch.no_grad():
            sim        = torch.clamp(embs @ train_embs.T, min=0)
            weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
            tile_preds = (weights @ train_expr).cpu().numpy()

        fold_preds.append(tile_preds[:, common_idx])  # (T, G)
        del embs, train_embs, train_expr

    torch.cuda.empty_cache()

    # 10 fold 평균 → ensemble pred
    ensemble_pred = np.mean(fold_preds, axis=0)  # (T, G)
    T = len(ensemble_pred)

    # ensemble pred vs bulk → spot-wise PCC score
    bulk_c = bulk - bulk.mean()
    tile_c = ensemble_pred - ensemble_pred.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
    scores = np.where(denom > 1e-8, num/denom, -999)

    valid  = scores > -999
    v_idx  = np.where(valid)[0]
    sorted_ = v_idx[np.argsort(scores[valid])[::-1]]

    # K별 sum aggregation
    for K in Ks:
        k   = min(K, T)
        top = sorted_[:k]
        ensemble_results[K]['preds'].append(ensemble_pred[top].sum(axis=0))
        ensemble_results[K]['bulks'].append(bulk)

    # Top-500 tile 리스트 저장 (TIDO용)
    top500_n = min(500, len(sorted_))
    top500   = sorted_[:top500_n]

    for rank, t in enumerate(top500):
        tile_list_records.append({
            'slide_id':  sid,
            'rank':      rank + 1,
            'coord_y':   int(coords[t, 0]),
            'coord_x':   int(coords[t, 1]),
            'score':     float(scores[t]),
            'total_tiles': T,
        })

# ── Step 2: Gene-wise PCC 계산 ────────────────────────────
print('\n' + '='*60)
print('Ensemble Ranking 결과')
print('='*60)
print(f'{"K":>6} | {"Gene-wise mean":>14} | {"Gene-wise med":>13} | {"Slide-wise":>11}')
print('-'*52)

for K in Ks:
    p_arr = np.array(ensemble_results[K]['preds'])
    b_arr = np.array(ensemble_results[K]['bulks'])
    gene_pccs = [pearsonr(p_arr[:,j], b_arr[:,j])[0]
                 for j in range(G)
                 if p_arr[:,j].std()>1e-8 and b_arr[:,j].std()>1e-8]
    slide_pccs = [pearsonr(p_arr[i], b_arr[i])[0]
                  for i in range(p_arr.shape[0])
                  if p_arr[i].std()>1e-8 and b_arr[i].std()>1e-8]
    g = np.array(gene_pccs); s = np.array(slide_pccs)
    print(f'{K:>6} | {g.mean():>14.4f} | {np.median(g):>13.4f} | {s.mean():>11.4f}')

# ── Step 3: Top-500 tile 리스트 저장 ──────────────────────
tile_df = pd.DataFrame(tile_list_records)
tile_df.to_csv(f'{OUT_DIR}/tile_lists/top500_tiles_ensemble.csv', index=False)
print(f'\nTop-500 tile list saved: {len(tile_df)} rows')
print(f'  Slides: {tile_df["slide_id"].nunique()}')
print(f'  Mean tiles per slide: {tile_df.groupby("slide_id").size().mean():.1f}')
print(f'  Score range: {tile_df["score"].min():.4f} ~ {tile_df["score"].max():.4f}')

# ── 슬라이드별 저장 (TIDO 입력 형식) ─────────────────────
slide_out_dir = f'{OUT_DIR}/tile_lists/per_slide'
os.makedirs(slide_out_dir, exist_ok=True)
for sid, grp in tile_df.groupby('slide_id'):
    grp.to_csv(f'{slide_out_dir}/{sid}_top500.csv', index=False)
print(f'  Per-slide CSVs saved to: {slide_out_dir}')

# ── 단일 fold vs 앙상블 비교 ──────────────────────────────
print('\n[단일 fold_03 vs 앙상블 비교]')
print(f'{"방법":<20} | {"K=50":>8} | {"K=100":>8} | {"K=300":>8} | {"K=500":>8}')
print('-'*52)

# fold_03 단일 결과 (이전 실험)
fold03_results = {50: 0.3795, 100: 0.3654, 300: 0.1772, 500: 0.1091}
ensemble_final = {}
for K in Ks:
    p_arr = np.array(ensemble_results[K]['preds'])
    b_arr = np.array(ensemble_results[K]['bulks'])
    gene_pccs = [pearsonr(p_arr[:,j], b_arr[:,j])[0]
                 for j in range(G)
                 if p_arr[:,j].std()>1e-8 and b_arr[:,j].std()>1e-8]
    ensemble_final[K] = np.array(gene_pccs).mean()

print(f'{"fold_03 single":<20} | {fold03_results[50]:>8.4f} | {fold03_results[100]:>8.4f} | {fold03_results[300]:>8.4f} | {fold03_results[500]:>8.4f}')
print(f'{"Ensemble (10 fold)":<20} | {ensemble_final[50]:>8.4f} | {ensemble_final[100]:>8.4f} | {ensemble_final[300]:>8.4f} | {ensemble_final[500]:>8.4f}')

print('\nDone!')
