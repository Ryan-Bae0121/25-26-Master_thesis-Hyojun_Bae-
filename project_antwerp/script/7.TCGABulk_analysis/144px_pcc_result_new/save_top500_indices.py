"""
각 슬라이드별 tile-wise PCC 기준 Top-500 tile index 저장
- 입력: TCGA_embeddings_144px/fold_03/{sid}.npy (tile embeddings)
- 출력: top500_indices/{sid}.npy (h5 images 기준 index)
- 튜터가 이 index로 h5에서 top-500 이미지 바로 꺼낼 수 있음

사용 예시:
  idx = np.load('top500_indices/SLIDE_ID.svs.npy')  # (500,)
  with h5py.File('tiles.h5', 'r') as f:
      images = f['images'][np.sort(idx)]  # (500, 144, 144, 3) uint8
      coords = f['coords'][np.sort(idx)]  # (500, 2)
"""

import numpy as np, torch, torch.nn.functional as F
import pandas as pd, os, h5py
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────
device    = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px/fold_03'
H5_DIR    = '/project_antwerp/hbae/data/TCGA_HNSC_tiles_144px_h5'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/top500_indices'
TOP_K     = 500

os.makedirs(OUT_DIR, exist_ok=True)

# ── 유전자 리스트 & ref_file ───────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df['slide_id'] = ref_df['wsi_file_name'] + '.svs'
rna_cols     = [c for c in ref_df.columns if c.startswith('rna_')]
ref_genes    = [c.replace('rna_', '') for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
common_idx   = [gene_list.index(g) for g in common_genes]
bulk_cols    = ['rna_' + g for g in common_genes]

# ── Train embedding 로드 ───────────────────────────────────
train_embs = F.normalize(torch.tensor(
    np.load(f'{FT_EMB}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
train_expr = torch.tensor(
    np.load(f'{FT_EMB}/train_exprs.npy'), dtype=torch.float32, device=device)

# ── 매칭 슬라이드 ─────────────────────────────────────────
matched = [
    (row['slide_id'], row[bulk_cols].values.astype(float))
    for _, row in ref_df.iterrows()
    if os.path.exists(f'{TCGA_EMB}/{row["slide_id"]}.npy') and
       os.path.exists(f'{H5_DIR}/{row["slide_id"]}/tiles.h5')
]
print(f'Matched slides: {len(matched)}')
print(f'Train spots: {train_embs.shape[0]}')
print(f'Saving top-{TOP_K} indices to: {OUT_DIR}')

# ── 메인 루프 ─────────────────────────────────────────────
summary_rows = []

for sid, bulk in tqdm(matched, desc='Computing tile-wise PCC'):

    # 이미 처리된 슬라이드 skip
    if os.path.exists(f'{OUT_DIR}/{sid}.npy'):
        continue

    # tile embedding 로드
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)

    # PredEx: tile별 유전자 발현 예측
    with torch.no_grad():
        sim = torch.clamp(embs @ train_embs.T, min=0)      # (T, N_train)
        w   = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)  # 가중치 정규화
        tp  = (w @ train_expr).cpu().numpy()[:, common_idx] # (T, G)

    # GPU 벡터화 tile-wise PCC
    P     = torch.tensor(tp,   dtype=torch.float32, device=device)
    b     = torch.tensor(bulk, dtype=torch.float32, device=device)
    P_c   = P - P.mean(dim=1, keepdim=True)
    b_c   = b - b.mean()
    scores = ((P_c * b_c).sum(dim=1) /
              (P_c.norm(dim=1) * b_c.norm() + 1e-8)).cpu().numpy()  # (T,)

    T = len(scores)
    k = min(TOP_K, T)

    # valid tile (PCC > -999) 중 상위 K개
    valid_idx    = np.where(scores > -999)[0]
    sorted_valid = valid_idx[np.argsort(scores[valid_idx])[::-1]]
    top_k_idx    = sorted_valid[:k]  # h5 images 기준 index, 정렬 안 된 상태

    # index 저장 (h5에서 꺼낼 때 np.sort() 해서 사용)
    np.save(f'{OUT_DIR}/{sid}.npy', top_k_idx)

    summary_rows.append({
        'slide_id':       sid,
        'total_tiles':    T,
        'valid_tiles':    int(len(valid_idx)),
        'top_k':          k,
        'top_k_mean_pcc': float(scores[top_k_idx].mean()),
        'top_k_min_pcc':  float(scores[top_k_idx].min()),
        'all_mean_pcc':   float(scores[valid_idx].mean()),
    })

    del embs, P, b, P_c, b_c
    torch.cuda.empty_cache()

# ── summary CSV 저장 ──────────────────────────────────────
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{OUT_DIR}/summary.csv', index=False)
    print(f'\nSummary stats:')
    print(f'  Processed:          {len(summary_rows)} slides')
    print(f'  Mean top-{TOP_K} PCC:  {summary_df["top_k_mean_pcc"].mean():.4f}')
    print(f'  Mean all-tile PCC:  {summary_df["all_mean_pcc"].mean():.4f}')
else:
    print('\nAll slides already processed (skipped).')

print(f'\nDone! Output: {OUT_DIR}')
print(f'\n── 사용 방법 ──────────────────────────────────────')
print(f'import numpy as np, h5py')
print(f'idx = np.load("{OUT_DIR}/SLIDE_ID.svs.npy")  # (500,) index array')
print(f'with h5py.File("{H5_DIR}/SLIDE_ID.svs/tiles.h5", "r") as f:')
print(f'    images = f["images"][np.sort(idx)]  # (500, 144, 144, 3) uint8')
print(f'    coords = f["coords"][np.sort(idx)]  # (500, 2)')