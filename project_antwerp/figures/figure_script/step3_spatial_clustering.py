"""
Top-K tile spatial clustering 시각화
K = 100, 200, 300, 400, 500
각 슬라이드별 폴더, K별 파일 저장
핵심: top/bot tile들이 공간적으로 모이는지 확인
"""

import numpy as np, torch, torch.nn.functional as F
import pandas as pd, os, glob, h5py, tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px/fold_03'
H5_DIR    = '/project_antwerp/hbae/data/TCGA_HNSC_tiles_144px_h5'
WSI_DIR   = '/project_antwerp/hbae/data/WSIs'
OUT_ROOT  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/wsi_visualization_144px'

K_LIST = [100, 200, 300, 400, 500]

TARGET_SIDS = [
    'TCGA-CV-6950-01Z-00-DX1.1D8CFDF2-998C-45E2-ACDF-581B679DAD6F.svs',
    'TCGA-F7-A61V-01Z-00-DX1.6F80F8B8-59B1-471F-9BD0-FEB02AB314D3.svs',
    'TCGA-MZ-A7D7-01Z-00-DX1.A9E4D12E-1B8B-4499-B805-7FBA8D69A7DA.svs',
    'TCGA-DQ-7591-01Z-00-DX1.8304B939-542C-4D30-8C77-F705DE1311FF.svs',
    'TCGA-F7-A620-01Z-00-DX1.E0DA4A79-6F9B-4F7C-912E-4A4514DF8F49.svs',
    'TCGA-IQ-A61O-01Z-00-DX1.9ECA5801-3913-45B1-845D-13674F15439E.svs',
    'TCGA-T2-A6WZ-01Z-00-DX1.52078A9C-735F-4781-999E-3B65EA1C4174.svs',
    'TCGA-CV-6943-01Z-00-DX1.40AC4E32-4F41-4424-9D15-A2FB9FABB104.svs',
    'TCGA-D6-6824-01Z-00-DX2.96A2C7FA-59F2-4704-ABD0-2D215C442D7C.svs',
    'TCGA-CN-4722-01Z-00-DX1.cf599bd8-c285-4f44-82d0-64f4b453d5e5.svs',
    'TCGA-CN-5359-01Z-00-DX1.30a19cad-c2b0-4c5f-bd5d-89aa8ac2bb91.svs',
    'TCGA-D6-6825-01Z-00-DX2.988BA622-0BBB-456A-8050-B52DA96A699A.svs',
    'TCGA-CQ-6223-01Z-00-DX1.95236dd8-13fd-4462-8f8c-a91b96c96802.svs',
    'TCGA-CV-5977-01Z-00-DX1.DF714997-E628-476C-BD4A-CEB52FEDABD3.svs',
    'TCGA-CQ-5330-01Z-00-DX1.a5651070-3cc9-4952-a947-8fc6aea0fde3.svs',
    'TCGA-CV-7252-01Z-00-DX1.01B710B3-B69F-4DE2-BF2B-06048EDA7A14.svs',
    'TCGA-CN-6022-01Z-00-DX1.46951ba2-f656-41b9-95c7-a3026ddbd562.svs',
]

# ── 데이터 로드 ───────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]
ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df['slide_id'] = ref_df['wsi_file_name'] + '.svs'
rna_cols     = [c for c in ref_df.columns if c.startswith('rna_')]
ref_genes    = [c.replace('rna_', '') for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
common_idx   = [gene_list.index(g) for g in common_genes]
bulk_cols    = ['rna_' + g for g in common_genes]

train_embs = F.normalize(torch.tensor(
    np.load(f'{FT_EMB}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
train_expr = torch.tensor(
    np.load(f'{FT_EMB}/train_exprs.npy'), dtype=torch.float32, device=device)

wsi_map = {os.path.basename(f): f for f in glob.glob(f'{WSI_DIR}/*.svs')}

# ── Helper ────────────────────────────────────────────────
def get_tile_pcc_scores(sid, bulk):
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim = torch.clamp(embs @ train_embs.T, min=0)
        w   = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tp  = (w @ train_expr).cpu().numpy()[:, common_idx]
    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum()) + 1e-8
    scores = np.where(denom > 1e-8, num / denom, -999)
    del embs; torch.cuda.empty_cache()
    return scores

def load_wsi_gray(wsi_path, target_px=2000):
    tif    = tifffile.TiffFile(wsi_path)
    series = tif.series[0]
    fullH  = series.levels[0].shape[0]
    fullW  = series.levels[0].shape[1]
    best_li = 0
    for li in range(len(series.levels)):
        if series.levels[li].shape[0] >= target_px:
            best_li = li
    img = tifffile.imread(wsi_path, level=best_li)
    if img.ndim == 3 and img.shape[2] == 4: img = img[:,:,:3]
    gray = np.array(Image.fromarray(img).convert('L'))
    img_gray = np.stack([gray, gray, gray], axis=-1)
    lH, lW   = img_gray.shape[:2]
    tif.close()
    return img_gray, lW / fullW, lH / fullH

def spatial_clustering_score(gc, gr, top_idx, bot_idx):
    """top/bot tile들의 평균 nearest-neighbor 거리 (낮을수록 클러스터링됨)"""
    def mean_nn_dist(idx):
        if len(idx) < 2: return np.nan
        pts = np.stack([gc[idx], gr[idx]], axis=1).astype(float)
        dists = []
        for i in range(len(pts)):
            d = np.sqrt(((pts - pts[i])**2).sum(axis=1))
            d[i] = np.inf
            dists.append(d.min())
        return np.mean(dists)
    return mean_nn_dist(top_idx), mean_nn_dist(bot_idx)

# ── 메인 루프 ─────────────────────────────────────────────
for sid in TARGET_SIDS:
    print(f'\n{"="*60}')
    print(f'Processing: {sid}')

    emb_path = f'{TCGA_EMB}/{sid}.npy'
    h5_path  = f'{H5_DIR}/{sid}/tiles.h5'
    if not os.path.exists(emb_path): print('  [SKIP] no embedding'); continue
    if not os.path.exists(h5_path):  print('  [SKIP] no h5');        continue
    if sid not in wsi_map:           print('  [SKIP] no WSI');        continue

    short_id = sid.split('.')[0]
    OUT_DIR  = os.path.join(OUT_ROOT, short_id)
    os.makedirs(OUT_DIR, exist_ok=True)

    bulk   = ref_df[ref_df['slide_id']==sid][bulk_cols].values[0].astype(float)
    scores = get_tile_pcc_scores(sid, bulk)

    with h5py.File(h5_path,'r') as f:
        coords = f['coords'][:]
    T      = min(len(coords), len(scores))
    coords = coords[:T]; scores = scores[:T]
    print(f'  T={T}, valid={(scores>-999).sum()}')

    img_gray, scale_x, scale_y = load_wsi_gray(wsi_map[sid])
    lH, lW = img_gray.shape[:2]

    ux     = np.unique(coords[:,0]); uy = np.unique(coords[:,1])
    sx     = int(np.median(np.diff(ux))) if len(ux)>1 else 144
    sy     = int(np.median(np.diff(uy))) if len(uy)>1 else 144
    gc     = ((coords[:,0]-coords[:,0].min())/sx).astype(int)
    gr     = ((coords[:,1]-coords[:,1].min())/sy).astype(int)
    tile_w = max(2, int(sx * scale_x))
    tile_h = max(2, int(sy * scale_y))

    vidx   = np.where(scores > -999)[0]
    sorted_idx = vidx[np.argsort(scores[vidx])[::-1]]

    # 클러스터링 점수 기록
    cluster_rows = []

    # ── K별 그림 ─────────────────────────────────────────
    for K in K_LIST:
        top_idx = sorted_idx[:K]
        bot_idx = sorted_idx[-K:]
        top_nn, bot_nn = spatial_clustering_score(gc, gr, top_idx, bot_idx)
        cluster_rows.append({'K': K,
                              'top_mean_PCC': scores[top_idx].mean(),
                              'bot_mean_PCC': scores[bot_idx].mean(),
                              'top_NN_dist':  top_nn,
                              'bot_NN_dist':  bot_nn})

        # Figure: grayscale WSI + overlay (좌) / spatial dot (우)
        fig, axes = plt.subplots(1, 2, figsize=(22, 10),
                                 gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(
            f'{short_id}\nTop/Bot-{K} tile selection  '
            f'(top mean PCC={scores[top_idx].mean():.4f}, '
            f'bot mean PCC={scores[bot_idx].mean():.4f})\n'
            f'Top NN dist={top_nn:.2f}  Bot NN dist={bot_nn:.2f}  '
            f'(낮을수록 공간적으로 뭉침)',
            fontsize=10)

        # 왼쪽: WSI overlay
        ax = axes[0]
        ax.imshow(img_gray, origin='upper')
        for t_idx in bot_idx:
            x_lr = int(coords[t_idx,0]*scale_x); y_lr = int(coords[t_idx,1]*scale_y)
            if 0<=x_lr<lW and 0<=y_lr<lH:
                ax.add_patch(mpatches.Rectangle((x_lr,y_lr), tile_w, tile_h,
                    lw=1, edgecolor='#2980b9', facecolor='#2980b9', alpha=0.5, zorder=3))
        for t_idx in top_idx:
            x_lr = int(coords[t_idx,0]*scale_x); y_lr = int(coords[t_idx,1]*scale_y)
            if 0<=x_lr<lW and 0<=y_lr<lH:
                ax.add_patch(mpatches.Rectangle((x_lr,y_lr), tile_w, tile_h,
                    lw=1, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.5, zorder=4))
        ax.axis('off')
        ax.set_title(f'WSI overlay  (red=Top-{K}, blue=Bot-{K})', fontsize=9)
        ax.legend(handles=[
            Line2D([0],[0],color='#e74c3c',lw=3,label=f'Top-{K} (high PCC)'),
            Line2D([0],[0],color='#2980b9',lw=3,label=f'Bot-{K} (low PCC)')],
            fontsize=9, loc='lower right')

        # 오른쪽: spatial dot plot
        ax2 = axes[1]
        ax2.scatter(gc, gr, s=1, c='#cccccc', alpha=0.2, zorder=1, label='All')
        ax2.scatter(gc[bot_idx], gr[bot_idx], s=8, c='#2980b9',
                    alpha=0.7, zorder=3, label=f'Bot-{K}')
        ax2.scatter(gc[top_idx], gr[top_idx], s=8, c='#e74c3c',
                    alpha=0.9, zorder=4, label=f'Top-{K}')
        ax2.set_xlabel('Grid col'); ax2.set_ylabel('Grid row')
        ax2.set_title(f'Spatial position\nNN dist: top={top_nn:.2f}, bot={bot_nn:.2f}', fontsize=9)
        ax2.invert_yaxis()
        ax2.legend(fontsize=8, markerscale=2)
        ax2.grid(alpha=0.2)
        ax2.set_aspect('equal')

        plt.tight_layout()
        fname = f'top{K}_bot{K}_overlay.png'
        plt.savefig(f'{OUT_DIR}/{fname}', dpi=130, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {fname}  (top_NN={top_nn:.2f}, bot_NN={bot_nn:.2f})')

    # ── K별 비교 summary figure ───────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle(f'K별 비교 | {short_id}', fontsize=11)

    Ks      = [r['K']            for r in cluster_rows]
    top_pcc = [r['top_mean_PCC'] for r in cluster_rows]
    bot_pcc = [r['bot_mean_PCC'] for r in cluster_rows]
    top_nn  = [r['top_NN_dist']  for r in cluster_rows]
    bot_nn  = [r['bot_NN_dist']  for r in cluster_rows]

    axes3[0].plot(Ks, top_pcc, 'o-', color='#e74c3c', label='Top-K mean PCC')
    axes3[0].plot(Ks, bot_pcc, 'o-', color='#2980b9', label='Bot-K mean PCC')
    axes3[0].set_xlabel('K'); axes3[0].set_ylabel('Mean tile-wise PCC')
    axes3[0].set_title('Mean PCC vs K'); axes3[0].legend(); axes3[0].grid(alpha=0.3)

    axes3[1].plot(Ks, top_nn, 'o-', color='#e74c3c', label='Top-K NN dist')
    axes3[1].plot(Ks, bot_nn, 'o-', color='#2980b9', label='Bot-K NN dist')
    axes3[1].set_xlabel('K'); axes3[1].set_ylabel('Mean NN distance (grid units)')
    axes3[1].set_title('Spatial Clustering (↓ = more clustered)')
    axes3[1].legend(); axes3[1].grid(alpha=0.3)

    # score histogram + K별 threshold
    valid_s = scores[scores > -999]
    axes3[2].hist(valid_s, bins=60, color='#95a5a6', alpha=0.6, density=True)
    colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']
    for K, c in zip(K_LIST, colors):
        thr = scores[sorted_idx[K-1]]
        axes3[2].axvline(thr, color=c, lw=1.5, ls='--', label=f'K={K} thr={thr:.3f}')
    axes3[2].set_xlabel('Tile-wise PCC'); axes3[2].set_ylabel('Density')
    axes3[2].set_title('Score distribution + K thresholds')
    axes3[2].legend(fontsize=7); axes3[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/summary_K_comparison.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: summary_K_comparison.png')

    # CSV 저장
    pd.DataFrame(cluster_rows).to_csv(f'{OUT_DIR}/clustering_scores.csv', index=False)
    print(f'  Saved: clustering_scores.csv')

print(f'\nAll done! Output: {OUT_ROOT}')