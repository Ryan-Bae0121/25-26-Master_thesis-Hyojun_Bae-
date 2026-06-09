"""
144px WSI 시각화 - 지정된 8개 슬라이드, Top-100 tile
K=30% spots vs K=100% spots 비교
"""

import numpy as np, torch, torch.nn.functional as F
import pandas as pd, os, glob, h5py, tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px/fold_03'
H5_DIR    = '/project_antwerp/hbae/data/TCGA_HNSC_tiles_144px_h5'
WSI_DIR   = '/project_antwerp/hbae/data/WSIs'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/wsi_visualization_144px'
os.makedirs(OUT_DIR, exist_ok=True)

TOP_K = 100

TARGET_SIDS = [
    'TCGA-CV-6950-01Z-00-DX1.1D8CFDF2-998C-45E2-ACDF-581B679DAD6F.svs',
    'TCGA-F7-A61V-01Z-00-DX1.6F80F8B8-59B1-471F-9BD0-FEB02AB314D3.svs',
    'TCGA-MZ-A7D7-01Z-00-DX1.A9E4D12E-1B8B-4499-B805-7FBA8D69A7DA.svs',
    'TCGA-DQ-7591-01Z-00-DX1.8304B939-542C-4D30-8C77-F705DE1311FF.svs',
    'TCGA-F7-A620-01Z-00-DX1.E0DA4A79-6F9B-4F7C-912E-4A4514DF8F49.svs',
    'TCGA-IQ-A61O-01Z-00-DX1.9ECA5801-3913-45B1-845D-13674F15439E.svs',
    'TCGA-T2-A6WZ-01Z-00-DX1.52078A9C-735F-4781-999E-3B65EA1C4174.svs',
    'TCGA-CV-6943-01Z-00-DX1.40AC4E32-4F41-4424-9D15-A2FB9FABB104.svs',
]

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

N_ST  = train_embs.shape[0]
K_30  = int(N_ST * 0.30)
print(f'Train spots: {N_ST}, K=30%: {K_30}')

wsi_map = {os.path.basename(f): f for f in glob.glob(f'{WSI_DIR}/*.svs')}

# ── Helper functions ──────────────────────────────────────
def get_tile_pcc_scores(sid, bulk, k_spots=None):
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim = torch.clamp(embs @ train_embs.T, min=0)
        if k_spots:
            tv, ti = torch.topk(sim, k=k_spots, dim=1)
            w = torch.zeros_like(sim)
            w.scatter_(1, ti, tv / (tv.sum(dim=1, keepdim=True) + 1e-8))
        else:
            w = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tp = (w @ train_expr).cpu().numpy()[:, common_idx]
    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum()) + 1e-8
    scores = np.where(denom > 1e-8, num / denom, -999)
    del embs
    torch.cuda.empty_cache()
    return scores

def load_wsi_lowres(wsi_path, target_px=2000):
    tif = tifffile.TiffFile(wsi_path)
    series = tif.series[0]
    fullH  = series.levels[0].shape[0]
    fullW  = series.levels[0].shape[1]
    best_li = 0
    for li in range(len(series.levels)):
        if series.levels[li].shape[0] >= target_px:
            best_li = li
    img = tifffile.imread(wsi_path, level=best_li)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    lH, lW = img.shape[:2]
    scale_x = lW / fullW
    scale_y = lH / fullH
    tif.close()
    print(f'  WSI level={best_li}: {lH}x{lW}, scale=({scale_x:.5f},{scale_y:.5f})')
    return img, scale_x, scale_y

def draw_wsi_topk(ax, img, coords, scores, scale_x, scale_y, tile_w_lr, tile_h_lr, top_k_idx, bot_k_idx, label):
    lH, lW = img.shape[:2]
    ax.imshow(img, origin='upper')
    for i in range(len(coords)):
        x_lr = int(coords[i, 0] * scale_x)
        y_lr = int(coords[i, 1] * scale_y)
        if 0 <= x_lr < lW and 0 <= y_lr < lH:
            ax.add_patch(mpatches.Rectangle(
                (x_lr, y_lr), tile_w_lr, tile_h_lr,
                lw=0.1, edgecolor='white', facecolor='none', alpha=0.1))
    for rank, t_idx in enumerate(top_k_idx):
        x_lr = int(coords[t_idx, 0] * scale_x)
        y_lr = int(coords[t_idx, 1] * scale_y)
        if not (0 <= x_lr < lW and 0 <= y_lr < lH): continue
        ax.add_patch(mpatches.Rectangle((x_lr, y_lr), tile_w_lr, tile_h_lr,
            lw=2, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.3, zorder=4))
        ax.add_patch(mpatches.Rectangle((x_lr, y_lr), tile_w_lr, tile_h_lr,
            lw=2, edgecolor='#e74c3c', facecolor='none', zorder=5))
        if rank < 5:
            ax.text(x_lr + tile_w_lr//2, y_lr - tile_h_lr//3, f'T{rank+1}',
                    color='#e74c3c', fontsize=8, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8), zorder=6)
    for t_idx in bot_k_idx:
        x_lr = int(coords[t_idx, 0] * scale_x)
        y_lr = int(coords[t_idx, 1] * scale_y)
        if not (0 <= x_lr < lW and 0 <= y_lr < lH): continue
        ax.add_patch(mpatches.Rectangle((x_lr, y_lr), tile_w_lr, tile_h_lr,
            lw=2, edgecolor='#2980b9', facecolor='#2980b9', alpha=0.25, zorder=4))
        ax.add_patch(mpatches.Rectangle((x_lr, y_lr), tile_w_lr, tile_h_lr,
            lw=2, edgecolor='#2980b9', facecolor='none', zorder=5))
    ax.set_title(f'{label}\nTop-{TOP_K} mean={scores[top_k_idx].mean():.4f}  Bot-{TOP_K} mean={scores[bot_k_idx].mean():.4f}', fontsize=9)
    ax.axis('off')
    ax.legend(handles=[
        Line2D([0],[0], color='#e74c3c', lw=3, label=f'Top-{TOP_K}'),
        Line2D([0],[0], color='#2980b9', lw=3, label=f'Bot-{TOP_K}')],
        fontsize=9, loc='lower right')

# ── 메인 루프 ─────────────────────────────────────────────
for sid in TARGET_SIDS:
    print(f'\n{"="*60}')
    print(f'Processing: {sid}')

    emb_path = f'{TCGA_EMB}/{sid}.npy'
    h5_path  = f'{H5_DIR}/{sid}/tiles.h5'
    if not os.path.exists(emb_path):
        print(f'  [SKIP] embedding not found'); continue
    if not os.path.exists(h5_path):
        print(f'  [SKIP] tiles.h5 not found'); continue
    if sid not in wsi_map:
        print(f'  [SKIP] WSI not found'); continue

    bulk = ref_df[ref_df['slide_id'] == sid][bulk_cols].values[0].astype(float)

    # 두 가지 방식으로 tile-wise PCC 계산
    scores_100 = get_tile_pcc_scores(sid, bulk, k_spots=None)   # K=100%
    scores_30  = get_tile_pcc_scores(sid, bulk, k_spots=K_30)   # K=30%

    # coords 로드
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]  # (T, 2): [x, y]
    T = min(len(coords), len(scores_100))
    coords     = coords[:T]
    scores_100 = scores_100[:T]
    scores_30  = scores_30[:T]
    print(f'  T={T}, K30={K_30}')

    img, scale_x, scale_y = load_wsi_lowres(wsi_map[sid])

    # tile size 추정
    ux = np.unique(coords[:, 0])
    stride_fullres = int(np.median(np.diff(ux))) if len(ux) > 1 else 144
    tile_w_lr = max(2, int(stride_fullres * scale_x))
    tile_h_lr = max(2, int(stride_fullres * scale_y))

    # top/bot K 인덱스
    def get_topbot(scores):
        vidx = np.where(scores > -999)[0]
        s    = vidx[np.argsort(scores[vidx])[::-1]]
        return s[:TOP_K], s[-TOP_K:]

    top100_100, bot100_100 = get_topbot(scores_100)
    top100_30,  bot100_30  = get_topbot(scores_30)

    # ── Figure: WSI 위 K=100% vs K=30% 비교 ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(f'Top-{TOP_K} Tile Selection on WSI\n{sid}', fontsize=11)

    draw_wsi_topk(axes[0], img, coords, scores_100, scale_x, scale_y,
                  tile_w_lr, tile_h_lr, top100_100, bot100_100, 'K=100% (전체 spot)')
    draw_wsi_topk(axes[1], img, coords, scores_30,  scale_x, scale_y,
                  tile_w_lr, tile_h_lr, top100_30,  bot100_30,  f'K=30% ({K_30} spots)')

    plt.tight_layout()
    short_id = sid.split('.')[0]
    out_path = f'{OUT_DIR}/wsi_{short_id}_top{TOP_K}_comparison.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')

    # ── Figure: Score Heatmap 비교 ────────────────────────
    x_vals = coords[:, 0]; y_vals = coords[:, 1]
    ux2 = np.unique(x_vals); uy2 = np.unique(y_vals)
    sx  = int(np.median(np.diff(ux2))) if len(ux2) > 1 else stride_fullres
    sy  = int(np.median(np.diff(uy2))) if len(uy2) > 1 else stride_fullres
    gc  = ((x_vals - x_vals.min()) / sx).astype(int)
    gr  = ((y_vals - y_vals.min()) / sy).astype(int)
    mr  = gr.max() + 1; mc = gc.max() + 1

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle(f'Score Heatmap | {short_id}', fontsize=11)

    for ax, scores, top_idx, label in [
        (axes2[0], scores_100, top100_100, 'K=100%'),
        (axes2[1], scores_30,  top100_30,  'K=30%'),
    ]:
        valid = scores[scores > -999]
        grid  = np.full((mr, mc), np.nan)
        for i in range(T):
            if scores[i] > -999:
                grid[gr[i], gc[i]] = scores[i]
        im = ax.imshow(grid, cmap='RdYlGn',
                       vmin=np.percentile(valid, 2),
                       vmax=np.percentile(valid, 98),
                       interpolation='nearest', origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.03)
        for rank, t_idx in enumerate(top_idx[:10]):
            ax.scatter(gc[t_idx], gr[t_idx], marker='*', s=200,
                       c='#e74c3c', zorder=5, edgecolors='white', lw=0.5)
        ax.set_title(f'{label} | top-{TOP_K} mean={scores[top_idx].mean():.4f}')
        ax.set_xlabel('Grid col'); ax.set_ylabel('Grid row')

    plt.tight_layout()
    out_path2 = f'{OUT_DIR}/heatmap_{short_id}_top{TOP_K}_comparison.png'
    plt.savefig(out_path2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path2}')

print(f'\nAll done! Output: {OUT_DIR}')