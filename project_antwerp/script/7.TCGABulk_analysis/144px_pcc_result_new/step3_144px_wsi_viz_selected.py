"""
144px WSI 시각화 - 8개 슬라이드, Top-100 tile
- 슬라이드명 폴더별 저장
- WSI grayscale 위에 컬러 tile 표시
- K=100% vs K=30% 비교
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

TOP_K = 100

TARGET_SIDS = [
    'TCGA-D6-6824-01Z-00-DX2.96A2C7FA-59F2-4704-ABD0-2D215C442D7C.svs',
    'TCGA-MZ-A7D7-01Z-00-DX1.A9E4D12E-1B8B-4499-B805-7FBA8D69A7DA.svs',
    'TCGA-CN-4722-01Z-00-DX1.cf599bd8-c285-4f44-82d0-64f4b453d5e5.svs',
    'TCGA-CN-5359-01Z-00-DX1.30a19cad-c2b0-4c5f-bd5d-89aa8ac2bb91.svs',
    'TCGA-D6-6825-01Z-00-DX2.988BA622-0BBB-456A-8050-B52DA96A699A.svs',
    'TCGA-CQ-6223-01Z-00-DX1.95236dd8-13fd-4462-8f8c-a91b96c96802.svs',
    'TCGA-CV-5977-01Z-00-DX1.DF714997-E628-476C-BD4A-CEB52FEDABD3.svs',
    'TCGA-CQ-5330-01Z-00-DX1.a5651070-3cc9-4952-a947-8fc6aea0fde3.svs',
    'TCGA-CV-7252-01Z-00-DX1.01B710B3-B69F-4DE2-BF2B-06048EDA7A14.svs',
    'TCGA-CN-6022-01Z-00-DX1.46951ba2-f656-41b9-95c7-a3026ddbd562.svs',
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

N_ST = train_embs.shape[0]
K_30 = int(N_ST * 0.30)
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
    print(f'  WSI level={best_li}: {lH}x{lW}')
    return img, scale_x, scale_y

def make_gray_overlay(img_rgb):
    """RGB → grayscale을 RGB로 변환 (오버레이용)"""
    gray = np.array(Image.fromarray(img_rgb).convert('L'))
    return np.stack([gray, gray, gray], axis=-1)

def get_topbot(scores, k):
    vidx = np.where(scores > -999)[0]
    s    = vidx[np.argsort(scores[vidx])[::-1]]
    return s[:k], s[-k:]

def draw_overlay(ax, img_gray_rgb, coords, scores, scale_x, scale_y,
                 tile_w, tile_h, top_idx, bot_idx, label, short_id):
    lH, lW = img_gray_rgb.shape[:2]
    ax.imshow(img_gray_rgb, origin='upper', cmap='gray')

    # top-K: 빨강
    for rank, t_idx in enumerate(top_idx):
        x_lr = int(coords[t_idx, 0] * scale_x)
        y_lr = int(coords[t_idx, 1] * scale_y)
        if not (0 <= x_lr < lW and 0 <= y_lr < lH): continue
        ax.add_patch(mpatches.Rectangle((x_lr, y_lr), tile_w, tile_h,
            lw=1.5, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.45, zorder=4))
        if rank < 5:
            ax.text(x_lr + tile_w//2, y_lr - tile_h//2, f'T{rank+1}',
                    color='white', fontsize=7, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#e74c3c', alpha=0.85), zorder=6)

    # bot-K: 파랑
    for t_idx in bot_idx:
        x_lr = int(coords[t_idx, 0] * scale_x)
        y_lr = int(coords[t_idx, 1] * scale_y)
        if not (0 <= x_lr < lW and 0 <= y_lr < lH): continue
        ax.add_patch(mpatches.Rectangle((x_lr, y_lr), tile_w, tile_h,
            lw=1.5, edgecolor='#2980b9', facecolor='#2980b9', alpha=0.45, zorder=4))

    ax.set_title(
        f'{label}\nTop-{TOP_K} mean={scores[top_idx].mean():.4f}  '
        f'Bot-{TOP_K} mean={scores[bot_idx].mean():.4f}', fontsize=9)
    ax.axis('off')
    ax.legend(handles=[
        Line2D([0],[0], color='#e74c3c', lw=3, label=f'Top-{TOP_K} (high PCC)'),
        Line2D([0],[0], color='#2980b9', lw=3, label=f'Bot-{TOP_K} (low PCC)')],
        fontsize=8, loc='lower right')

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

    # 슬라이드명 폴더 생성
    short_id = sid.split('.')[0]  # e.g. TCGA-CV-6950-01Z-00-DX1
    OUT_DIR  = os.path.join(OUT_ROOT, short_id)
    os.makedirs(OUT_DIR, exist_ok=True)

    bulk = ref_df[ref_df['slide_id'] == sid][bulk_cols].values[0].astype(float)

    scores_100 = get_tile_pcc_scores(sid, bulk, k_spots=None)
    scores_30  = get_tile_pcc_scores(sid, bulk, k_spots=K_30)

    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
    T = min(len(coords), len(scores_100))
    coords     = coords[:T]
    scores_100 = scores_100[:T]
    scores_30  = scores_30[:T]
    print(f'  T={T}')

    img_rgb, scale_x, scale_y = load_wsi_lowres(wsi_map[sid])
    img_gray = make_gray_overlay(img_rgb)

    ux = np.unique(coords[:, 0])
    stride = int(np.median(np.diff(ux))) if len(ux) > 1 else 144
    tile_w = max(2, int(stride * scale_x))
    tile_h = max(2, int(stride * scale_y))

    top100_100, bot100_100 = get_topbot(scores_100, TOP_K)
    top100_30,  bot100_30  = get_topbot(scores_30,  TOP_K)

    # ── Figure 1: WSI grayscale + overlay (K=100% vs K=30%) ──
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(f'Top-{TOP_K} Tile Selection | {short_id}', fontsize=11)
    draw_overlay(axes[0], img_gray, coords, scores_100, scale_x, scale_y,
                 tile_w, tile_h, top100_100, bot100_100, 'K=100% (전체 spot)', short_id)
    draw_overlay(axes[1], img_gray, coords, scores_30,  scale_x, scale_y,
                 tile_w, tile_h, top100_30,  bot100_30,  f'K=30% ({K_30} spots)', short_id)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/wsi_overlay_top{TOP_K}.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: wsi_overlay_top{TOP_K}.png')

    # ── Figure 2: Score Heatmap (K=100% vs K=30%) ────────────
    x_vals = coords[:, 0]; y_vals = coords[:, 1]
    ux2 = np.unique(x_vals); uy2 = np.unique(y_vals)
    sx  = int(np.median(np.diff(ux2))) if len(ux2) > 1 else stride
    sy  = int(np.median(np.diff(uy2))) if len(uy2) > 1 else stride
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
            ax.text(gc[t_idx]+0.5, gr[t_idx], f'T{rank+1}',
                    fontsize=7, color='white', fontweight='bold')
        ax.set_title(f'{label} | top-{TOP_K} mean={scores[top_idx].mean():.4f}')
        ax.set_xlabel('Grid col'); ax.set_ylabel('Grid row')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/heatmap_top{TOP_K}.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: heatmap_top{TOP_K}.png')

    # ── Figure 3: Spatial position (grid dot plot) ────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
    fig3.suptitle(f'Top/Bot-{TOP_K} Spatial Position | {short_id}', fontsize=11)

    for ax, scores, top_idx, bot_idx, label in [
        (axes3[0], scores_100, top100_100, bot100_100, 'K=100%'),
        (axes3[1], scores_30,  top100_30,  bot100_30,  'K=30%'),
    ]:
        ax.scatter(gc, gr, s=2, c='#bdc3c7', alpha=0.3, zorder=1, label='All tiles')
        ax.scatter(gc[top_idx], gr[top_idx], s=15, c='#e74c3c', alpha=0.9, zorder=3, label=f'Top-{TOP_K}')
        ax.scatter(gc[bot_idx], gr[bot_idx], s=15, c='#2980b9', alpha=0.9, zorder=3, label=f'Bot-{TOP_K}')
        cx = gc.mean(); cy = gr.mean()
        ax.scatter([cx],[cy], s=100, c='black', marker='+', zorder=4, label='center')
        ax.set_title(f'{label}')
        ax.set_xlabel('Grid col'); ax.set_ylabel('Grid row')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/spatial_top{TOP_K}.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: spatial_top{TOP_K}.png')

print(f'\nAll done! Output: {OUT_ROOT}')
print('Folder structure:')
for sid in TARGET_SIDS:
    short_id = sid.split('.')[0]
    out_dir  = os.path.join(OUT_ROOT, short_id)
    if os.path.exists(out_dir):
        files = os.listdir(out_dir)
        print(f'  {short_id}/: {files}')