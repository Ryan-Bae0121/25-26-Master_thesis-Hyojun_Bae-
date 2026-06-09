nohup python3 << 'EOF' > /tmp/wsi_viz3.log 2>&1 &
echo $!

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
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
HDF5_DIR  = '/project_antwerp/TCGA-HNSC/TCGA_patch'
WSI_DIR   = '/project_antwerp/hbae/data/WSIs'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/wsi_visualization'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 좌표 확정 ──────────────────────────────────────────────
# key = "x_y" (a=x_topleft, b=y_topleft, level0 fullres)
# HDF5 tile = level0에서 512×512 crop → 256 resize
# 시각화: WSI lowres 위에 x,y 좌표로 박스 그리기
CROP_PX = 512  # level0 기준 실제 crop 크기

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

train_embs = F.normalize(torch.tensor(
    np.load(f'{FT_EMB}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
train_expr = torch.tensor(
    np.load(f'{FT_EMB}/train_exprs.npy'), dtype=torch.float32, device=device)
N_ST = len(train_embs)
K_30 = int(N_ST * 0.30)

wsi_map = {os.path.basename(f).split('.')[0]: f
           for f in glob.glob(f'{WSI_DIR}/*.svs')}

# 슬라이드 선택
selected = []
for _, row in ref_df.iterrows():
    sid = row['slide_id']
    hdf5 = f'{HDF5_DIR}/{sid}/{sid}.hdf5'
    if (os.path.exists(f'{TCGA_EMB}/{sid}.npy') and
        os.path.exists(hdf5) and sid in wsi_map):
        n = np.load(f'{TCGA_EMB}/{sid}.npy').shape[0]
        if 1000 <= n <= 8000:
            selected.append((sid, row[bulk_cols].values.astype(float),
                             wsi_map[sid], hdf5, n))
selected = selected[:3]
print(f'Selected: {[s[0] for s in selected]}')

def get_scores(sid, bulk, k_spots=None):
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim_pos = torch.clamp(embs @ train_embs.T, min=0)
        if k_spots:
            tv, ti = torch.topk(sim_pos, k=k_spots, dim=1)
            w = torch.zeros_like(sim_pos)
            w.scatter_(1, ti, tv/(tv.sum(dim=1,keepdim=True)+1e-8))
        else:
            w = sim_pos / (sim_pos.sum(dim=1,keepdim=True)+1e-8)
        tp = (w @ train_expr).cpu().numpy()[:, common_idx]
    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num   = (tile_c * bulk_c).sum(axis=1)
    denom = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
    scores = np.where(denom>1e-8, num/denom, -999)
    del embs; torch.cuda.empty_cache()
    return scores

def load_coords(hdf5_path):
    # key = "x_y" → x=col0, y=col1 (level0 fullres)
    with h5py.File(hdf5_path, 'r') as f:
        keys = list(f.keys())
    coords = np.array([[int(k.split('_')[0]), int(k.split('_')[1])]
                        for k in keys])  # (T,2): [x, y]
    return coords

def load_wsi_lowres(wsi_path, target_px=2000):
    tif = tifffile.TiffFile(wsi_path)
    series = tif.series[0]
    fullH = series.levels[0].shape[0]
    fullW = series.levels[0].shape[1]
    # target_px에 가장 가까운 레벨
    best_li = 0
    for li in range(len(series.levels)):
        h = series.levels[li].shape[0]
        if h >= target_px:
            best_li = li
    img = tifffile.imread(wsi_path, level=best_li)
    if img.ndim==3 and img.shape[2]==4: img=img[:,:,:3]
    lH, lW = img.shape[:2]
    scale_x = lW / fullW
    scale_y = lH / fullH
    tif.close()
    print(f'  WSI level={best_li}: {lH}×{lW}, scale=({scale_x:.5f},{scale_y:.5f})')
    return img, scale_x, scale_y

# ── Figure 1: Violin Plot ─────────────────────────────────
print('=== Violin Plot ===')
fig1, axes = plt.subplots(1, len(selected), figsize=(6*len(selected), 8))
if len(selected)==1: axes=[axes]
fig1.suptitle('Tile-wise PCC Distribution\nAll spots vs K=30%', fontsize=13)

for ax, (sid, bulk, wsi_path, hdf5_path, n_emb) in zip(axes, selected):
    s_all = get_scores(sid, bulk)
    s_k30 = get_scores(sid, bulk, K_30)
    v_all = s_all[s_all>-999]
    v_k30 = s_k30[s_k30>-999]

    vp = ax.violinplot([v_all, v_k30], positions=[1,2],
                        showmedians=True, showmeans=True, showextrema=True)
    vp['bodies'][0].set_facecolor('#95a5a6'); vp['bodies'][0].set_alpha(0.7)
    vp['bodies'][1].set_facecolor('#e74c3c'); vp['bodies'][1].set_alpha(0.7)
    for vals, pos, c in [(v_all,1,'#2c3e50'),(v_k30,2,'#c0392b')]:
        top3 = np.sort(vals)[::-1][:3]
        ax.scatter([pos]*3, top3, s=120, zorder=6, color=c, marker='*')
    ax.set_xticks([1,2])
    ax.set_xticklabels([f'All spots\n(100%)\nμ={v_all.mean():.4f}\nσ={v_all.std():.5f}',
                        f'K=30%\n({K_30})\nμ={v_k30.mean():.4f}\nσ={v_k30.std():.5f}'], fontsize=9)
    ax.set_ylabel('Tile-wise PCC')
    ax.set_title(f'{sid[-16:]}\nT={n_emb}', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/violin_v3.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: violin_v3.png')

# ── Figure 2+3: WSI + Grid Heatmap ───────────────────────
for sid, bulk, wsi_path, hdf5_path, n_emb in selected:
    print(f'\nProcessing {sid}...')
    s_all = get_scores(sid, bulk)
    s_k30 = get_scores(sid, bulk, K_30)
    coords = load_coords(hdf5_path)  # (T,2): [x, y] level0 fullres
    T = min(len(coords), len(s_all))
    coords = coords[:T]; s_all = s_all[:T]; s_k30 = s_k30[:T]

    img, scale_x, scale_y = load_wsi_lowres(wsi_path)
    lH, lW = img.shape[:2]

    # tile 크기 (lowres 기준) - level0에서 512px crop이었으므로
    tile_w_lr = int(CROP_PX * scale_x)
    tile_h_lr = int(CROP_PX * scale_y)

    vidx = np.where((s_all>-999)&(s_k30>-999))[0]

    def top_bot3(scores, vidx):
        vs = scores[vidx]
        order = np.argsort(vs)[::-1]
        return vidx[order[:3]], vidx[order[-3:]]

    top3_all, bot3_all = top_bot3(s_all, vidx)
    top3_k30, bot3_k30 = top_bot3(s_k30, vidx)

    # ── WSI 시각화 ────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(22, 11))
    fig2.suptitle(f'Top-3 / Bot-3 Tile on WSI\n{sid}  (T={T})', fontsize=12)

    for ax, top3, bot3, scores, label, ct, cb in [
        (axes2[0], top3_all, bot3_all, s_all, 'All spots (100%)', '#e74c3c', '#2980b9'),
        (axes2[1], top3_k30, bot3_k30, s_k30, f'K=30% ({K_30})', '#e74c3c', '#2980b9'),
    ]:
        ax.imshow(img, origin='upper')

        # 전체 tile 위치 (반투명)
        for i in range(T):
            x_lr = int(coords[i,0] * scale_x)
            y_lr = int(coords[i,1] * scale_y)
            if 0<=x_lr<lW and 0<=y_lr<lH:
                rect = mpatches.Rectangle(
                    (x_lr, y_lr), tile_w_lr, tile_h_lr,
                    lw=0.2, edgecolor='white', facecolor='none', alpha=0.15)
                ax.add_patch(rect)

        # top-3
        for rank, t_idx in enumerate(top3):
            x_lr = int(coords[t_idx,0] * scale_x)
            y_lr = int(coords[t_idx,1] * scale_y)
            if not (0<=x_lr<lW and 0<=y_lr<lH): continue
            rect = mpatches.Rectangle(
                (x_lr, y_lr), tile_w_lr, tile_h_lr,
                lw=4, edgecolor=ct, facecolor=ct, alpha=0.3, zorder=5)
            ax.add_patch(rect)
            rect2 = mpatches.Rectangle(
                (x_lr, y_lr), tile_w_lr, tile_h_lr,
                lw=4, edgecolor=ct, facecolor='none', zorder=6)
            ax.add_patch(rect2)
            ax.text(x_lr+tile_w_lr//2, y_lr-tile_h_lr//4,
                    f'T{rank+1}  {scores[t_idx]:.4f}',
                    color=ct, fontsize=10, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.85), zorder=7)

        # bot-3
        for rank, t_idx in enumerate(bot3):
            x_lr = int(coords[t_idx,0] * scale_x)
            y_lr = int(coords[t_idx,1] * scale_y)
            if not (0<=x_lr<lW and 0<=y_lr<lH): continue
            rect = mpatches.Rectangle(
                (x_lr, y_lr), tile_w_lr, tile_h_lr,
                lw=4, edgecolor=cb, facecolor=cb, alpha=0.25, zorder=5)
            ax.add_patch(rect)
            rect2 = mpatches.Rectangle(
                (x_lr, y_lr), tile_w_lr, tile_h_lr,
                lw=4, edgecolor=cb, facecolor='none', zorder=6)
            ax.add_patch(rect2)
            ax.text(x_lr+tile_w_lr//2, y_lr+tile_h_lr+tile_h_lr//4,
                    f'B{rank+1}  {scores[t_idx]:.4f}',
                    color=cb, fontsize=10, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.85), zorder=7)

        ax.set_title(f'{label}\nTop-3={scores[top3].mean():.4f}  Bot-3={scores[bot3].mean():.4f}',
                     fontsize=10)
        ax.axis('off')
        ax.legend(handles=[Line2D([0],[0],color=ct,lw=3,label='Top-3 (high PCC)'),
                            Line2D([0],[0],color=cb,lw=3,label='Bot-3 (low PCC)')],
                  fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/wsi_{sid}_v3.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: wsi_{sid}_v3.png')

    # ── Grid Heatmap + Spatial Position ───────────────────
    for k_label, scores, top3, bot3 in [
        ('All_100pct', s_all, top3_all, bot3_all),
        ('K30pct',     s_k30, top3_k30, bot3_k30),
    ]:
        # grid 좌표
        x_vals = coords[:T,0]; y_vals = coords[:T,1]
        # stride = 가장 작은 간격
        ux = np.unique(x_vals); uy = np.unique(y_vals)
        stride_x = int(np.median(np.diff(ux))) if len(ux)>1 else CROP_PX
        stride_y = int(np.median(np.diff(uy))) if len(uy)>1 else CROP_PX
        grid_col = ((x_vals - x_vals.min()) / stride_x).astype(int)
        grid_row = ((y_vals - y_vals.min()) / stride_y).astype(int)
        max_row = grid_row.max()+1; max_col = grid_col.max()+1

        valid_scores = scores[scores>-999]
        vmin = np.percentile(valid_scores, 2)
        vmax = np.percentile(valid_scores, 98)

        fig3 = plt.figure(figsize=(20,7))
        fig3.suptitle(f'{sid} | {k_label} | T={T}', fontsize=11)
        gs = GridSpec(1, 3, figure=fig3, wspace=0.3)

        # Panel 1: 분포 히스토그램
        ax1 = fig3.add_subplot(gs[0,0])
        ax1.hist(valid_scores, bins=60, color='#3498db', alpha=0.75, density=True)
        for t_idx in top3:
            ax1.axvline(scores[t_idx], color='#e74c3c', lw=2, alpha=0.9)
        for t_idx in bot3:
            ax1.axvline(scores[t_idx], color='#2980b9', lw=2, ls='--', alpha=0.9)
        ax1.set_xlabel('Tile PCC score'); ax1.set_ylabel('Density')
        ax1.set_title('Score Distribution\nred=top3, blue=bot3')
        ax1.legend(handles=[Line2D([0],[0],color='#e74c3c',lw=2,label='top-3'),
                             Line2D([0],[0],color='#2980b9',lw=2,ls='--',label='bot-3')],fontsize=9)
        ax1.grid(alpha=0.3)

        # Panel 2: Score Heatmap
        ax2 = fig3.add_subplot(gs[0,1])
        score_grid = np.full((max_row, max_col), np.nan)
        for i in range(T):
            if scores[i]>-999:
                score_grid[grid_row[i], grid_col[i]] = scores[i]
        im = ax2.imshow(score_grid, cmap='RdYlGn', vmin=vmin, vmax=vmax,
                        interpolation='nearest', origin='upper')
        plt.colorbar(im, ax=ax2, fraction=0.03)
        for rank, t_idx in enumerate(top3):
            ax2.scatter(grid_col[t_idx], grid_row[t_idx],
                        marker='*', s=300, c='#e74c3c', zorder=5, edgecolors='white', lw=0.5)
            ax2.text(grid_col[t_idx]+0.5, grid_row[t_idx],
                     f'T{rank+1}', fontsize=8, color='white', fontweight='bold')
        for rank, t_idx in enumerate(bot3):
            ax2.scatter(grid_col[t_idx], grid_row[t_idx],
                        marker='X', s=200, c='#2980b9', zorder=5, edgecolors='white', lw=0.5)
        ax2.set_title('Score Heatmap\n★=Top-3  X=Bot-3')
        ax2.set_xlabel('Grid col'); ax2.set_ylabel('Grid row')
        ax2.legend(handles=[Line2D([0],[0],marker='*',color='w',markerfacecolor='#e74c3c',markersize=12,label='Top-3'),
                             Line2D([0],[0],marker='X',color='w',markerfacecolor='#2980b9',markersize=10,label='Bot-3')],
                   fontsize=9, loc='lower right')

        # Panel 3: Spatial Position
        ax3 = fig3.add_subplot(gs[0,2])
        ax3.scatter(grid_col, grid_row, s=4, c='#bdc3c7', alpha=0.4, zorder=1)
        cx = grid_col.mean(); cy = grid_row.mean()
        ax3.scatter([cx],[cy], s=150, c='black', marker='+', zorder=4, label='center')
        for rank, t_idx in enumerate(top3):
            ax3.scatter(grid_col[t_idx], grid_row[t_idx],
                        marker='*', s=400, c='#e74c3c', zorder=5, edgecolors='white', lw=0.8)
            ax3.text(grid_col[t_idx]+0.5, grid_row[t_idx],
                     f'T{rank+1}\n({grid_col[t_idx]},{grid_row[t_idx]})',
                     fontsize=8, color='#c0392b', fontweight='bold')
        ax3.set_title('Top-3 Spatial Position')
        ax3.set_xlabel('Grid col'); ax3.set_ylabel('Grid row')
        ax3.invert_yaxis()
        ax3.legend(fontsize=9); ax3.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/grid_{sid}_{k_label}.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f'  Saved: grid_{sid}_{k_label}.png')

print('\nAll done!')
EOF