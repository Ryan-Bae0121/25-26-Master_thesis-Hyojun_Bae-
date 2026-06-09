
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tile_selection_vis_v2'
os.makedirs(OUT_DIR, exist_ok=True)

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

for sid in sids:
    row = ref_df[ref_df['slide_id'] == sid]
    if len(row) == 0:
        continue
    bulk = row.iloc[0][bulk_cols].values.astype(float)

    emb_path    = f'{TCGA_EMB}/{sid}.npy'
    coords_path = f'{TCGA_EMB}/{sid}_coords.npy'
    if not os.path.exists(emb_path):
        continue

    embs   = F.normalize(torch.tensor(
        np.load(emb_path), dtype=torch.float32, device=device), dim=-1)
    coords = np.load(coords_path)

    with torch.no_grad():
        # 방법 1: tile image embedding → ST train과 max similarity
        sim_to_train = embs @ train_embs.T          # (T, N_train)
        max_sim      = sim_to_train.max(dim=1).values.cpu().numpy()  # (T,)

        # PredEx로 tile 예측값도 계산 (비교용)
        sim_w    = torch.clamp(sim_to_train, min=0)
        weights  = sim_w / (sim_w.sum(dim=1, keepdim=True) + 1e-8)
        tile_pred = (weights @ train_expr).cpu().numpy()  # (T, 2000)

    tile_pred_common = tile_pred[:, common_idx]  # (T, 1968)

    # tile pred vs bulk cosine sim (이전 방법, 비교용)
    tile_norm = F.normalize(
        torch.tensor(tile_pred_common, dtype=torch.float32, device=device), dim=-1)
    bulk_norm = F.normalize(
        torch.tensor(bulk, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
    pred_bulk_sim = (tile_norm @ bulk_norm.T).squeeze().cpu().numpy()  # (T,)

    # 격자 좌표 변환
    y_px, x_px = coords[:, 0], coords[:, 1]
    x_vals     = np.unique(x_px)
    tile_step  = int(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 512
    x_grid     = ((x_px - x_px.min()) / tile_step).astype(int)
    y_grid     = ((y_px - y_px.min()) / tile_step).astype(int)
    grid_h     = y_grid.max() + 1
    grid_w     = x_grid.max() + 1

    # YES/NO (상위 10% / 하위 10%)
    p10_new = np.percentile(max_sim, 10)
    p90_new = np.percentile(max_sim, 90)
    labels_new = np.where(max_sim >= p90_new, 2,
                 np.where(max_sim <= p10_new, 0, 1))

    p10_old = np.percentile(pred_bulk_sim, 10)
    p90_old = np.percentile(pred_bulk_sim, 90)
    labels_old = np.where(pred_bulk_sim >= p90_old, 2,
                 np.where(pred_bulk_sim <= p10_old, 0, 1))

    # 격자 이미지
    def make_grids(vals, labels):
        sim_grid   = np.full((grid_h, grid_w), np.nan)
        label_grid = np.full((grid_h, grid_w), -1)
        for i in range(len(vals)):
            sim_grid[y_grid[i], x_grid[i]]   = vals[i]
            label_grid[y_grid[i], x_grid[i]] = labels[i]
        return sim_grid, label_grid

    sg_new, lg_new = make_grids(max_sim,      labels_new)
    sg_old, lg_old = make_grids(pred_bulk_sim, labels_old)

    # 시각화 (2행 2열)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'{sid}  |  Tiles: {len(max_sim)}, Grid: {grid_h}×{grid_w}', fontsize=13)

    cmap_label = plt.cm.colors.ListedColormap(['red', 'lightgray', 'green'])
    patches = [
        mpatches.Patch(color='red',       label='NO  (bottom 10%)'),
        mpatches.Patch(color='lightgray', label='Neutral'),
        mpatches.Patch(color='green',     label='YES (top 10%)'),
    ]

    # Row 1: 새 방법 (ST train max sim)
    im0 = axes[0,0].imshow(sg_new, cmap='RdYlGn', aspect='auto',
                            vmin=np.nanpercentile(sg_new, 2),
                            vmax=np.nanpercentile(sg_new, 98))
    plt.colorbar(im0, ax=axes[0,0])
    axes[0,0].set_title(f'[NEW] Tile-ST Max Similarity\nstd={max_sim.std():.4f}  range=[{max_sim.min():.3f}, {max_sim.max():.3f}]')

    axes[0,1].imshow(np.where(lg_new >= 0, lg_new, np.nan),
                     cmap=cmap_label, vmin=0, vmax=2, aspect='auto')
    axes[0,1].legend(handles=patches, loc='upper right', fontsize=9)
    axes[0,1].set_title(f'[NEW] YES/NO  (p10={p10_new:.4f}, p90={p90_new:.4f})')

    # Row 2: 이전 방법 (pred vs bulk sim)
    im2 = axes[1,0].imshow(sg_old, cmap='RdYlGn', aspect='auto',
                            vmin=np.nanpercentile(sg_old, 2),
                            vmax=np.nanpercentile(sg_old, 98))
    plt.colorbar(im2, ax=axes[1,0])
    axes[1,0].set_title(f'[OLD] Tile-Pred vs Bulk Similarity\nstd={pred_bulk_sim.std():.4f}  range=[{pred_bulk_sim.min():.3f}, {pred_bulk_sim.max():.3f}]')

    axes[1,1].imshow(np.where(lg_old >= 0, lg_old, np.nan),
                     cmap=cmap_label, vmin=0, vmax=2, aspect='auto')
    axes[1,1].legend(handles=patches, loc='upper right', fontsize=9)
    axes[1,1].set_title(f'[OLD] YES/NO  (p10={p10_old:.4f}, p90={p90_old:.4f})')

    for ax in axes.flat:
        ax.set_xlabel('X grid')
        ax.set_ylabel('Y grid')

    plt.tight_layout()
    save_path = f'{OUT_DIR}/{sid}_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')
    print(f'  NEW std={max_sim.std():.4f}  OLD std={pred_bulk_sim.std():.4f}')
    print(f'  YES: {(labels_new==2).sum()} tiles')

print('Done!')
