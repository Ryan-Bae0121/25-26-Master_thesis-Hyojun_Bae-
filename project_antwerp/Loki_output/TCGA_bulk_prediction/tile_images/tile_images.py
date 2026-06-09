python3 << 'EOF'
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from PIL import Image
import os

device = 'cuda'
GENE_LIST  = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE   = '/project_antwerp/hbae/ref_file.csv'
FT_EMB     = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
PATCH_DIR  = '/project_antwerp/TCGA-HNSC/TCGA_patch'
OUT_DIR    = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tile_images'
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

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/{row["slide_id"]}.npy')]

# 슬라이드 5개 샘플
sample_sids = [matched[i][0] for i in [0, 10, 30, 50, 100]]
K = 3

for sid, bulk in matched:
    if sid not in sample_sids:
        continue

    # HDF5 파일 경로
    hdf5_path = f'{PATCH_DIR}/{sid}/{sid}.hdf5'
    if not os.path.exists(hdf5_path):
        print(f'HDF5 not found: {hdf5_path}')
        continue

    embs   = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    coords = np.load(f'{TCGA_EMB}/{sid}_coords.npy')
    T      = len(embs)

    with torch.no_grad():
        sim        = torch.clamp(embs @ train_embs.T, min=0)
        weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()

    tp = tile_preds[:, common_idx]

    # tile-wise PCC score
    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
    scores = np.where(denom > 1e-8, num/denom, -999)

    valid   = scores > -999
    v_idx   = np.where(valid)[0]
    sorted_ = v_idx[np.argsort(scores[valid])[::-1]]
    top3_idx = sorted_[:K]
    bot3_idx = sorted_[-K:]
    mid3_idx = sorted_[len(sorted_)//2 - 1: len(sorted_)//2 + 2]

    # HDF5에서 tile 이미지 로드
    # coords (y_px, x_px) → HDF5 key = "y_x"
    def load_tile(hdf5_file, y_px, x_px):
        key = f'{y_px}_{x_px}'
        if key in hdf5_file:
            return Image.fromarray(hdf5_file[key][:])
        return None

    print(f'\n{sid}: loading tile images...')

    with h5py.File(hdf5_path, 'r') as f:
        # 각 그룹별 tile 이미지 로드
        groups = {
            'Top-3 (highest PCC)': top3_idx,
            'Mid-3 (median PCC)':  mid3_idx,
            'Bot-3 (lowest PCC)':  bot3_idx,
        }

        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f'H&E Tile Images: {sid}\n'
                     f'Top-3 PCC={scores[top3_idx].mean():.4f} | '
                     f'Mid-3 PCC={scores[mid3_idx].mean():.4f} | '
                     f'Bot-3 PCC={scores[bot3_idx].mean():.4f}',
                     fontsize=12)

        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

        for col_i, (group_name, idx_list) in enumerate(groups.items()):
            color = ['red', 'green', 'blue'][col_i]

            for row_i, t in enumerate(idx_list):
                y_px, x_px = int(coords[t, 0]), int(coords[t, 1])
                img = load_tile(f, y_px, x_px)

                ax = fig.add_subplot(gs[row_i, col_i])
                if img is not None:
                    ax.imshow(img)
                    ax.set_title(
                        f'{group_name}\nTile {row_i+1}\n'
                        f'score={scores[t]:.4f}\n'
                        f'pos=({x_px//512},{y_px//512})',
                        fontsize=8, color=color
                    )
                else:
                    ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
                    ax.set_title(f'{group_name} T{row_i+1}', fontsize=8)
                ax.axis('off')

        # 마지막 행: gene expression bar chart (top-3 vs bot-3)
        ax_bar = fig.add_subplot(gs[3, :])
        top3_pred = tp[top3_idx].mean(axis=0)
        bot3_pred = tp[bot3_idx].mean(axis=0)

        # bulk 기준 top-15 gene
        top15 = np.argsort(bulk)[::-1][:15]
        x_pos = np.arange(15)
        w = 0.3
        ax_bar.bar(x_pos - w, bulk[top15],      w, label='Bulk (true)', color='black', alpha=0.8)
        ax_bar.bar(x_pos,     top3_pred[top15], w, label='Top-3 pred',  color='red',   alpha=0.7)
        ax_bar.bar(x_pos + w, bot3_pred[top15], w, label='Bot-3 pred',  color='blue',  alpha=0.5)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([common_genes[i] for i in top15],
                               rotation=45, ha='right', fontsize=8)
        ax_bar.set_title('Top-15 Expressed Genes: Bulk vs Top-3 vs Bot-3', fontsize=10)
        ax_bar.set_ylabel('Expression value')
        ax_bar.legend(fontsize=9)

        save_path = f'{OUT_DIR}/{sid}_tile_images.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {save_path}')
    del embs

print('\nDone!')
EOF