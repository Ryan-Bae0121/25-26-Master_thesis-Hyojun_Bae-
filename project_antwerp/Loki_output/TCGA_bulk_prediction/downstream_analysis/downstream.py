python3 << 'EOF'
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/downstream_analysis'
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

def coords_to_grid(coords):
    y_px, x_px = coords[:,0], coords[:,1]
    x_vals = np.unique(x_px)
    step   = int(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 512
    return ((x_px-x_px.min())/step).astype(int), ((y_px-y_px.min())/step).astype(int)

# HNSCC 관련 gene 리스트
hnscc_genes = {
    'Squamous': ['KRT5','KRT14','KRT6A','KRT6B','KRT16','TP63'],
    'Tumor':    ['EGFR','CCND1','MYC','CDK6','CDKN2A'],
    'EMT':      ['CDH1','VIM','FN1','ZEB1','SNAI1'],
    'Immune':   ['CD274','CD8A','CD68','FOXP3','PDCD1'],
    'ECM':      ['COL1A1','COL3A1','FN1','MMP1','MMP9'],
    'HNSCC_sig':['S100A8','S100A9','CXCL1','CXCL8','IL6'],
}

# 슬라이드 5개 샘플링
sample_sids = [matched[i][0] for i in [0, 10, 30, 50, 100]]
K = 3

print('Analyzing top-3 tiles for sample slides...')

all_analysis = []

for sid_idx, (sid, bulk) in enumerate(matched):
    if sid not in sample_sids:
        continue

    embs   = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    coords = np.load(f'{TCGA_EMB}/{sid}_coords.npy')
    x_grid, y_grid = coords_to_grid(coords)
    T = len(embs)

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
    sorted_by_score = v_idx[np.argsort(scores[valid])[::-1]]
    top3_idx = sorted_by_score[:K]
    bot3_idx = sorted_by_score[-K:]

    # ── Figure: 슬라이드별 분석 ──────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Downstream Analysis: {sid}\nTiles={T}, K=3', fontsize=13)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Score 분포 + top-3/bot-3 위치
    ax1 = fig.add_subplot(gs[0, 0])
    valid_scores = scores[valid]
    ax1.hist(valid_scores, bins=60, color='steelblue', alpha=0.7, density=True)
    for s in scores[top3_idx]:
        ax1.axvline(s, color='red', linewidth=2, alpha=0.8)
    for s in scores[bot3_idx]:
        ax1.axvline(s, color='blue', linewidth=1.5, linestyle='--', alpha=0.6)
    ax1.set_title(f'Score Distribution\nred=top3, blue=bot3', fontsize=9)
    ax1.set_xlabel('Tile PCC score')
    ax1.set_ylabel('Density')

    # top-3 score 통계
    top3_scores = scores[top3_idx]
    all_pct = [(scores[t] - valid_scores.min()) / (valid_scores.max() - valid_scores.min()) * 100
               for t in top3_idx]
    print(f'\n{sid}:')
    print(f'  T={T}, score range=[{valid_scores.min():.4f}, {valid_scores.max():.4f}]')
    print(f'  Top-3 scores: {top3_scores}')
    print(f'  Top-3 percentile: {[f"{p:.1f}%" for p in all_pct]}')

    # 2. Spatial map (score heatmap + top3/bot3 표시)
    ax2 = fig.add_subplot(gs[0, 1])
    grid_h = y_grid.max() + 1
    grid_w = x_grid.max() + 1
    score_grid = np.full((grid_h, grid_w), np.nan)
    for i in range(T):
        if scores[i] > -999:
            score_grid[y_grid[i], x_grid[i]] = scores[i]

    im = ax2.imshow(score_grid, cmap='RdYlGn', aspect='auto',
                    vmin=np.nanpercentile(score_grid, 5),
                    vmax=np.nanpercentile(score_grid, 95))
    plt.colorbar(im, ax=ax2)

    # top-3, bot-3 tile 표시
    for i, t in enumerate(top3_idx):
        ax2.scatter(x_grid[t], y_grid[t], c='red', s=200, marker='*',
                   zorder=5, label='Top-3' if i==0 else '')
    for i, t in enumerate(bot3_idx):
        ax2.scatter(x_grid[t], y_grid[t], c='blue', s=100, marker='x',
                   zorder=5, label='Bot-3' if i==0 else '')
    ax2.set_title('Score Heatmap\n★=Top3, x=Bot3', fontsize=9)
    ax2.legend(fontsize=7)

    # 3. Top-3 tile의 spatial 위치 (슬라이드 중심 대비)
    ax3 = fig.add_subplot(gs[0, 2])
    cx = x_grid.mean(); cy = y_grid.mean()
    ax3.scatter(x_grid, y_grid, c='lightgray', s=5, alpha=0.5)
    for i, t in enumerate(top3_idx):
        ax3.scatter(x_grid[t], y_grid[t], c='red', s=200, marker='*', zorder=5)
        ax3.annotate(f'T{i+1}\n({x_grid[t]},{y_grid[t]})',
                    (x_grid[t], y_grid[t]), fontsize=7, color='red',
                    xytext=(5, 5), textcoords='offset points')
    ax3.scatter(cx, cy, c='black', s=100, marker='+', label='center')
    ax3.set_title('Top-3 Spatial Position', fontsize=9)
    ax3.legend(fontsize=7)

    # 4. Top-3 vs Bottom-3 vs Bulk gene expression 비교
    ax4 = fig.add_subplot(gs[1, :])
    top3_pred  = tp[top3_idx].mean(axis=0)   # (1968,)
    bot3_pred  = tp[bot3_idx].mean(axis=0)
    all_pred   = tp.mean(axis=0)

    # 상위 30개 gene만 표시 (bulk 기준 발현량 높은 것)
    top30_idx  = np.argsort(bulk)[::-1][:30]
    x_pos      = np.arange(30)
    width      = 0.25
    ax4.bar(x_pos - width, bulk[top30_idx], width, label='Bulk (true)', color='black', alpha=0.7)
    ax4.bar(x_pos,         top3_pred[top30_idx], width, label='Top-3 pred', color='red', alpha=0.7)
    ax4.bar(x_pos + width, bot3_pred[top30_idx], width, label='Bot-3 pred', color='blue', alpha=0.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([common_genes[i] for i in top30_idx], rotation=45, ha='right', fontsize=7)
    ax4.set_title('Top-30 Expressed Genes: Bulk vs Top-3 pred vs Bot-3 pred', fontsize=10)
    ax4.set_ylabel('Expression value')
    ax4.legend(fontsize=8)

    # 5. Gene group별 평균 발현량 비교
    ax5 = fig.add_subplot(gs[2, :2])
    group_names = []
    top3_means, bot3_means, bulk_means = [], [], []

    for grp_name, grp_genes in hnscc_genes.items():
        idx_g = [i for i, g in enumerate(common_genes) if g in grp_genes]
        if len(idx_g) == 0:
            continue
        group_names.append(f'{grp_name}\n(n={len(idx_g)})')
        top3_means.append(top3_pred[idx_g].mean())
        bot3_means.append(bot3_pred[idx_g].mean())
        bulk_means.append(bulk[idx_g].mean())

    x_g   = np.arange(len(group_names))
    width = 0.28
    ax5.bar(x_g - width, bulk_means,  width, label='Bulk',      color='black', alpha=0.7)
    ax5.bar(x_g,         top3_means,  width, label='Top-3 pred', color='red',   alpha=0.7)
    ax5.bar(x_g + width, bot3_means,  width, label='Bot-3 pred', color='blue',  alpha=0.5)
    ax5.set_xticks(x_g)
    ax5.set_xticklabels(group_names, fontsize=8)
    ax5.set_title('Compare mean of gene expression of each Gene Group', fontsize=10)
    ax5.set_ylabel('Mean expression')
    ax5.legend(fontsize=8)

    # 6. Top-3 tile PCC with bulk vs all tiles
    ax6 = fig.add_subplot(gs[2, 2])
    r_top3, _ = pearsonr(tp[top3_idx].sum(axis=0), bulk)
    r_all, _  = pearsonr(tp.mean(axis=0), bulk)
    r_bot3, _ = pearsonr(tp[bot3_idx].sum(axis=0), bulk)
    ax6.bar(['All tiles\n(avg)', 'Bot-3\n(sum)', 'Top-3\n(sum)'],
            [r_all, r_bot3, r_top3],
            color=['gray', 'blue', 'red'], alpha=0.8)
    ax6.set_title('Compare Slide-wise PCC', fontsize=10)
    ax6.set_ylabel('Slide-wise PCC')
    ax6.axhline(0, color='black', linestyle='--', alpha=0.3)
    for i, v in enumerate([r_all, r_bot3, r_top3]):
        ax6.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

    plt.savefig(f'{OUT_DIR}/{sid}_downstream.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {sid}_downstream.png')

    all_analysis.append({
        'sid': sid, 'T': T,
        'top3_scores': scores[top3_idx].tolist(),
        'top3_x': x_grid[top3_idx].tolist(),
        'top3_y': y_grid[top3_idx].tolist(),
        'top3_pcc': float(pearsonr(tp[top3_idx].sum(axis=0), bulk)[0]),
        'all_pcc':  float(pearsonr(tp.mean(axis=0), bulk)[0]),
    })

# ── 전체 331 슬라이드: top-3 tile의 위치 통계 ────────────
print('\nComputing top-3 position statistics for all slides...')
rel_positions = []  # top-3 tile의 슬라이드 중심 대비 상대 위치

for sid, bulk in matched:
    embs   = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    coords = np.load(f'{TCGA_EMB}/{sid}_coords.npy')
    x_grid, y_grid = coords_to_grid(coords)
    T = len(embs)

    with torch.no_grad():
        sim        = torch.clamp(embs @ train_embs.T, min=0)
        weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tile_preds = (weights @ train_expr).cpu().numpy()

    tp = tile_preds[:, common_idx]
    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
    scores = np.where(denom > 1e-8, num/denom, -999)

    valid   = scores > -999
    v_idx   = np.where(valid)[0]
    top3_idx = v_idx[np.argsort(scores[valid])[::-1][:K]]

    # 중심 대비 상대 위치 (0=중앙, 1=가장자리)
    cx = (x_grid.max() + x_grid.min()) / 2
    cy = (y_grid.max() + y_grid.min()) / 2
    max_dist = np.sqrt(((x_grid-cx)**2 + (y_grid-cy)**2).max())

    for t in top3_idx:
        dist = np.sqrt((x_grid[t]-cx)**2 + (y_grid[t]-cy)**2)
        rel_positions.append(dist / (max_dist + 1e-8))
    del embs

# 위치 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Top-3 Tile Position Analysis (331 slides)', fontsize=13)

axes[0].hist(rel_positions, bins=50, color='#e74c3c', alpha=0.8, density=True)
axes[0].axvline(np.mean(rel_positions), color='black', linestyle='--',
                label=f'mean={np.mean(rel_positions):.3f}')
axes[0].set_xlabel('Relative distance from center\n(0=center, 1=edge)', fontsize=10)
axes[0].set_ylabel('Density')
axes[0].set_title('Top-3 Tile Distance from Slide Center')
axes[0].legend()

# 중앙 vs 가장자리 비율
center_ratio = np.mean(np.array(rel_positions) < 0.5)
axes[1].bar(['Center\n(dist<0.5)', 'Edge\n(dist>=0.5)'],
            [center_ratio, 1-center_ratio],
            color=['#3498db', '#e74c3c'], alpha=0.8)
axes[1].set_title(f'Top-3 Tile Location\nCenter: {center_ratio:.1%}, Edge: {1-center_ratio:.1%}')
axes[1].set_ylabel('Proportion')
for i, v in enumerate([center_ratio, 1-center_ratio]):
    axes[1].text(i, v+0.01, f'{v:.1%}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/top3_position_stats.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: top3_position_stats.png')
print('Done!')
EOF