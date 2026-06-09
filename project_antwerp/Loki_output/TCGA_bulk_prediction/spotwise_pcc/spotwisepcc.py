nohup python3 << 'EOF' > /project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/logs/spotwise_pcc.log 2>&1 &
echo $!
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/spotwise_pcc'
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

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/fold_01/{row["slide_id"]}.npy')]
print(f'Slides: {len(matched)}')

FOLDS = [f'fold_{i:02d}' for i in range(1, 11)]
K = 3

# fold별 결과 저장
# {fold: {
#   'all_scores':    (N_total_tiles,) 전체 tile spot-wise PCC
#   'top3_scores':   (331,) 각 슬라이드 top-3 mean score
#   'bot3_scores':   (331,) 각 슬라이드 bot-3 mean score
#   'rand3_scores':  (331,) 각 슬라이드 random-3 mean score
#   'slide_n_tiles': (331,) 각 슬라이드 tile 수
#   'top3_pct':      (331,) top-3의 percentile 위치 (0~100)
# }}
fold_results = {}

for fold in FOLDS:
    print(f'\n[{fold}]')
    train_embs = F.normalize(torch.tensor(
        np.load(f'{FT_EMB}/{fold}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
    train_expr = torch.tensor(
        np.load(f'{FT_EMB}/{fold}/train_exprs.npy'), dtype=torch.float32, device=device)

    all_scores_fold   = []  # 전체 tile scores 모음
    top3_scores_fold  = []
    bot3_scores_fold  = []
    rand3_scores_fold = []
    slide_n_tiles     = []
    top3_pct_fold     = []

    np.random.seed(42)

    for sid, bulk in matched:
        embs = F.normalize(torch.tensor(
            np.load(f'{TCGA_EMB}/{fold}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
        with torch.no_grad():
            sim        = torch.clamp(embs @ train_embs.T, min=0)
            weights    = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
            tile_preds = (weights @ train_expr).cpu().numpy()
        tp = tile_preds[:, common_idx]
        T  = len(tp)

        # spot-wise PCC: 각 tile vs bulk
        bulk_c = bulk - bulk.mean()
        tile_c = tp - tp.mean(axis=1, keepdims=True)
        num    = (tile_c * bulk_c).sum(axis=1)
        denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
        scores = np.where(denom > 1e-8, num/denom, np.nan)

        valid_mask  = ~np.isnan(scores)
        valid_scores = scores[valid_mask]
        valid_idx    = np.where(valid_mask)[0]

        # top-3, bot-3, random-3
        sorted_v    = valid_idx[np.argsort(valid_scores)[::-1]]
        top3_idx    = sorted_v[:K]
        bot3_idx    = sorted_v[-K:]
        rand3_idx   = valid_idx[np.random.choice(len(valid_idx), K, replace=False)]

        # top-3의 percentile 위치
        top3_pct = [(np.sum(valid_scores <= scores[t]) / len(valid_scores)) * 100
                    for t in top3_idx]

        all_scores_fold.extend(valid_scores.tolist())
        top3_scores_fold.append(scores[top3_idx].mean())
        bot3_scores_fold.append(scores[bot3_idx].mean())
        rand3_scores_fold.append(scores[rand3_idx].mean())
        slide_n_tiles.append(T)
        top3_pct_fold.append(np.mean(top3_pct))
        del embs

    del train_embs, train_expr
    torch.cuda.empty_cache()

    fold_results[fold] = {
        'all_scores':    np.array(all_scores_fold),
        'top3_scores':   np.array(top3_scores_fold),
        'bot3_scores':   np.array(bot3_scores_fold),
        'rand3_scores':  np.array(rand3_scores_fold),
        'slide_n_tiles': np.array(slide_n_tiles),
        'top3_pct':      np.array(top3_pct_fold),
    }

    r  = fold_results[fold]
    print(f'  All tiles:   mean={r["all_scores"].mean():.4f}  std={r["all_scores"].std():.4f}')
    print(f'  Top-3:       mean={r["top3_scores"].mean():.4f}  std={r["top3_scores"].std():.4f}')
    print(f'  Bot-3:       mean={r["bot3_scores"].mean():.4f}  std={r["bot3_scores"].std():.4f}')
    print(f'  Random-3:    mean={r["rand3_scores"].mean():.4f}  std={r["rand3_scores"].std():.4f}')
    print(f'  Top-3 pct:   mean={r["top3_pct"].mean():.1f}%')

# ── Figure 1: Fold별 Spot-wise PCC 분포 violin ────────────
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
fig.suptitle('Spot-wise PCC Distribution per Fold\n(All tiles, Top-3, Bot-3, Random-3)', fontsize=13)

colors = {'all': '#95a5a6', 'top3': '#e74c3c', 'bot3': '#3498db', 'rand3': '#2ecc71'}

for i, fold in enumerate(FOLDS):
    ax  = axes[i//5, i%5]
    r   = fold_results[fold]

    # violin: all tiles 분포
    vp = ax.violinplot([r['all_scores']], positions=[0], showmedians=True)
    for pc in vp['bodies']:
        pc.set_facecolor(colors['all']); pc.set_alpha(0.6)

    # scatter: top-3, bot-3, rand-3 mean
    ax.scatter([0], [r['top3_scores'].mean()],  color=colors['top3'],  s=100, zorder=5, marker='^', label='top3 mean')
    ax.scatter([0], [r['bot3_scores'].mean()],  color=colors['bot3'],  s=100, zorder=5, marker='v', label='bot3 mean')
    ax.scatter([0], [r['rand3_scores'].mean()], color=colors['rand3'], s=80,  zorder=5, marker='s', label='rand3 mean')

    # 통계 텍스트
    all_m = r['all_scores'].mean()
    top_m = r['top3_scores'].mean()
    bot_m = r['bot3_scores'].mean()
    ax.set_title(f'{fold}\nall={all_m:.4f}  top3={top_m:.4f}\nbot3={bot_m:.4f}  pct={r["top3_pct"].mean():.1f}%',
                 fontsize=8)
    ax.set_xticks([])
    ax.set_ylabel('Spot-wise PCC', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/spotwise_violin_per_fold.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: spotwise_violin_per_fold.png')

# ── Figure 2: Top-3 vs Bot-3 vs Random-3 비교 (fold별) ───
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Spot-wise PCC: Top-3 vs Bot-3 vs Random-3 (All Folds)', fontsize=13)

fold_labels = [f'f{i:02d}' for i in range(1, 11)]
top3_means  = [fold_results[f]['top3_scores'].mean()  for f in FOLDS]
bot3_means  = [fold_results[f]['bot3_scores'].mean()  for f in FOLDS]
rand3_means = [fold_results[f]['rand3_scores'].mean() for f in FOLDS]
all_means   = [fold_results[f]['all_scores'].mean()   for f in FOLDS]

x = np.arange(10)
w = 0.2
axes[0].bar(x - 1.5*w, all_means,   w, label=f'All tiles  (mean={np.mean(all_means):.4f})',   color=colors['all'],   alpha=0.8)
axes[0].bar(x - 0.5*w, rand3_means, w, label=f'Random-3  (mean={np.mean(rand3_means):.4f})', color=colors['rand3'], alpha=0.8)
axes[0].bar(x + 0.5*w, bot3_means,  w, label=f'Bot-3     (mean={np.mean(bot3_means):.4f})',  color=colors['bot3'],  alpha=0.8)
axes[0].bar(x + 1.5*w, top3_means,  w, label=f'Top-3     (mean={np.mean(top3_means):.4f})',  color=colors['top3'],  alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(fold_labels, fontsize=9)
axes[0].set_ylabel('Mean Spot-wise PCC')
axes[0].set_title('Mean Spot-wise PCC per Fold')
axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.3)

# 차이 비교
top3_vs_all  = [t - a for t, a in zip(top3_means, all_means)]
top3_vs_rand = [t - r for t, r in zip(top3_means, rand3_means)]
top3_vs_bot  = [t - b for t, b in zip(top3_means, bot3_means)]
axes[1].plot(range(1,11), top3_vs_all,  '-o', color='gray',           label=f'Top3 - All   (mean={np.mean(top3_vs_all):.4f})',  linewidth=2)
axes[1].plot(range(1,11), top3_vs_rand, '-s', color=colors['rand3'],  label=f'Top3 - Rand  (mean={np.mean(top3_vs_rand):.4f})', linewidth=2)
axes[1].plot(range(1,11), top3_vs_bot,  '-^', color=colors['bot3'],   label=f'Top3 - Bot3  (mean={np.mean(top3_vs_bot):.4f})',  linewidth=2)
axes[1].axhline(0, color='black', linestyle='--', alpha=0.4)
axes[1].set_xticks(range(1,11)); axes[1].set_xticklabels(fold_labels, fontsize=9)
axes[1].set_ylabel('PCC difference'); axes[1].set_title('Top-3 vs Others: Difference')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/spotwise_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: spotwise_comparison.png')

# ── Figure 3: Spot-wise PCC 분포 (전체 합산) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Spot-wise PCC: Overall Distribution (All Folds Combined)', fontsize=13)

all_combined   = np.concatenate([fold_results[f]['all_scores']   for f in FOLDS])
top3_combined  = np.concatenate([fold_results[f]['top3_scores']  for f in FOLDS])
bot3_combined  = np.concatenate([fold_results[f]['bot3_scores']  for f in FOLDS])
rand3_combined = np.concatenate([fold_results[f]['rand3_scores'] for f in FOLDS])
pct_combined   = np.concatenate([fold_results[f]['top3_pct']     for f in FOLDS])

axes[0].hist(all_combined,  bins=80, density=True, alpha=0.5, color=colors['all'],   label=f'All (mean={all_combined.mean():.4f}, std={all_combined.std():.4f})')
axes[0].hist(top3_combined, bins=50, density=True, alpha=0.7, color=colors['top3'],  label=f'Top-3 (mean={top3_combined.mean():.4f})')
axes[0].hist(bot3_combined, bins=50, density=True, alpha=0.7, color=colors['bot3'],  label=f'Bot-3 (mean={bot3_combined.mean():.4f})')
axes[0].hist(rand3_combined,bins=50, density=True, alpha=0.7, color=colors['rand3'], label=f'Random-3 (mean={rand3_combined.mean():.4f})')
axes[0].set_xlabel('Spot-wise PCC'); axes[0].set_ylabel('Density')
axes[0].set_title('Distribution: All tiles vs Top/Bot/Random-3')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Top-3 percentile 분포
axes[1].hist(pct_combined, bins=40, color='#e74c3c', alpha=0.8)
axes[1].axvline(pct_combined.mean(), color='black', linestyle='--',
                label=f'mean={pct_combined.mean():.1f}%')
axes[1].axvline(95, color='red', linestyle=':', alpha=0.7, label='95th pct')
axes[1].set_xlabel('Percentile position of Top-3 tiles'); axes[1].set_ylabel('Count')
axes[1].set_title(f'Top-3 Tile Percentile Position\n(mean={pct_combined.mean():.1f}%)')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/spotwise_overall_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: spotwise_overall_dist.png')

# ── 요약 출력 ─────────────────────────────────────────────
print('\n' + '='*65)
print('Spot-wise PCC 요약')
print('='*65)
print(f'{"fold":>8} | {"all mean":>9} | {"all std":>8} | {"top3 mean":>10} | {"bot3 mean":>10} | {"top3 pct":>9}')
print('-'*65)
for fold in FOLDS:
    r = fold_results[fold]
    print(f'{fold:>8} | {r["all_scores"].mean():>9.4f} | {r["all_scores"].std():>8.4f} | '
          f'{r["top3_scores"].mean():>10.4f} | {r["bot3_scores"].mean():>10.4f} | '
          f'{r["top3_pct"].mean():>8.1f}%')

print(f'\nOverall:')
print(f'  All tiles:  mean={all_combined.mean():.4f}  std={all_combined.std():.4f}')
print(f'  Top-3:      mean={top3_combined.mean():.4f}')
print(f'  Bot-3:      mean={bot3_combined.mean():.4f}')
print(f'  Random-3:   mean={rand3_combined.mean():.4f}')
print(f'  Top-3 pct:  mean={pct_combined.mean():.1f}%')
print(f'  Top3 - All:  {top3_combined.mean()-all_combined.mean():+.4f}')
print(f'  Top3 - Rand: {top3_combined.mean()-rand3_combined.mean():+.4f}')
print(f'  Top3 - Bot3: {top3_combined.mean()-bot3_combined.mean():+.4f}')
print('\nDone!')
EOF