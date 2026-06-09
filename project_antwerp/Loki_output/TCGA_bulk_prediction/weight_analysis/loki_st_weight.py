python3 << 'EOF'
import numpy as np, torch, torch.nn.functional as F
import pandas as pd, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, entropy

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/weight_analysis'
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
train_expr_np = train_expr.cpu().numpy()[:, common_idx]  # (N_ST, G)
N_ST = len(train_expr_np)

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/{row["slide_id"]}.npy')]
print(f'Slides: {len(matched)}, ST spots: {N_ST}')

K = 3

# 결과 저장
weight_entropies_top3  = []  # 가중치 분포의 entropy (낮을수록 집중)
weight_entropies_bot3  = []
weight_entropies_rand  = []
top_spot_expr_top3     = []  # top-weight ST spot의 gene expression
top_spot_expr_bot3     = []
top_k_weight_top3      = []  # 상위 K개 spot이 전체 가중치에서 차지하는 비율
top_k_weight_bot3      = []

np.random.seed(42)
sample_sids = [matched[i][0] for i in range(min(20, len(matched)))]

print('Analyzing weight distributions...')
for sid, bulk in matched:
    if sid not in sample_sids:
        # 전체에 대해선 entropy만
        pass

    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim_raw = embs @ train_embs.T          # (T, N_ST) cosine sim
        sim_pos = torch.clamp(sim_raw, min=0)
        w       = sim_pos / (sim_pos.sum(dim=1, keepdim=True) + 1e-8)  # (T, N_ST)
        tp      = (w @ train_expr).cpu().numpy()[:, common_idx]         # (T, G)
        w_np    = w.cpu().numpy()               # (T, N_ST)

    # tile-wise PCC score
    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
    scores = np.where(denom > 1e-8, num/denom, -999)

    valid  = scores > -999
    v_idx  = np.where(valid)[0]
    sorted_ = v_idx[np.argsort(scores[valid])[::-1]]
    top3_idx = sorted_[:K]
    bot3_idx = sorted_[-K:]
    rand_idx = v_idx[np.random.choice(len(v_idx), K, replace=False)]

    # 각 그룹에 대해 가중치 분포 분석
    for group_idx, group_list, ent_list, topk_list, expr_list in [
        (top3_idx, None, weight_entropies_top3, top_k_weight_top3, top_spot_expr_top3),
        (bot3_idx, None, weight_entropies_bot3, top_k_weight_bot3, top_spot_expr_bot3),
        (rand_idx, None, weight_entropies_rand, None, None),
    ]:
        for t in group_idx:
            wt = w_np[t]  # (N_ST,) 이 tile의 ST spot별 가중치

            # Entropy: 낮을수록 가중치가 특정 spot에 집중
            # Uniform entropy = log(N_ST)
            wt_nonzero = wt[wt > 1e-10]
            ent = entropy(wt_nonzero) / np.log(len(wt_nonzero))  # normalized
            ent_list.append(ent)

            if topk_list is not None:
                # 상위 100개 spot이 전체 가중치의 몇 %?
                top100_w = np.sort(wt)[::-1][:100].sum()
                topk_list.append(top100_w)

                # 가장 높은 가중치 받은 spot의 gene expression
                top_spot = np.argmax(wt)
                expr_list.append(train_expr_np[top_spot])

    del embs, w_np

torch.cuda.empty_cache()

# ── 결과 출력 ─────────────────────────────────────────────
print('\n=== 가중치 분포 분석 ===')
print(f'Normalized entropy (0=concentrated, 1=uniform):')
print(f'  Top-3 tiles: mean={np.mean(weight_entropies_top3):.4f}  std={np.std(weight_entropies_top3):.4f}')
print(f'  Bot-3 tiles: mean={np.mean(weight_entropies_bot3):.4f}  std={np.std(weight_entropies_bot3):.4f}')
print(f'  Random tiles: mean={np.mean(weight_entropies_rand):.4f}  std={np.std(weight_entropies_rand):.4f}')

print(f'\nTop-100 ST spots weight concentration:')
print(f'  Top-3 tiles: mean={np.mean(top_k_weight_top3):.4f}  (top-100/{N_ST} spots = {100/N_ST*100:.2f}%)')
print(f'  Bot-3 tiles: mean={np.mean(top_k_weight_bot3):.4f}')

# ── 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PredEx Weight Distribution Analysis\n(Which ST spots does each tile reference?)', fontsize=13)

# 1. Entropy 비교
axes[0,0].hist(weight_entropies_bot3,  bins=40, alpha=0.7, color='#3498db',
               density=True, label=f'Bot-3 (mean={np.mean(weight_entropies_bot3):.4f})')
axes[0,0].hist(weight_entropies_rand,  bins=40, alpha=0.7, color='#95a5a6',
               density=True, label=f'Random (mean={np.mean(weight_entropies_rand):.4f})')
axes[0,0].hist(weight_entropies_top3,  bins=40, alpha=0.7, color='#e74c3c',
               density=True, label=f'Top-3 (mean={np.mean(weight_entropies_top3):.4f})')
axes[0,0].set_xlabel('Normalized entropy (0=concentrated, 1=uniform)')
axes[0,0].set_ylabel('Density')
axes[0,0].set_title('Weight Distribution Entropy\nTop-3 vs Bot-3 vs Random')
axes[0,0].legend(fontsize=9)
axes[0,0].grid(alpha=0.3)

# 2. Top-100 weight concentration
axes[0,1].hist(top_k_weight_bot3, bins=40, alpha=0.7, color='#3498db',
               density=True, label=f'Bot-3 (mean={np.mean(top_k_weight_bot3):.3f})')
axes[0,1].hist(top_k_weight_top3, bins=40, alpha=0.7, color='#e74c3c',
               density=True, label=f'Top-3 (mean={np.mean(top_k_weight_top3):.3f})')
axes[0,1].set_xlabel(f'Weight in top-100 ST spots (out of {N_ST})')
axes[0,1].set_ylabel('Density')
axes[0,1].set_title(f'Top-100 Spot Weight Concentration\n(top-100/{N_ST} = {100/N_ST*100:.2f}%)')
axes[0,1].legend(fontsize=9)
axes[0,1].grid(alpha=0.3)

# 3. Top-weight ST spot gene expression 비교
top3_expr_mean = np.array(top_spot_expr_top3).mean(axis=0)   # (G,)
bot3_expr_mean = np.array(top_spot_expr_bot3).mean(axis=0)   # (G,)
st_overall_mean = train_expr_np.mean(axis=0)                  # (G,)

# top-20 gene만 표시
top20_idx = np.argsort(st_overall_mean)[::-1][:20]
x = np.arange(20); w = 0.28
axes[1,0].bar(x - w, st_overall_mean[top20_idx], w,
              label='ST overall mean', color='gray', alpha=0.7)
axes[1,0].bar(x,     top3_expr_mean[top20_idx],  w,
              label='Top-3 referenced spot', color='#e74c3c', alpha=0.7)
axes[1,0].bar(x + w, bot3_expr_mean[top20_idx],  w,
              label='Bot-3 referenced spot', color='#3498db', alpha=0.7)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels([common_genes[i] for i in top20_idx],
                          rotation=45, ha='right', fontsize=7)
axes[1,0].set_title('Top-weight ST Spot Gene Expression\n(Top-3 vs Bot-3 vs ST mean)')
axes[1,0].set_ylabel('Expression value')
axes[1,0].legend(fontsize=8)
axes[1,0].grid(axis='y', alpha=0.3)

# 4. Entropy vs PCC score scatter (top-3만)
axes[1,1].scatter(weight_entropies_top3, weight_entropies_bot3,
                  alpha=0.3, s=5, color='purple')
axes[1,1].set_xlabel('Top-3 weight entropy')
axes[1,1].set_ylabel('Bot-3 weight entropy')
axes[1,1].set_title('Top-3 vs Bot-3 Weight Entropy\nper slide')
axes[1,1].grid(alpha=0.3)
r_ent, _ = pearsonr(weight_entropies_top3[:len(weight_entropies_bot3)],
                     weight_entropies_bot3)
axes[1,1].text(0.05, 0.95, f'r={r_ent:.3f}',
               transform=axes[1,1].transAxes, fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/weight_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: weight_distribution.png')
print('Done!')
EOF