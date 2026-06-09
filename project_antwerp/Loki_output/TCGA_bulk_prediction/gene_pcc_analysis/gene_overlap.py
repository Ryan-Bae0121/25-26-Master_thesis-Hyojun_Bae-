python3 << 'EOF'
import numpy as np, torch, torch.nn.functional as F
import pandas as pd, os
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = 'cuda'
GENE_LIST = '/project_antwerp/hbae/data/0317_hvg_2000_list.txt'
REF_FILE  = '/project_antwerp/hbae/ref_file.csv'
FT_EMB    = '/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03'
TCGA_EMB  = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03'
OUT_DIR   = '/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/gene_pcc_analysis'
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
G = len(common_genes)

train_embs = F.normalize(torch.tensor(
    np.load(f'{FT_EMB}/train_img_embs.npy'), dtype=torch.float32, device=device), dim=-1)
train_expr = torch.tensor(
    np.load(f'{FT_EMB}/train_exprs.npy'), dtype=torch.float32, device=device)

matched = [(row['slide_id'], row[bulk_cols].values.astype(float))
           for _, row in ref_df.iterrows()
           if os.path.exists(f'{TCGA_EMB}/{row["slide_id"]}.npy')]
print(f'Slides: {len(matched)}, Genes: {G}')

# ── 분석 4 먼저: ST train에서 gene 통계 ──────────────────
print('\n[ST train gene statistics]')
train_expr_np = train_expr.cpu().numpy()[:, common_idx]  # (N_spots, G)
# 각 gene의 통계
gene_mean_st   = train_expr_np.mean(axis=0)    # (G,) ST에서 평균 발현량
gene_std_st    = train_expr_np.std(axis=0)     # (G,) ST에서 표준편차
gene_nonzero   = (train_expr_np > 0).sum(axis=0)  # (G,) 발현된 spot 수
gene_nonzero_pct = gene_nonzero / len(train_expr_np) * 100  # 발현 비율
print(f'ST train spots: {len(train_expr_np)}')
print(f'Gene nonzero % range: {gene_nonzero_pct.min():.1f}% ~ {gene_nonzero_pct.max():.1f}%')

# ── Top-3 tile sum으로 pseudo-bulk 생성 ──────────────────
print('\nComputing pseudo-bulk (top-3 sum)...')
K = 3
pseudo_bulks = []
bulk_list    = []

for sid, bulk in matched:
    embs = F.normalize(torch.tensor(
        np.load(f'{TCGA_EMB}/{sid}.npy'), dtype=torch.float32, device=device), dim=-1)
    with torch.no_grad():
        sim   = torch.clamp(embs @ train_embs.T, min=0)
        w     = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        tp    = (w @ train_expr).cpu().numpy()[:, common_idx]
    bulk_c = bulk - bulk.mean()
    tile_c = tp - tp.mean(axis=1, keepdims=True)
    num    = (tile_c * bulk_c).sum(axis=1)
    denom  = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
    scores = np.where(denom > 1e-8, num/denom, -999)
    valid  = scores > -999
    v_idx  = np.where(valid)[0]
    top3   = v_idx[np.argsort(scores[valid])[::-1][:K]]
    pseudo_bulks.append(tp[top3].sum(axis=0))
    bulk_list.append(bulk)
    del embs

torch.cuda.empty_cache()

pred_arr = np.array(pseudo_bulks)   # (331, G)
bulk_arr = np.array(bulk_list)      # (331, G)

# ── 분석 1: Gene-wise PCC 전체 리스트 ────────────────────
print('\nComputing gene-wise PCC...')
gene_pccs   = []
gene_r_vals = []
for j in range(G):
    if pred_arr[:,j].std() > 1e-8 and bulk_arr[:,j].std() > 1e-8:
        r, p = pearsonr(pred_arr[:,j], bulk_arr[:,j])
    else:
        r = np.nan
    gene_pccs.append(r)
gene_pccs = np.array(gene_pccs)

# ── 분석 2: 예측값 vs Ground truth 통계 ─────────────────
pred_mean  = pred_arr.mean(axis=0)   # (G,) gene별 평균 예측값
bulk_mean  = bulk_arr.mean(axis=0)   # (G,) gene별 평균 bulk
pred_std   = pred_arr.std(axis=0)    # (G,) gene별 예측 std
bulk_std   = bulk_arr.std(axis=0)    # (G,) gene별 bulk std
scale_ratio = bulk_mean / (pred_mean + 1e-8)  # bulk/pred 배율
mae         = np.abs(pred_mean - bulk_mean)   # 평균 절대 오차

# ── 분석 3: HNSCC marker gene 분류 ───────────────────────
hnscc_markers = {
    'Squamous_diff': ['KRT1','KRT5','KRT6A','KRT6B','KRT14','KRT16','KRT17',
                      'SPRR1B','SPRR2A','SPRR2G','DSG1','KRTDAP','IVL'],
    'HNSCC_sig':     ['S100A8','S100A9','S100A12','CXCL1','CXCL8','IL6'],
    'ECM':           ['COL1A1','COL3A1','FN1','MMP1','MMP3','MMP9','VCAN'],
    'Immune':        ['CD274','CD8A','CD4','FOXP3','CD68','PDCD1'],
    'EMT':           ['VIM','CDH1','ZEB1','SNAI1','TWIST1'],
    'Stem_cell':     ['ALDH1A1','KRT8','GPC3','CLU'],
    'IFN_response':  ['ISG15','IFITM1','IFITM3','MX1','OAS1'],
}

# ── 전체 gene 결과 DataFrame ─────────────────────────────
gene_category = []
for g in common_genes:
    cat = 'Other'
    for c, genes in hnscc_markers.items():
        if g in genes:
            cat = c
            break
    gene_category.append(cat)

df = pd.DataFrame({
    'gene':            common_genes,
    'gene_wise_pcc':   gene_pccs,
    'pred_mean':       pred_mean,
    'bulk_mean':       bulk_mean,
    'pred_std':        pred_std,
    'bulk_std':        bulk_std,
    'scale_ratio':     scale_ratio,
    'mae':             mae,
    'st_mean':         gene_mean_st,
    'st_std':          gene_std_st,
    'st_nonzero_pct':  gene_nonzero_pct,
    'category':        gene_category,
}).sort_values('gene_wise_pcc', ascending=False)

df.to_csv(f'{OUT_DIR}/gene_pcc_full_list.csv', index=False)
print(f'Saved: gene_pcc_full_list.csv ({len(df)} genes)')

# ── 상위/하위 20개 출력 ───────────────────────────────────
print('\n=== Top-20 Gene-wise PCC ===')
print(f'{"gene":>12} | {"PCC":>6} | {"pred_mean":>10} | {"bulk_mean":>10} | {"scale_ratio":>12} | {"st_nonzero%":>12} | {"category":>15}')
print('-'*85)
for _, row in df.head(20).iterrows():
    print(f'{row.gene:>12} | {row.gene_wise_pcc:>6.3f} | {row.pred_mean:>10.3f} | '
          f'{row.bulk_mean:>10.3f} | {row.scale_ratio:>12.2f} | '
          f'{row.st_nonzero_pct:>11.1f}% | {row.category:>15}')

print('\n=== Bottom-20 Gene-wise PCC ===')
print(f'{"gene":>12} | {"PCC":>6} | {"pred_mean":>10} | {"bulk_mean":>10} | {"scale_ratio":>12} | {"st_nonzero%":>12} | {"category":>15}')
print('-'*85)
for _, row in df.tail(20).iterrows():
    print(f'{row.gene:>12} | {row.gene_wise_pcc:>6.3f} | {row.pred_mean:>10.3f} | '
          f'{row.bulk_mean:>10.3f} | {row.scale_ratio:>12.2f} | '
          f'{row.st_nonzero_pct:>11.1f}% | {row.category:>15}')

# ── 분석 4: ST nonzero% vs gene-wise PCC 상관관계 ────────
valid_mask = ~np.isnan(gene_pccs)
r_nonzero, _ = pearsonr(gene_nonzero_pct[valid_mask], gene_pccs[valid_mask])
r_stmean, _  = pearsonr(gene_mean_st[valid_mask],     gene_pccs[valid_mask])
r_ststd, _   = pearsonr(gene_std_st[valid_mask],      gene_pccs[valid_mask])
print(f'\n=== ST train 통계 vs Gene-wise PCC 상관관계 ===')
print(f'r(ST nonzero%,  gene PCC) = {r_nonzero:.4f}')
print(f'r(ST mean expr, gene PCC) = {r_stmean:.4f}')
print(f'r(ST std expr,  gene PCC) = {r_ststd:.4f}')

# ── 카테고리별 PCC 분포 ───────────────────────────────────
print(f'\n=== HNSCC Category별 Gene-wise PCC ===')
for cat in ['Squamous_diff','HNSCC_sig','ECM','Immune','EMT','Stem_cell','IFN_response','Other']:
    sub = df[df['category']==cat]['gene_wise_pcc'].dropna()
    if len(sub) > 0:
        print(f'{cat:>15}: n={len(sub):>4}, mean={sub.mean():>6.3f}, '
              f'min={sub.min():>6.3f}, max={sub.max():>6.3f}')

# ── 시각화 ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Gene-wise PCC Analysis: Prediction vs Ground Truth', fontsize=13)

# 1. ST nonzero% vs gene-wise PCC
axes[0,0].scatter(gene_nonzero_pct[valid_mask], gene_pccs[valid_mask],
                  alpha=0.3, s=5, color='steelblue')
axes[0,0].set_xlabel('ST nonzero % (발현된 spot 비율)')
axes[0,0].set_ylabel('Gene-wise PCC')
axes[0,0].set_title(f'ST Nonzero% vs Gene-wise PCC\nr={r_nonzero:.4f}')
axes[0,0].grid(alpha=0.3)
axes[0,0].axhline(0, color='red', linestyle='--', alpha=0.5)

# 2. ST mean vs gene-wise PCC
axes[0,1].scatter(gene_mean_st[valid_mask], gene_pccs[valid_mask],
                  alpha=0.3, s=5, color='orange')
axes[0,1].set_xlabel('ST mean expression')
axes[0,1].set_ylabel('Gene-wise PCC')
axes[0,1].set_title(f'ST Mean Expression vs Gene-wise PCC\nr={r_stmean:.4f}')
axes[0,1].grid(alpha=0.3)
axes[0,1].axhline(0, color='red', linestyle='--', alpha=0.5)

# 3. Category별 PCC 분포
cat_order = ['Squamous_diff','HNSCC_sig','ECM','Immune','EMT','Stem_cell','IFN_response']
cat_data  = [df[df['category']==c]['gene_wise_pcc'].dropna().values for c in cat_order]
cat_colors = ['#e74c3c','#f39c12','#3498db','#9b59b6','#2ecc71','#e67e22','#1abc9c']
bp = axes[1,0].boxplot(cat_data, labels=cat_order, patch_artist=True)
for patch, color in zip(bp['boxes'], cat_colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1,0].set_xticklabels(cat_order, rotation=30, ha='right', fontsize=8)
axes[1,0].set_ylabel('Gene-wise PCC')
axes[1,0].set_title('Gene Category별 PCC 분포')
axes[1,0].axhline(0, color='black', linestyle='--', alpha=0.4)
axes[1,0].grid(axis='y', alpha=0.3)

# 4. pred_mean vs bulk_mean scatter
axes[1,1].scatter(bulk_mean[valid_mask], pred_mean[valid_mask],
                  alpha=0.3, s=5, c=gene_pccs[valid_mask], cmap='RdYlGn',
                  vmin=-0.2, vmax=0.6)
axes[1,1].plot([0, bulk_mean.max()], [0, bulk_mean.max()], 'k--', alpha=0.3, label='y=x')
axes[1,1].set_xlabel('Bulk mean expression (ground truth)')
axes[1,1].set_ylabel('Pred mean expression (top-3 sum)')
axes[1,1].set_title('Predicted vs Ground Truth\n(color=gene-wise PCC)')
axes[1,1].legend(fontsize=8)
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/gene_pcc_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: gene_pcc_analysis.png')
print('Done!')
EOF