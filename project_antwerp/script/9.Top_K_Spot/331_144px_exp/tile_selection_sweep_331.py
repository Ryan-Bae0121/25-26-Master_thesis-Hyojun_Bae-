"""
Tile Selection Sweep - 331 patient-level slides
- K=30% sparse retrieval for tile-wise PCC scoring
- Tile ranking: tile-wise PCC vs cosine similarity (comparison)
- Sweep: K = 50, 100, 200, 500, 1000 tiles + all-tiles baseline
- fold_03, 144px
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import torch.nn.functional as F

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
REF_FILE     = "/project_antwerp/hbae/ref_file_331.csv"
GENE_LIST    = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
EMB_BASE     = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px"
FINETUNE_EMB = "/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tile_selection_331"
FOLD         = "fold_03"
K_SPARSE_PCT = 30       # sparse retrieval K%
TILE_KS      = [50, 100, 200, 500, 1000]  # tile selection sweep
TILE_BATCH   = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | {torch.cuda.get_device_name(0)}")

# ─── 유전자 & ref file ────────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
rna_cols     = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes    = [c.replace("rna_", "") for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
bulk_cols    = ["rna_" + g for g in common_genes]
pred_idx_np  = np.array([gene_list.index(g) for g in common_genes])
print(f"Common genes: {len(common_genes)}, Slides: {len(ref_df)}")

# ─── ST corpus → GPU ──────────────────────────────────────────────────────────
ft_st_dir      = os.path.join(FINETUNE_EMB, FOLD)
train_img      = np.load(os.path.join(ft_st_dir, "train_img_embs.npy"))
train_expr     = np.load(os.path.join(ft_st_dir, "train_exprs.npy"))
train_img_gpu  = F.normalize(
    torch.tensor(train_img,  dtype=torch.float32, device=device), dim=-1)
train_expr_gpu = torch.tensor(
    train_expr, dtype=torch.float32, device=device)
N_SPOTS  = train_img_gpu.shape[0]
K_SPARSE = max(1, int(N_SPOTS * K_SPARSE_PCT / 100))
print(f"ST corpus: {train_img_gpu.shape}, K={K_SPARSE_PCT}% = {K_SPARSE} spots")

# ─── 임베딩 매핑 ──────────────────────────────────────────────────────────────
emb_dir   = os.path.join(EMB_BASE, FOLD)
emb_files = sorted([f for f in os.listdir(emb_dir)
                    if f.endswith(".npy") and "_coords" not in f])
slide_emb_map = {fname.split(".")[0]: os.path.join(emb_dir, fname)
                 for fname in emb_files}
matched = [(row["slide_id"], slide_emb_map[row["slide_id"]], row)
           for _, row in ref_df.iterrows()
           if row["slide_id"] in slide_emb_map]
print(f"Matched: {len(matched)} slides")

# ─── PCC helpers ──────────────────────────────────────────────────────────────
def gene_pcc_arr(pred_arr, bulk_arr):
    pccs = []
    for i in range(pred_arr.shape[1]):
        p, b = pred_arr[:, i], bulk_arr[:, i]
        if p.std() < 1e-8 or b.std() < 1e-8:
            continue
        r, _ = pearsonr(p, b)
        pccs.append(r)
    return np.array(pccs)

def slide_pcc_arr(pred_arr, bulk_arr):
    pccs = []
    for p, b in zip(pred_arr, bulk_arr):
        if p.std() < 1e-8 or b.std() < 1e-8:
            pccs.append(np.nan)
            continue
        r, _ = pearsonr(p, b)
        pccs.append(r)
    return np.array(pccs)

# ─── 슬라이드별 tile-wise PCC + cosine sim 점수 + 전체 예측 계산 ──────────────
@torch.no_grad()
def compute_tile_scores(tile_np, bulk):
    """
    반환:
      tile_preds: (T, n_common) - K=30% sparse retrieval 예측
      tile_pcc_scores: (T,) - tile-wise PCC vs bulk
      cos_scores: (T,) - cosine similarity vs bulk
      all_tile_pred: (n_common,) - 전체 타일 평균 예측 (baseline)
    """
    tile_gpu = F.normalize(
        torch.tensor(tile_np, dtype=torch.float32, device=device), dim=-1)
    T = tile_gpu.shape[0]
    bulk_t = torch.tensor(bulk, dtype=torch.float32, device=device)

    all_preds = []
    for i in range(0, T, TILE_BATCH):
        batch              = tile_gpu[i:i+TILE_BATCH]
        sim                = batch @ train_img_gpu.T          # (B, N)
        topk_sim, topk_idx = sim.topk(K_SPARSE, dim=1)       # (B, K_SPARSE)
        s_k  = torch.clamp(topk_sim, min=0)
        w_k  = s_k / (s_k.sum(dim=1, keepdim=True) + 1e-8)
        expr_k = train_expr_gpu[topk_idx]                     # (B, K_SPARSE, 2000)
        pred_b = (w_k.unsqueeze(-1) * expr_k).sum(dim=1)     # (B, 2000)
        all_preds.append(pred_b)

    all_preds_gpu = torch.cat(all_preds, dim=0)               # (T, 2000)
    tile_preds    = all_preds_gpu[:, pred_idx_np].cpu().numpy()  # (T, n_common)

    # tile-wise PCC vs bulk (GPU 벡터화)
    P   = torch.tensor(tile_preds, dtype=torch.float32, device=device)
    b   = torch.tensor(bulk,       dtype=torch.float32, device=device)
    P_c = P - P.mean(dim=1, keepdim=True)
    b_c = b - b.mean()
    tile_pcc_scores = (
        (P_c * b_c).sum(dim=1) /
        (P_c.norm(dim=1) * b_c.norm() + 1e-8)
    ).cpu().numpy()

    # cosine similarity vs bulk (정규화된 벡터)
    P_norm = F.normalize(P, dim=1)
    b_norm = F.normalize(b.unsqueeze(0), dim=1)
    cos_scores = (P_norm @ b_norm.T).squeeze().cpu().numpy()

    # 전체 타일 평균 (all-tiles baseline)
    all_tile_pred = tile_preds.mean(axis=0)

    return tile_preds, tile_pcc_scores, cos_scores, all_tile_pred

# ─── 메인 루프 ────────────────────────────────────────────────────────────────
print(f"\nProcessing {len(matched)} slides...")

# 결과 저장용
results_pcc  = {k: [] for k in TILE_KS}   # tile-wise PCC ranking
results_cos  = {k: [] for k in TILE_KS}   # cosine sim ranking
results_all  = []                           # all-tiles baseline
all_bulks    = []
slide_summary = []

for sid, emb_path, row in tqdm(matched, desc="Tile selection"):
    tile_np = np.load(emb_path).astype(np.float32)
    bulk    = row[bulk_cols].values.astype(float)
    T       = tile_np.shape[0]

    tile_preds, pcc_scores, cos_scores, all_tile_pred = \
        compute_tile_scores(tile_np, bulk)

    # all-tiles baseline
    results_all.append(all_tile_pred)
    all_bulks.append(bulk)

    # tile ranking별 top-K 선택
    valid_mask = pcc_scores > -999
    pcc_sorted = np.where(valid_mask)[0][
        np.argsort(pcc_scores[valid_mask])[::-1]]
    cos_sorted = np.argsort(cos_scores)[::-1]

    for k in TILE_KS:
        kk = min(k, T)

        # tile-wise PCC ranking
        top_idx_pcc = pcc_sorted[:kk]
        results_pcc[k].append(tile_preds[top_idx_pcc].mean(axis=0))

        # cosine similarity ranking
        top_idx_cos = cos_sorted[:kk]
        results_cos[k].append(tile_preds[top_idx_cos].mean(axis=0))

    # slide summary (biological validation용)
    slide_summary.append({
        "slide_id":        sid,
        "n_tiles":         T,
        "mean_pcc_score":  pcc_scores[valid_mask].mean() if valid_mask.sum() > 0 else np.nan,
        "top500_mean_pcc": pcc_scores[pcc_sorted[:min(500, T)]].mean()
                           if T >= 1 else np.nan,
    })

# ─── 결과 계산 ────────────────────────────────────────────────────────────────
bulk_arr = np.array(all_bulks)

print(f"\n{'='*65}")
print(f"Tile Selection Results (fold_03, 331 slides, K={K_SPARSE_PCT}% sparse)")
print(f"{'='*65}")
print(f"{'K':>6} | {'Method':>14} | {'gene_pcc':>10} | "
      f"{'slide_pcc':>10} | {'var_ratio':>10}")
print(f"  {'-'*55}")

sweep_rows = []

# all-tiles baseline
all_arr      = np.array(results_all)
base_gpccs   = gene_pcc_arr(all_arr, bulk_arr)
base_spccs   = slide_pcc_arr(all_arr, bulk_arr)
base_var     = (all_arr.var(axis=0).mean() /
                (bulk_arr.var(axis=0).mean() + 1e-8))
valid_b      = ~np.isnan(base_spccs)
print(f"{'all':>6} | {'baseline':>14} | "
      f"{base_gpccs.mean():>10.4f} | "
      f"{base_spccs[valid_b].mean():>10.4f} | "
      f"{base_var:>10.4f}")
sweep_rows.append({
    "k_tiles": "all", "method": "baseline",
    "gene_pcc": base_gpccs.mean(),
    "slide_pcc": base_spccs[valid_b].mean(),
    "var_ratio": base_var
})

for k in TILE_KS:
    # tile-wise PCC ranking
    pcc_arr   = np.array(results_pcc[k])
    g_pcc     = gene_pcc_arr(pcc_arr, bulk_arr)
    s_pcc     = slide_pcc_arr(pcc_arr, bulk_arr)
    var_r     = pcc_arr.var(axis=0).mean() / (bulk_arr.var(axis=0).mean() + 1e-8)
    valid_s   = ~np.isnan(s_pcc)
    print(f"{k:>6} | {'tile_pcc':>14} | "
          f"{g_pcc.mean():>10.4f} | "
          f"{s_pcc[valid_s].mean():>10.4f} | "
          f"{var_r:>10.4f}")
    sweep_rows.append({
        "k_tiles": k, "method": "tile_pcc",
        "gene_pcc": g_pcc.mean(),
        "slide_pcc": s_pcc[valid_s].mean(),
        "var_ratio": var_r
    })

    # cosine similarity ranking
    cos_arr   = np.array(results_cos[k])
    g_cos     = gene_pcc_arr(cos_arr, bulk_arr)
    s_cos     = slide_pcc_arr(cos_arr, bulk_arr)
    var_cos   = cos_arr.var(axis=0).mean() / (bulk_arr.var(axis=0).mean() + 1e-8)
    valid_c   = ~np.isnan(s_cos)
    print(f"{k:>6} | {'cosine_sim':>14} | "
          f"{g_cos.mean():>10.4f} | "
          f"{s_cos[valid_c].mean():>10.4f} | "
          f"{var_cos:>10.4f}")
    sweep_rows.append({
        "k_tiles": k, "method": "cosine_sim",
        "gene_pcc": g_cos.mean(),
        "slide_pcc": s_cos[valid_c].mean(),
        "var_ratio": var_cos
    })

# ─── 저장 ────────────────────────────────────────────────────────────────────
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(os.path.join(OUTPUT_DIR, "tile_selection_sweep.csv"),
                index=False)

summary_df = pd.DataFrame(slide_summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "slide_summary.csv"),
                  index=False)

# top-500 tile predictions 저장 (biological validation용)
np.save(os.path.join(OUTPUT_DIR, "top500_preds.npy"),
        np.array(results_pcc[500]))
np.save(os.path.join(OUTPUT_DIR, "bulk_arr.npy"), bulk_arr)

print(f"\nSaved to {OUTPUT_DIR}")
print("Done!")