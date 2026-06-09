"""
TCGA 144px K% Sparse Retrieval Sweep only (fold_03)
- baseline_pred.npy / baseline_bulk.npy 이미 있으므로 로드해서 사용
- K sweep: 슬라이드당 배치 처리, topk_expr 미리 안 만들고 배치 안에서 완결
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import torch.nn.functional as F

GENE_LIST    = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
EMB_BASE     = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px"
FINETUNE_EMB = "/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/144px_finetuned_ksweep"
FOLD         = "fold_03"
K_VALUES     = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
TILE_BATCH   = 32  # OOM 방지

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | {torch.cuda.get_device_name(0)}")

# ─── 기존 결과 로드 ────────────────────────────────────────────────────────────
print("[Step 1] Loading existing baseline results...")
ft_bulk_arr  = np.load(os.path.join(OUTPUT_DIR, "baseline_bulk.npy"))
common_genes = np.load(os.path.join(OUTPUT_DIR, "common_genes.npy"), allow_pickle=True).tolist()
slide_ids    = np.load(os.path.join(OUTPUT_DIR, "slide_ids.npy"),    allow_pickle=True).tolist()
ft_pred_arr  = np.load(os.path.join(OUTPUT_DIR, "baseline_pred.npy"))
print(f"  Slides: {len(slide_ids)}, Common genes: {len(common_genes)}")

# baseline PCC 출력
def gene_pcc(pred_arr, bulk_arr):
    pccs, genes = [], []
    for i, g in enumerate(common_genes):
        p, b = pred_arr[:, i], bulk_arr[:, i]
        if p.std() < 1e-8 or b.std() < 1e-8: continue
        r, _ = pearsonr(p, b); pccs.append(r); genes.append(g)
    return np.array(pccs), genes

def slide_pcc(pred_arr, bulk_arr):
    pccs = []
    for p, b in zip(pred_arr, bulk_arr):
        if p.std() < 1e-8 or b.std() < 1e-8: pccs.append(np.nan); continue
        r, _ = pearsonr(p, b); pccs.append(r)
    return np.array(pccs)

ft_gpccs, _ = gene_pcc(ft_pred_arr, ft_bulk_arr)
ft_spccs = slide_pcc(ft_pred_arr, ft_bulk_arr)
v = ~np.isnan(ft_spccs)
print(f"  Baseline | gene_pcc={ft_gpccs.mean():.4f}  slide_pcc={ft_spccs[v].mean():.4f}")

# ─── ST corpus → GPU ───────────────────────────────────────────────────────────
print("[Step 2] Loading ST corpus → GPU...")
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]
pred_idx_np = np.array([gene_list.index(g) for g in common_genes])

ft_st_dir  = os.path.join(FINETUNE_EMB, FOLD)
train_img  = np.load(os.path.join(ft_st_dir, "train_img_embs.npy"))
train_expr = np.load(os.path.join(ft_st_dir, "train_exprs.npy"))
train_img_gpu  = F.normalize(torch.tensor(train_img,  dtype=torch.float32, device=device), dim=-1)
train_expr_gpu = torch.tensor(train_expr, dtype=torch.float32, device=device)
N_SPOTS = train_img_gpu.shape[0]
print(f"  ST corpus: {train_img_gpu.shape}, mem={torch.cuda.memory_allocated()/1e9:.2f}GB")

# ─── Tile embedding 경로 매핑 ──────────────────────────────────────────────────
print("[Step 3] Mapping tile embeddings...")
emb_dir   = os.path.join(EMB_BASE, FOLD)
emb_files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".npy") and "_coords" not in f])
slide_emb_map = {fname.split(".")[0]: os.path.join(emb_dir, fname) for fname in emb_files}
# slide_ids 순서 유지
slide_emb_list = [(sid, slide_emb_map[sid]) for sid in slide_ids if sid in slide_emb_map]
print(f"  Mapped: {len(slide_emb_list)} slides")

# ─── K sweep 함수: 타일 배치 안에서 완결 ──────────────────────────────────────
@torch.no_grad()
def predict_all_k(tile_np):
    """
    (T, 768) numpy → {k_pct: (n_common,) numpy}
    배치마다 topk → weighted sum → accum, (T, k_max, 2000) 텐서 안 만듦
    """
    k_max    = max(1, int(N_SPOTS * max(K_VALUES) / 100))
    tile_gpu = F.normalize(torch.tensor(tile_np, dtype=torch.float32, device=device), dim=-1)
    T        = tile_gpu.shape[0]
    k_accum  = {k: torch.zeros(2000, device=device) for k in K_VALUES}

    for i in range(0, T, TILE_BATCH):
        batch             = tile_gpu[i:i+TILE_BATCH]          # (B, 768)
        sim               = batch @ train_img_gpu.T            # (B, N)
        topk_sim, topk_idx = sim.topk(k_max, dim=1)           # (B, k_max)

        for k_pct in K_VALUES:
            k      = max(1, int(N_SPOTS * k_pct / 100))
            s_k    = torch.clamp(topk_sim[:, :k], min=0)      # (B, k)
            w_k    = s_k / (s_k.sum(dim=1, keepdim=True) + 1e-8)
            idx_k  = topk_idx[:, :k]                           # (B, k)
            expr_k = train_expr_gpu[idx_k]                     # (B, k, 2000)
            pred_b = (w_k.unsqueeze(-1) * expr_k).sum(dim=1)  # (B, 2000)
            k_accum[k_pct] += pred_b.sum(dim=0)

    return {k_pct: (k_accum[k_pct] / T).cpu().numpy()[pred_idx_np]
            for k_pct in K_VALUES}

# ─── K sweep 메인 ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nK% Sparse Retrieval Sweep ({FOLD}, 144px)\n{'='*60}")

k_preds_dict = {k: [] for k in K_VALUES}

for sid, emb_path in tqdm(slide_emb_list, desc="K sweep"):
    tile_np = np.load(emb_path).astype(np.float32)
    results = predict_all_k(tile_np)
    for k_pct, pred in results.items():
        k_preds_dict[k_pct].append(pred)

# ─── PCC 계산 및 저장 ──────────────────────────────────────────────────────────
sweep_results = []
for k_pct in K_VALUES:
    k_pred_arr = np.array(k_preds_dict[k_pct])
    k_gpccs, _ = gene_pcc(k_pred_arr, ft_bulk_arr)
    k_spccs    = slide_pcc(k_pred_arr, ft_bulk_arr)
    valid      = ~np.isnan(k_spccs)
    var_ratio  = k_pred_arr.var(axis=0).mean() / (ft_bulk_arr.var(axis=0).mean() + 1e-8)

    sweep_results.append({
        "k_pct":           k_pct,
        "gene_pcc_mean":   k_gpccs.mean(),
        "gene_pcc_median": np.median(k_gpccs),
        "slide_pcc_mean":  k_spccs[valid].mean(),
        "var_ratio":       var_ratio,
    })
    print(f"  K={k_pct:3d}% | gene_pcc={k_gpccs.mean():.4f}  "
          f"slide_pcc={k_spccs[valid].mean():.4f}  var_ratio={var_ratio:.4f}")

sweep_df = pd.DataFrame(sweep_results)
sweep_df.to_csv(os.path.join(OUTPUT_DIR, "ksweep_results.csv"), index=False)

best_k   = sweep_df.loc[sweep_df["gene_pcc_mean"].idxmax(), "k_pct"]
best_pcc = sweep_df["gene_pcc_mean"].max()
print(f"\n  Baseline (full): {ft_gpccs.mean():.4f}")
print(f"  Best K: {best_k}%  (gene_pcc={best_pcc:.4f})")
print(f"  Improvement: +{best_pcc - ft_gpccs.mean():.4f}")
print(f"\nSaved to {OUTPUT_DIR}\nDone!")