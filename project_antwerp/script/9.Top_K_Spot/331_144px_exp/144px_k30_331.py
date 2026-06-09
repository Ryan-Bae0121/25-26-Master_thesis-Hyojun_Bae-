"""
331 slides K=30% 검증만 (빠른 버전)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import torch.nn.functional as F

REF_FILE     = "/project_antwerp/hbae/ref_file_331.csv"
GENE_LIST    = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
EMB_BASE     = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px"
FINETUNE_EMB = "/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/144px_finetuned_ksweep_331"
FOLD         = "fold_03"
K_PCT        = 30   # K=30%만
TILE_BATCH   = 128  # 크게 해도 됨 (K=30%만 하면 메모리 OK)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | {torch.cuda.get_device_name(0)}")

# ─── 로드 ──────────────────────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
rna_cols     = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes    = [c.replace("rna_", "") for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
bulk_cols    = ["rna_" + g for g in common_genes]
pred_idx_np  = np.array([gene_list.index(g) for g in common_genes])

# baseline 로드
ft_pred_arr = np.load(os.path.join(OUTPUT_DIR, "baseline_pred.npy"))
ft_bulk_arr = np.load(os.path.join(OUTPUT_DIR, "baseline_bulk.npy"))
slide_ids   = np.load(os.path.join(OUTPUT_DIR, "slide_ids.npy"),
                       allow_pickle=True).tolist()

# ─── ST corpus → GPU ──────────────────────────────────────────────────────────
ft_st_dir      = os.path.join(FINETUNE_EMB, FOLD)
train_img      = np.load(os.path.join(ft_st_dir, "train_img_embs.npy"))
train_expr     = np.load(os.path.join(ft_st_dir, "train_exprs.npy"))
train_img_gpu  = F.normalize(
    torch.tensor(train_img,  dtype=torch.float32, device=device), dim=-1)
train_expr_gpu = torch.tensor(
    train_expr, dtype=torch.float32, device=device)
N_SPOTS = train_img_gpu.shape[0]
K_SPOTS = max(1, int(N_SPOTS * K_PCT / 100))
print(f"ST corpus: {train_img_gpu.shape}, K={K_PCT}% = {K_SPOTS} spots")

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

# ─── PCC helper ───────────────────────────────────────────────────────────────
def gene_pcc(pred_arr, bulk_arr):
    pccs = []
    for i in range(pred_arr.shape[1]):
        p, b = pred_arr[:, i], bulk_arr[:, i]
        if p.std() < 1e-8 or b.std() < 1e-8:
            continue
        r, _ = pearsonr(p, b)
        pccs.append(r)
    return np.array(pccs)

def slide_pcc(pred_arr, bulk_arr):
    pccs = []
    for p, b in zip(pred_arr, bulk_arr):
        if p.std() < 1e-8 or b.std() < 1e-8:
            pccs.append(np.nan)
            continue
        r, _ = pearsonr(p, b)
        pccs.append(r)
    return np.array(pccs)

# ─── K=30% 예측 ───────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_k30(tile_np):
    tile_gpu = F.normalize(
        torch.tensor(tile_np, dtype=torch.float32, device=device), dim=-1)
    T     = tile_gpu.shape[0]
    accum = torch.zeros(2000, device=device)
    for i in range(0, T, TILE_BATCH):
        batch              = tile_gpu[i:i+TILE_BATCH]
        sim                = batch @ train_img_gpu.T
        topk_sim, topk_idx = sim.topk(K_SPOTS, dim=1)
        s_k  = torch.clamp(topk_sim, min=0)
        w_k  = s_k / (s_k.sum(dim=1, keepdim=True) + 1e-8)
        expr_k = train_expr_gpu[topk_idx]           # (B, K_SPOTS, 2000)
        pred_b = (w_k.unsqueeze(-1) * expr_k).sum(dim=1)
        accum += pred_b.sum(dim=0)
    return (accum / T).cpu().numpy()[pred_idx_np]

# ─── 메인 ─────────────────────────────────────────────────────────────────────
print(f"\nRunning K={K_PCT}% on {len(matched)} slides...")
k30_preds, k30_bulks = [], []

for sid, emb_path, row in tqdm(matched, desc=f"K={K_PCT}%"):
    tile_np = np.load(emb_path).astype(np.float32)
    pred    = predict_k30(tile_np)
    bulk    = row[bulk_cols].values.astype(float)
    k30_preds.append(pred)
    k30_bulks.append(bulk)

k30_pred_arr = np.array(k30_preds)
k30_bulk_arr = np.array(k30_bulks)

# ─── 결과 출력 ────────────────────────────────────────────────────────────────
baseline_gpccs = gene_pcc(ft_pred_arr, ft_bulk_arr)
k30_gpccs      = gene_pcc(k30_pred_arr, k30_bulk_arr)
k30_spccs      = slide_pcc(k30_pred_arr, k30_bulk_arr)
valid          = ~np.isnan(k30_spccs)
var_ratio      = (k30_pred_arr.var(axis=0).mean() /
                  (k30_bulk_arr.var(axis=0).mean() + 1e-8))

print(f"\n{'='*50}")
print(f"Results (fold_03, 331 slides)")
print(f"{'='*50}")
print(f"  Baseline (K=100%) | gene_pcc = {baseline_gpccs.mean():.4f}")
print(f"  K={K_PCT}%            | gene_pcc = {k30_gpccs.mean():.4f}")
print(f"  Improvement       | +{k30_gpccs.mean() - baseline_gpccs.mean():.4f}")
print(f"  Slide-wise PCC    | {k30_spccs[valid].mean():.4f}")
print(f"  Var ratio         | {var_ratio:.4f}  "
      f"(baseline est: 0.073)")

# 저장
np.save(os.path.join(OUTPUT_DIR, "k30_pred.npy"), k30_pred_arr)
pd.DataFrame({
    "metric": ["baseline_gene_pcc", "k30_gene_pcc",
                "improvement", "var_ratio"],
    "value":  [baseline_gpccs.mean(), k30_gpccs.mean(),
               k30_gpccs.mean() - baseline_gpccs.mean(), var_ratio]
}).to_csv(os.path.join(OUTPUT_DIR, "k30_results.csv"), index=False)

print(f"\nSaved to {OUTPUT_DIR}")
print("Done!")