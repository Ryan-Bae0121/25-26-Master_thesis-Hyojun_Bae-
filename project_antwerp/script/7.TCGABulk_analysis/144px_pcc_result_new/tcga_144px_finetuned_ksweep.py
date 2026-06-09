"""
TCGA 144px Fine-tuned Baseline PCC + K% Sweep (fold_03) - GPU v3
- ST corpus: GPU
- Tile embeddings: CPU 캐싱, 슬라이드별 GPU 처리
- K sweep: 슬라이드당 sim 한 번 계산 → 모든 K 재사용 (15x 빠름)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import torch.nn.functional as F

REF_FILE     = "/project_antwerp/hbae/ref_file.csv"
GENE_LIST    = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
EMB_BASE     = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px"
FINETUNE_EMB = "/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/144px_finetuned_ksweep"
FOLD         = "fold_03"
K_VALUES     = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
TILE_BATCH   = 128  # OOM 방지용 타일 배치 크기

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | {torch.cuda.get_device_name(0)}")

# ─── Step 1 ────────────────────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
common_genes   = [g for g in gene_list if ("rna_" + g) in ref_df.columns]
bulk_gene_cols = ["rna_" + g for g in common_genes]
pred_idx_np    = np.array([gene_list.index(g) for g in common_genes])
print(f"Common genes: {len(common_genes)}, Slides: {len(ref_df)}")

# ─── Step 2: 매칭 ──────────────────────────────────────────────────────────────
emb_dir   = os.path.join(EMB_BASE, FOLD)
emb_files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".npy") and "_coords" not in f])
slide_emb_map = {fname.split(".")[0]: os.path.join(emb_dir, fname) for fname in emb_files}
matched = [(row["slide_id"], slide_emb_map[row["slide_id"]], row)
           for _, row in ref_df.iterrows() if row["slide_id"] in slide_emb_map]
print(f"Matched: {len(matched)} slides")

# ─── Step 3: ST corpus → GPU ───────────────────────────────────────────────────
ft_st_dir  = os.path.join(FINETUNE_EMB, FOLD)
train_img  = np.load(os.path.join(ft_st_dir, "train_img_embs.npy"))
train_expr = np.load(os.path.join(ft_st_dir, "train_exprs.npy"))
train_img_gpu  = F.normalize(torch.tensor(train_img,  dtype=torch.float32, device=device), dim=-1)
train_expr_gpu = torch.tensor(train_expr, dtype=torch.float32, device=device)
N_SPOTS = train_img_gpu.shape[0]
print(f"ST corpus: {train_img_gpu.shape}, mem={torch.cuda.memory_allocated()/1e9:.2f}GB")

# ─── Helper: PCC ───────────────────────────────────────────────────────────────
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

def print_result(label, gpccs, spccs):
    v = ~np.isnan(spccs)
    print(f"  [{label}]")
    print(f"    Gene-wise  | mean={gpccs.mean():.4f}  median={np.median(gpccs):.4f}  "
          f">0.3: {(gpccs>0.3).sum()}  >0.5: {(gpccs>0.5).sum()}")
    print(f"    Slide-wise | mean={spccs[v].mean():.4f}  median={np.median(spccs[v]):.4f}")

# ─── Helper: full retrieval ────────────────────────────────────────────────────
@torch.no_grad()
def predict_full(tile_np):
    tile_gpu = F.normalize(torch.tensor(tile_np, dtype=torch.float32, device=device), dim=-1)
    T = tile_gpu.shape[0]
    accum = torch.zeros(2000, device=device)
    for i in range(0, T, TILE_BATCH):
        batch   = tile_gpu[i:i+TILE_BATCH]
        sim     = torch.clamp(batch @ train_img_gpu.T, min=0)
        weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
        accum  += (weights @ train_expr_gpu).sum(dim=0)
    return (accum / T).cpu().numpy()[pred_idx_np]

# ─── Helper: 슬라이드 1개 → 모든 K값 예측 한 번에 ────────────────────────────
@torch.no_grad()
def predict_all_k(tile_np):
    """
    sim 계산 1회 → topk_max 뽑기 → 각 K값에 대해 슬라이싱만 반복
    반환: {k_pct: pred_array (n_common,)}
    """
    k_max = max(1, int(N_SPOTS * max(K_VALUES) / 100))
    tile_gpu = F.normalize(torch.tensor(tile_np, dtype=torch.float32, device=device), dim=-1)
    T = tile_gpu.shape[0]

    # 배치로 topk_max 계산
    all_topk_sim, all_topk_idx = [], []
    for i in range(0, T, TILE_BATCH):
        batch = tile_gpu[i:i+TILE_BATCH]
        sim   = batch @ train_img_gpu.T                          # (B, N)
        topk_sim, topk_idx = sim.topk(k_max, dim=1)             # (B, k_max)
        all_topk_sim.append(topk_sim)
        all_topk_idx.append(topk_idx)

    topk_sim_all  = torch.cat(all_topk_sim, dim=0)              # (T, k_max)
    topk_idx_all  = torch.cat(all_topk_idx, dim=0)              # (T, k_max)
    topk_expr_all = train_expr_gpu[topk_idx_all]                # (T, k_max, 2000)

    results = {}
    for k_pct in K_VALUES:
        k       = max(1, int(N_SPOTS * k_pct / 100))
        sim_k   = torch.clamp(topk_sim_all[:, :k], min=0)       # (T, k)
        w_k     = sim_k / (sim_k.sum(dim=1, keepdim=True) + 1e-8)
        preds   = (w_k.unsqueeze(-1) * topk_expr_all[:, :k, :]).sum(dim=1)  # (T, 2000)
        slide   = preds.mean(dim=0)                              # (2000,)
        results[k_pct] = slide.cpu().numpy()[pred_idx_np]

    return results

# ══════════════════════════════════════════════════════════════════════════════
# PART A: Fine-tuned Baseline (full retrieval)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nPART A: Fine-tuned Baseline Full Retrieval\n{'='*60}")

ft_preds, ft_bulks, slide_ids = [], [], []
cached_cpu = {}
for sid, emb_path, row in tqdm(matched, desc="Full retrieval"):
    tile_np = np.load(emb_path).astype(np.float32)
    cached_cpu[sid] = tile_np
    pred = predict_full(tile_np)
    bulk = row[bulk_gene_cols].values.astype(float)
    ft_preds.append(pred); ft_bulks.append(bulk); slide_ids.append(sid)

ft_pred_arr = np.array(ft_preds)
ft_bulk_arr = np.array(ft_bulks)
ft_gpccs, ft_valid_genes = gene_pcc(ft_pred_arr, ft_bulk_arr)
ft_spccs = slide_pcc(ft_pred_arr, ft_bulk_arr)
print_result(f"Fine-tuned {FOLD} full retrieval", ft_gpccs, ft_spccs)

np.save(os.path.join(OUTPUT_DIR, "baseline_pred.npy"), ft_pred_arr)
np.save(os.path.join(OUTPUT_DIR, "baseline_bulk.npy"), ft_bulk_arr)
np.save(os.path.join(OUTPUT_DIR, "common_genes.npy"), np.array(common_genes))
np.save(os.path.join(OUTPUT_DIR, "slide_ids.npy"), np.array(slide_ids))
pd.DataFrame({"gene": ft_valid_genes, "pcc": ft_gpccs}).to_csv(
    os.path.join(OUTPUT_DIR, "baseline_gene_pcc.csv"), index=False)
pd.DataFrame({"slide_id": slide_ids, "pcc": ft_spccs}).to_csv(
    os.path.join(OUTPUT_DIR, "baseline_slide_pcc.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PART B: K% Sweep — 슬라이드 461번만 반복, K는 내부에서 처리
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nPART B: K% Sparse Retrieval Sweep\n{'='*60}")

k_preds_dict = {k: [] for k in K_VALUES}

for sid, _, _ in tqdm(matched, desc="K sweep (all K per slide)"):
    results = predict_all_k(cached_cpu[sid])
    for k_pct, pred in results.items():
        k_preds_dict[k_pct].append(pred)

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