"""
K=300 + median 보완 계산
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os

REF_FILE     = "/project_antwerp/hbae/ref_file_331.csv"
GENE_LIST    = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
EMB_BASE     = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px"
FINETUNE_EMB = "/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tile_selection_331"
FOLD         = "fold_03"
K_SPARSE_PCT = 30
TILE_BATCH   = 64
ADD_KS       = [300]  # 추가할 K값

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
rna_cols     = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes    = [c.replace("rna_", "") for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
bulk_cols    = ["rna_" + g for g in common_genes]
pred_idx_np  = np.array([gene_list.index(g) for g in common_genes])

bulk_arr = np.load(os.path.join(OUTPUT_DIR, "bulk_arr.npy"))

ft_st_dir      = os.path.join(FINETUNE_EMB, FOLD)
train_img      = np.load(os.path.join(ft_st_dir, "train_img_embs.npy"))
train_expr     = np.load(os.path.join(ft_st_dir, "train_exprs.npy"))
train_img_gpu  = F.normalize(
    torch.tensor(train_img,  dtype=torch.float32, device=device), dim=-1)
train_expr_gpu = torch.tensor(
    train_expr, dtype=torch.float32, device=device)
N_SPOTS  = train_img_gpu.shape[0]
K_SPARSE = max(1, int(N_SPOTS * K_SPARSE_PCT / 100))

emb_dir   = os.path.join(EMB_BASE, FOLD)
emb_files = sorted([f for f in os.listdir(emb_dir)
                    if f.endswith(".npy") and "_coords" not in f])
slide_emb_map = {fname.split(".")[0]: os.path.join(emb_dir, fname)
                 for fname in emb_files}
matched = [(row["slide_id"], slide_emb_map[row["slide_id"]], row)
           for _, row in ref_df.iterrows()
           if row["slide_id"] in slide_emb_map]

# 기존 sweep 결과 로드
existing = pd.read_csv(os.path.join(OUTPUT_DIR, "tile_selection_sweep.csv"))
print("Existing results:")
print(existing[existing["method"] == "tile_pcc"][
    ["k_tiles", "gene_pcc", "slide_pcc"]].to_string())

@torch.no_grad()
def compute_tile_preds(tile_np):
    tile_gpu = F.normalize(
        torch.tensor(tile_np, dtype=torch.float32, device=device), dim=-1)
    T = tile_gpu.shape[0]
    all_preds = []
    for i in range(0, T, TILE_BATCH):
        batch              = tile_gpu[i:i+TILE_BATCH]
        sim                = batch @ train_img_gpu.T
        topk_sim, topk_idx = sim.topk(K_SPARSE, dim=1)
        s_k    = torch.clamp(topk_sim, min=0)
        w_k    = s_k / (s_k.sum(dim=1, keepdim=True) + 1e-8)
        expr_k = train_expr_gpu[topk_idx]
        pred_b = (w_k.unsqueeze(-1) * expr_k).sum(dim=1)
        all_preds.append(pred_b)
    preds = torch.cat(all_preds, dim=0)
    return preds[:, pred_idx_np].cpu().numpy()

def gene_pcc_stats(pred_arr, bulk_arr):
    pccs = []
    for i in range(pred_arr.shape[1]):
        p, b = pred_arr[:, i], bulk_arr[:, i]
        if p.std() < 1e-8 or b.std() < 1e-8:
            continue
        r, _ = pearsonr(p, b)
        pccs.append(r)
    pccs = np.array(pccs)
    return pccs.mean(), np.median(pccs)

# K=300 + 기존 K들 median 계산
ALL_KS = [50, 100, 200, 300, 500, 1000]
results_pcc = {k: [] for k in ALL_KS}

print(f"\nProcessing {len(matched)} slides...")
for sid, emb_path, row in tqdm(matched, desc="Computing"):
    tile_np   = np.load(emb_path).astype(np.float32)
    bulk      = row[bulk_cols].values.astype(float)
    T         = tile_np.shape[0]
    tile_preds = compute_tile_preds(tile_np)

    # tile_pcc 점수
    P   = torch.tensor(tile_preds, dtype=torch.float32, device=device)
    b   = torch.tensor(bulk,       dtype=torch.float32, device=device)
    P_c = P - P.mean(dim=1, keepdim=True)
    b_c = b - b.mean()
    scores = ((P_c * b_c).sum(dim=1) /
              (P_c.norm(dim=1) * b_c.norm() + 1e-8)).cpu().numpy()

    valid_idx  = np.where(scores > -999)[0]
    pcc_sorted = valid_idx[np.argsort(scores[valid_idx])[::-1]]

    for k in ALL_KS:
        kk = min(k, T)
        top_idx = pcc_sorted[:kk]
        results_pcc[k].append(tile_preds[top_idx].mean(axis=0))

# 결과 출력
print(f"\n{'='*55}")
print(f"Updated Results (331 slides, K=30% sparse, tile_pcc)")
print(f"{'='*55}")
print(f"{'K':>6} | {'mean':>8} | {'median':>8} | {'slide_pcc':>10}")
print(f"  {'-'*45}")

for k in ALL_KS:
    pred_arr  = np.array(results_pcc[k])
    g_mean, g_med = gene_pcc_stats(pred_arr, bulk_arr)
    s_pccs = []
    for p, b in zip(pred_arr, bulk_arr):
        if p.std() < 1e-8 or b.std() < 1e-8:
            s_pccs.append(np.nan)
            continue
        r, _ = pearsonr(p, b)
        s_pccs.append(r)
    s_pccs = np.array(s_pccs)
    valid  = ~np.isnan(s_pccs)
    print(f"{k:>6} | {g_mean:>8.4f} | {g_med:>8.4f} | "
          f"{s_pccs[valid].mean():>10.4f}")

# all-tiles baseline median
all_preds = np.load(os.path.join(OUTPUT_DIR, "tile_selection_sweep.csv"))
# baseline는 이미 0.0391 알고 있음
print(f"{'all':>6} | {'0.0391':>8} | {'?':>8} | {'0.7156':>10}")
print("\nDone!")