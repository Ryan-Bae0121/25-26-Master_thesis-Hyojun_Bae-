"""
Step 2: 144px Embedding으로 PredEx (Top 30% spots) + Sliding Window + Bulk PCC
변경: predex에서 전체 train spot 대신 cosine similarity 상위 30%만 사용
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
TCGA_EMB_DIR  = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px"
FINETUNE_EMB  = "/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
ZEROSHOT_EMB  = "/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03"
REF_FILE      = "/project_antwerp/hbae/ref_file.csv"
GENE_LIST     = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
HVG300_PATH   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tcga_bulk_hvg300_from_common.npy"
OUTPUT_DIR    = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/finetuned_ensemble_144px_top30pct"

WINDOW_SIZE = 10
STRIDE      = 5
MIN_TILES   = 50
TOP_K_FRAC  = 0.30   # ← 상위 30% spot만 사용
FOLDS       = ["fold_03"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Top-K fraction: {TOP_K_FRAC} (top {int(TOP_K_FRAC*100)}% spots)")

# ─── Gene list + ref_file ─────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"] + ".svs"
rna_cols     = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes    = [c.replace("rna_", "") for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
bulk_cols    = ["rna_" + g for g in common_genes]

common_in_hvg_idx = [gene_list.index(g) for g in common_genes]

print(f"Gene list: {len(gene_list)}")
print(f"Common genes: {len(common_genes)}")

# ─── Top 300 Gene 선택 ────────────────────────────────────────────────────────
val_exprs = np.load(os.path.join(ZEROSHOT_EMB, "val_exprs.npy"))
top300_A  = set(gene_list[i] for i in np.argsort(val_exprs.mean(axis=0))[::-1][:300])
idx_A     = [i for i, g in enumerate(common_genes) if g in top300_A]

hvg300_B = set(np.load(HVG300_PATH, allow_pickle=True).tolist())
idx_B    = [i for i, g in enumerate(common_genes) if g in hvg300_B]

print(f"Method A top300 in common: {len(idx_A)}")
print(f"Method B top300 in common: {len(idx_B)}")

# ─── matched slides ───────────────────────────────────────────────────────────
fold_03_dir  = os.path.join(TCGA_EMB_DIR, "fold_03")
matched_sids = []
slide_bulks_dict = {}

for _, row in ref_df.iterrows():
    sid = row["slide_id"]
    if os.path.exists(os.path.join(fold_03_dir, f"{sid}.npy")):
        matched_sids.append(sid)
        slide_bulks_dict[sid] = row[bulk_cols].values.astype(float)

print(f"Matched slides: {len(matched_sids)}")

# ─── Helper functions ─────────────────────────────────────────────────────────
def coords_to_grid(coords):
    y_px, x_px = coords[:, 0], coords[:, 1]
    x_vals    = np.unique(x_px)
    tile_step = int(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 512
    return pd.DataFrame({
        "tile_idx": np.arange(len(coords)),
        "x_grid":   ((x_px - x_px.min()) / tile_step).astype(int),
        "y_grid":   ((y_px - y_px.min()) / tile_step).astype(int)
    })

@torch.no_grad()
def predex_topk(emb, train_embs_gpu, train_exprs_gpu, top_k_frac=0.30):
    """상위 top_k_frac의 train spot만 사용해서 PredEx"""
    emb = emb.unsqueeze(0)  # (1, D)
    sim = emb @ train_embs_gpu.T  # (1, N_train)
    sim = torch.clamp(sim, min=0)

    # 상위 K개만 선택
    k = max(1, int(top_k_frac * train_embs_gpu.shape[0]))
    topk_vals, topk_idx = sim[0].topk(k)  # (K,)

    # 선택된 K개만으로 가중평균
    topk_vals = topk_vals.unsqueeze(0)  # (1, K)
    weights   = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-8)
    pred      = weights @ train_exprs_gpu[topk_idx]  # (1, 2000)
    return pred.squeeze(0).cpu().numpy()  # (2000,)

def sliding_window_predict(tile_embs_gpu, df_tiles, train_embs_gpu, train_exprs_gpu):
    max_x = df_tiles["x_grid"].max()
    max_y = df_tiles["y_grid"].max()
    tile_preds = defaultdict(list)
    n_windows  = 0

    for x in range(0, max_x + 1, STRIDE):
        for y in range(0, max_y + 1, STRIDE):
            mask = (
                (df_tiles["x_grid"] >= x) & (df_tiles["x_grid"] < x + WINDOW_SIZE) &
                (df_tiles["y_grid"] >= y) & (df_tiles["y_grid"] < y + WINDOW_SIZE)
            )
            window_df = df_tiles[mask]
            if len(window_df) < MIN_TILES:
                continue
            idxs     = window_df["tile_idx"].values
            mean_emb = F.normalize(tile_embs_gpu[idxs].mean(dim=0), dim=-1)
            pred     = predex_topk(mean_emb, train_embs_gpu, train_exprs_gpu, TOP_K_FRAC)
            for idx in idxs:
                tile_preds[idx].append(pred)
            n_windows += 1

    if n_windows == 0:
        return None, 0

    all_preds = [np.mean(tile_preds[idx], axis=0) for idx in sorted(tile_preds.keys())]
    return np.mean(all_preds, axis=0), n_windows

def calc_pcc(pred_arr, bulk_arr, idx, label):
    p, b = pred_arr[:, idx], bulk_arr[:, idx]
    gpccs = [pearsonr(p[:,i], b[:,i])[0]
             for i in range(p.shape[1])
             if p[:,i].std() > 1e-8 and b[:,i].std() > 1e-8]
    spccs = [pearsonr(p[i], b[i])[0]
             for i in range(p.shape[0])
             if p[i].std() > 1e-8 and b[i].std() > 1e-8]
    g = np.array(gpccs)
    s = np.array(spccs)
    print(f"  [{label}] Gene: mean={g.mean():.4f} median={np.median(g):.4f} "
          f"PCC>0.1:{(g>0.1).sum()} | Slide: mean={s.mean():.4f} median={np.median(s):.4f}")
    return g, s

# ─── 메인 ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Fold-wise Results (Top 30% spots)")
print("="*60)

fold_results   = {}
ensemble_preds = defaultdict(list)

for fold in FOLDS:
    fold_emb_dir   = os.path.join(TCGA_EMB_DIR, fold)
    train_emb_path = os.path.join(FINETUNE_EMB, fold, "train_img_embs.npy")
    train_exp_path = os.path.join(FINETUNE_EMB, fold, "train_exprs.npy")

    if not os.path.exists(fold_emb_dir) or not os.path.exists(train_emb_path):
        print(f"\n[{fold}] Missing files. Skipping.")
        continue

    print(f"\n[{fold}]")
    print(f"  Train spots: {np.load(train_emb_path).shape[0]}, using top {int(TOP_K_FRAC*100)}%"
          f" = {int(TOP_K_FRAC * np.load(train_emb_path).shape[0])} spots")

    train_embs_gpu  = F.normalize(
        torch.tensor(np.load(train_emb_path), dtype=torch.float32, device=device), dim=-1)
    train_exprs_gpu = torch.tensor(
        np.load(train_exp_path), dtype=torch.float32, device=device)

    preds_fold = []
    bulks_fold = []

    for sid in tqdm(matched_sids, desc=f"  {fold}", leave=False):
        emb_path    = os.path.join(fold_emb_dir, f"{sid}.npy")
        coords_path = os.path.join(fold_emb_dir, f"{sid}_coords.npy")

        if not os.path.exists(emb_path):
            continue

        embs   = np.load(emb_path)
        coords = np.load(coords_path)

        tile_embs_gpu = F.normalize(
            torch.tensor(embs, dtype=torch.float32, device=device), dim=-1)
        df_tiles = coords_to_grid(coords)

        slide_pred_2000, _ = sliding_window_predict(
            tile_embs_gpu, df_tiles, train_embs_gpu, train_exprs_gpu)

        if slide_pred_2000 is None:
            continue

        slide_pred = slide_pred_2000[common_in_hvg_idx]

        preds_fold.append(slide_pred)
        bulks_fold.append(slide_bulks_dict[sid])
        ensemble_preds[sid].append(slide_pred)

        del tile_embs_gpu

    del train_embs_gpu, train_exprs_gpu
    torch.cuda.empty_cache()

    pred_arr = np.array(preds_fold)
    bulk_arr = np.array(bulks_fold)

    all_g, all_s = calc_pcc(pred_arr, bulk_arr, list(range(len(common_genes))), "All genes")
    a_g,   a_s   = calc_pcc(pred_arr, bulk_arr, idx_A, "Method A top300")
    b_g,   b_s   = calc_pcc(pred_arr, bulk_arr, idx_B, "Method B top300")

    fold_results[fold] = {
        "pred_arr": pred_arr, "bulk_arr": bulk_arr,
        "gene_pcc_all": all_g, "slide_pcc_all": all_s,
        "gene_pcc_A": a_g, "slide_pcc_A": a_s,
        "gene_pcc_B": b_g, "slide_pcc_B": b_s,
    }

    pd.DataFrame({"slide_id": matched_sids[:len(all_s)], "slide_pcc": all_s}).to_csv(
        os.path.join(OUTPUT_DIR, f"slide_pcc_{fold}_all.csv"), index=False)

# ─── 저장 ─────────────────────────────────────────────────────────────────────
ens_sids  = [sid for sid in matched_sids if sid in ensemble_preds]
ens_preds = np.array([np.mean(ensemble_preds[sid], axis=0) for sid in ens_sids])
ens_bulks = np.array([slide_bulks_dict[sid] for sid in ens_sids])

np.save(os.path.join(OUTPUT_DIR, "ensemble_preds.npy"), ens_preds)
np.save(os.path.join(OUTPUT_DIR, "ensemble_bulks.npy"), ens_bulks)
np.save(os.path.join(OUTPUT_DIR, "common_genes.npy"),   np.array(common_genes))
np.save(os.path.join(OUTPUT_DIR, "slide_ids.npy"),      np.array(ens_sids))

summary_rows = []
for fold, res in fold_results.items():
    summary_rows.append({
        "fold":           fold,
        "top_k_frac":     TOP_K_FRAC,
        "gene_mean_all":  res["gene_pcc_all"].mean(),
        "slide_mean_all": res["slide_pcc_all"].mean(),
        "gene_mean_A":    res["gene_pcc_A"].mean(),
        "slide_mean_A":   res["slide_pcc_A"].mean(),
        "gene_mean_B":    res["gene_pcc_B"].mean(),
        "slide_mean_B":   res["slide_pcc_B"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "fold_summary.csv"), index=False)
print(f"\nFold summary:")
print(summary_df.to_string(index=False))
print(f"\nAll results saved to {OUTPUT_DIR}")
print("Done!")