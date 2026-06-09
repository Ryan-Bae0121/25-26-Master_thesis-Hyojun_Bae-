"""
Step 2: 저장된 Embedding으로 PredEx + Sliding Window + Bulk PCC
fold별 결과 + ensemble 결과 모두 계산
버그 수정: pred (2000,) → common_genes (1968,) 인덱스 변환 추가
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
TCGA_EMB_DIR  = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings"
FINETUNE_EMB  = "/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding"
ZEROSHOT_EMB  = "/project_antwerp/hbae/Loki_output/0228_embeddings_zeroshot/fold_01"
REF_FILE      = "/project_antwerp/hbae/ref_file.csv"
GENE_LIST     = "/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt"
HVG300_PATH   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tcga_bulk_hvg300_from_common.npy"
OUTPUT_DIR    = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/finetuned_ensemble_final_v2"

WINDOW_SIZE = 10
STRIDE      = 5
MIN_TILES   = 50
FOLDS       = [f"fold_{i:02d}" for i in range(1, 11)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Gene list + ref_file ─────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]  # HVG 2000개

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
rna_cols     = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes    = [c.replace("rna_", "") for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]  # 1968개
bulk_cols    = ["rna_" + g for g in common_genes]

# ★ 핵심 버그 수정: HVG 2000 → common_genes 1968 변환 인덱스
common_in_hvg_idx = [gene_list.index(g) for g in common_genes]  # (1968,)

print(f"Gene list: {len(gene_list)}")
print(f"Common genes: {len(common_genes)}")
print(f"common_in_hvg_idx length: {len(common_in_hvg_idx)}")

# ─── Top 300 Gene 선택 ────────────────────────────────────────────────────────
# 방법 A: ST val_exprs 기준 (common_genes 내 index)
val_exprs = np.load(os.path.join(ZEROSHOT_EMB, "val_exprs.npy"))
top300_A  = set(gene_list[i] for i in np.argsort(val_exprs.mean(axis=0))[::-1][:300])
idx_A     = [i for i, g in enumerate(common_genes) if g in top300_A]

# 방법 B: TCGA bulk HVG (common_genes 내 index)
hvg300_B = set(np.load(HVG300_PATH, allow_pickle=True).tolist())
idx_B    = [i for i, g in enumerate(common_genes) if g in hvg300_B]

print(f"Method A top300 in common: {len(idx_A)}")
print(f"Method B top300 in common: {len(idx_B)}")

# ─── matched slides ───────────────────────────────────────────────────────────
fold_01_dir  = os.path.join(TCGA_EMB_DIR, "fold_01")
matched_sids = []
slide_bulks_dict = {}

for _, row in ref_df.iterrows():
    sid = row["slide_id"]
    if os.path.exists(os.path.join(fold_01_dir, f"{sid}.npy")):
        matched_sids.append(sid)
        slide_bulks_dict[sid] = row[bulk_cols].values.astype(float)  # (1968,)

print(f"Matched slides: {len(matched_sids)}")

# ─── Helper: 좌표 → 격자 ──────────────────────────────────────────────────────
def coords_to_grid(coords):
    y_px, x_px = coords[:, 0], coords[:, 1]
    x_vals    = np.unique(x_px)
    tile_step = int(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 512
    return pd.DataFrame({
        "tile_idx": np.arange(len(coords)),
        "x_grid":   ((x_px - x_px.min()) / tile_step).astype(int),
        "y_grid":   ((y_px - y_px.min()) / tile_step).astype(int)
    })

# ─── Helper: PredEx ───────────────────────────────────────────────────────────
@torch.no_grad()
def predex(emb, train_embs_gpu, train_exprs_gpu):
    emb     = emb.unsqueeze(0)
    sim     = emb @ train_embs_gpu.T
    sim     = torch.clamp(sim, min=0)
    weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
    pred    = weights @ train_exprs_gpu
    return pred.squeeze(0).cpu().numpy()  # (2000,)

# ─── Helper: Sliding Window ───────────────────────────────────────────────────
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
            pred     = predex(mean_emb, train_embs_gpu, train_exprs_gpu)  # (2000,)
            for idx in idxs:
                tile_preds[idx].append(pred)
            n_windows += 1

    if n_windows == 0:
        return None, 0

    all_preds = [np.mean(tile_preds[idx], axis=0) for idx in sorted(tile_preds.keys())]
    return np.mean(all_preds, axis=0), n_windows  # (2000,)

# ─── PCC 계산 함수 ────────────────────────────────────────────────────────────
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

# ─── 메인: fold별 예측 ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Fold-wise Results")
print("="*60)

# fold별 결과 저장
fold_results = {}
ensemble_preds = defaultdict(list)  # {sid: [fold1_pred, ...]} (1968,) 단위

for fold in FOLDS:
    fold_emb_dir  = os.path.join(TCGA_EMB_DIR, fold)
    train_emb_path = os.path.join(FINETUNE_EMB, fold, "train_img_embs.npy")
    train_exp_path = os.path.join(FINETUNE_EMB, fold, "train_exprs.npy")

    if not os.path.exists(fold_emb_dir) or not os.path.exists(train_emb_path):
        print(f"\n[{fold}] Missing files. Skipping.")
        continue

    print(f"\n[{fold}]")

    train_embs_gpu  = F.normalize(
        torch.tensor(np.load(train_emb_path), dtype=torch.float32, device=device), dim=-1)
    train_exprs_gpu = torch.tensor(
        np.load(train_exp_path), dtype=torch.float32, device=device)

    preds_fold = []  # (1968,) 단위
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

        # ★ 버그 수정: HVG 2000 → common_genes 1968 변환
        slide_pred_1968 = slide_pred_2000[common_in_hvg_idx]  # (1968,)

        preds_fold.append(slide_pred_1968)
        bulks_fold.append(slide_bulks_dict[sid])
        ensemble_preds[sid].append(slide_pred_1968)

        del tile_embs_gpu

    del train_embs_gpu, train_exprs_gpu
    torch.cuda.empty_cache()

    pred_arr = np.array(preds_fold)
    bulk_arr = np.array(bulks_fold)

    all_g, all_s = calc_pcc(pred_arr, bulk_arr, list(range(len(common_genes))), "All 1968")
    a_g,   a_s   = calc_pcc(pred_arr, bulk_arr, idx_A, "Method A top300")
    b_g,   b_s   = calc_pcc(pred_arr, bulk_arr, idx_B, "Method B top300")

    fold_results[fold] = {
        "pred_arr": pred_arr, "bulk_arr": bulk_arr,
        "gene_pcc_all": all_g, "slide_pcc_all": all_s,
        "gene_pcc_A": a_g, "slide_pcc_A": a_s,
        "gene_pcc_B": b_g, "slide_pcc_B": b_s,
    }

    # fold별 CSV 저장
    pd.DataFrame({"slide_id": matched_sids[:len(all_s)], "slide_pcc": all_s}).to_csv(
        os.path.join(OUTPUT_DIR, f"slide_pcc_{fold}_all.csv"), index=False)

# ─── Ensemble 평균 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Ensemble Result (10 fold average)")
print("="*60)

ens_sids  = [sid for sid in matched_sids if sid in ensemble_preds]
ens_preds = np.array([np.mean(ensemble_preds[sid], axis=0) for sid in ens_sids])
ens_bulks = np.array([slide_bulks_dict[sid] for sid in ens_sids])

print(f"Slides: {len(ens_sids)}, Avg folds: {np.mean([len(v) for v in ensemble_preds.values()]):.1f}")
print(f"pred shape: {ens_preds.shape}, bulk shape: {ens_bulks.shape}")

calc_pcc(ens_preds, ens_bulks, list(range(len(common_genes))), "Ensemble All 1968")
calc_pcc(ens_preds, ens_bulks, idx_A, "Ensemble Method A top300")
calc_pcc(ens_preds, ens_bulks, idx_B, "Ensemble Method B top300")

# ─── 저장 ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "ensemble_preds.npy"), ens_preds)
np.save(os.path.join(OUTPUT_DIR, "ensemble_bulks.npy"), ens_bulks)
np.save(os.path.join(OUTPUT_DIR, "common_genes.npy"),   np.array(common_genes))
np.save(os.path.join(OUTPUT_DIR, "slide_ids.npy"),      np.array(ens_sids))

pd.DataFrame({"slide_id": ens_sids,
              "slide_pcc_all": [pearsonr(ens_preds[i], ens_bulks[i])[0]
                                for i in range(len(ens_sids))]
             }).to_csv(os.path.join(OUTPUT_DIR, "slide_pcc_ensemble_all.csv"), index=False)

# fold별 요약 표
summary_rows = []
for fold, res in fold_results.items():
    summary_rows.append({
        "fold": fold,
        "gene_mean_all":   res["gene_pcc_all"].mean(),
        "slide_mean_all":  res["slide_pcc_all"].mean(),
        "gene_mean_A":     res["gene_pcc_A"].mean(),
        "slide_mean_A":    res["slide_pcc_A"].mean(),
        "gene_mean_B":     res["gene_pcc_B"].mean(),
        "slide_mean_B":    res["slide_pcc_B"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "fold_summary.csv"), index=False)
print(f"\nFold summary:")
print(summary_df.to_string(index=False))

print(f"\nAll results saved to {OUTPUT_DIR}")
print("Done!")
