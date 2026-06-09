"""
TCGA Bulk Prediction - Fine-tuned Ensemble + Top 300 genes
===========================================================
fold별 val_exprs에서 top 300 expressed genes 추출 → 해당 fold 예측에 적용
10 fold 예측 평균 → ensemble → bulk PCC 계산

ST 평가와 동일한 방식으로 top 300 선택 (일관성 유지)
"""

import os
import glob
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
import torch
import torch.nn.functional as F
import open_clip
from scipy.stats import pearsonr
from tqdm import tqdm

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
PATCH_DIR    = "/project_antwerp/TCGA-HNSC/TCGA_patch"
REF_FILE     = "/project_antwerp/hbae/ref_file.csv"
FINETUNE_DIR = "/project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_"
FINETUNE_EMB = "/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding"
GENE_LIST    = "/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt"
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/finetuned_ensemble_top300"

WINDOW_SIZE = 10
STRIDE      = 5
MIN_TILES   = 50
FOLDS       = ["fold_01"] #[f"fold_{i:02d}" for i in range(1, 11)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Gene list + ref_file ─────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]
print(f"Gene list: {len(gene_list)} genes")

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
rna_cols  = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes = [c.replace("rna_", "") for c in rna_cols]

patch_dirs = {p.name: p for p in Path(PATCH_DIR).iterdir() if p.is_dir()}
matched = [
    (row["slide_id"], patch_dirs[row["slide_id"]], row)
    for _, row in ref_df.iterrows()
    if row["slide_id"] in patch_dirs
]
print(f"Matched slides: {len(matched)}")

# ─── Helper: checkpoint 탐색 ─────────────────────────────────────────────────
def find_checkpoint(fold_name):
    pattern = os.path.join(
        FINETUNE_DIR, fold_name, "finetune_hvg_*", "checkpoints", "epoch_latest.pt"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint: {pattern}")
    return matches[0]

# ─── Helper: 모델 로드 ────────────────────────────────────────────────────────
def load_model(ckpt_path):
    model, _, preprocess = open_clip.create_model_and_transforms("coca_ViT-L-14")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model, preprocess

# ─── Helper: tile 이미지 → embedding ─────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(pil_images, model, preprocess, batch_size=256):
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch   = pil_images[i:i+batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        embs    = model.encode_image(tensors)
        embs    = F.normalize(embs, dim=-1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)

# ─── Helper: HDF5 로드 ────────────────────────────────────────────────────────
def load_tiles(hdf5_path):
    records, pil_images = [], []
    with h5py.File(hdf5_path, "r") as f:
        for idx, k in enumerate(f.keys()):
            pil_images.append(Image.fromarray(f[k][:]))
            y_px, x_px = map(int, k.split("_"))
            records.append({"tile_idx": idx, "y_px": y_px, "x_px": x_px})
    df = pd.DataFrame(records)
    x_vals    = sorted(df["x_px"].unique())
    tile_step = int(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 512
    df["x_grid"] = ((df["x_px"] - df["x_px"].min()) / tile_step).astype(int)
    df["y_grid"] = ((df["y_px"] - df["y_px"].min()) / tile_step).astype(int)
    return pil_images, df

# ─── Helper: PredEx ───────────────────────────────────────────────────────────
@torch.no_grad()
def predex(emb, train_embs_gpu, train_exprs_gpu):
    emb     = emb.unsqueeze(0)
    sim     = emb @ train_embs_gpu.T
    sim     = torch.clamp(sim, min=0)
    weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
    pred    = weights @ train_exprs_gpu
    return pred.squeeze(0).cpu().numpy()

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
            pred     = predex(mean_emb, train_embs_gpu, train_exprs_gpu)
            for idx in idxs:
                tile_preds[idx].append(pred)
            n_windows += 1
    if n_windows == 0:
        return None, 0
    all_preds = [np.mean(tile_preds[idx], axis=0) for idx in sorted(tile_preds.keys())]
    return np.mean(all_preds, axis=0), n_windows  # (2000,)

# ─── 메인: fold별 예측 → ensemble ────────────────────────────────────────────
print("\nRunning fold ensemble predictions...")

# {slide_id: list of (300,) predictions}
slide_fold_preds = defaultdict(list)
slide_bulks_dict = {}

for fold in FOLDS:
    print(f"\n{'='*50}")
    print(f"  Fold: {fold}")
    print(f"{'='*50}")

    # ── 1. Checkpoint 로드 ──
    try:
        ckpt_path = find_checkpoint(fold)
        print(f"  Checkpoint: {ckpt_path}")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        continue

    fold_model, fold_preprocess = load_model(ckpt_path)

    # ── 2. ST train embeddings 로드 ──
    emb_dir = os.path.join(FINETUNE_EMB, fold)
    try:
        train_img_embs = np.load(os.path.join(emb_dir, "train_img_embs.npy"))
        train_exprs    = np.load(os.path.join(emb_dir, "train_exprs.npy"))
        val_exprs      = np.load(os.path.join(emb_dir, "val_exprs.npy"))
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        del fold_model
        torch.cuda.empty_cache()
        continue

    train_embs_gpu  = F.normalize(
        torch.tensor(train_img_embs, dtype=torch.float32, device=device), dim=-1
    )
    train_exprs_gpu = torch.tensor(train_exprs, dtype=torch.float32, device=device)

    # ── 3. 이 fold의 top 300 gene index 추출 ──
    # val_exprs 기준 평균 발현량 top 300 (ST 평가와 동일한 방식)
    mean_expr  = val_exprs.mean(axis=0)           # (2000,)
    top300_idx = np.argsort(mean_expr)[::-1][:300] # top 300 HVG index
    top300_genes = [gene_list[i] for i in top300_idx]

    # bulk에서 top300 gene에 해당하는 컬럼
    top300_in_bulk   = [g for g in top300_genes if g in ref_genes]
    bulk_top300_cols = ["rna_" + g for g in top300_in_bulk]
    pred_top300_idx  = [gene_list.index(g) for g in top300_in_bulk]

    print(f"  Train embs: {train_img_embs.shape}")
    print(f"  Top 300 genes in bulk: {len(top300_in_bulk)} / 300")

    # ── 4. 슬라이드별 예측 ──
    for sid, patch_path, row in tqdm(matched, desc=f"  {fold}"):
        hdf5_files = list(patch_path.glob("*.hdf5"))
        if not hdf5_files:
            continue
        try:
            pil_images, df_tiles = load_tiles(hdf5_files[0])
        except Exception as e:
            tqdm.write(f"    [SKIP] {sid}: {e}")
            continue
        if not pil_images:
            continue

        tile_embs = extract_embeddings(pil_images, fold_model, fold_preprocess)
        slide_pred, _ = sliding_window_predict(
            tile_embs, df_tiles, train_embs_gpu, train_exprs_gpu
        )
        if slide_pred is None:
            continue

        # top 300만 추출
        pred_top300 = slide_pred[pred_top300_idx]             # (300,)
        slide_fold_preds[sid].append(pred_top300)

        # bulk 정답 (첫 번째 fold에서만 저장)
        if sid not in slide_bulks_dict:
            slide_bulks_dict[sid] = row[bulk_top300_cols].values.astype(float)

    del fold_model, train_embs_gpu, train_exprs_gpu
    torch.cuda.empty_cache()
    print(f"  Done.")

# ─── Ensemble 평균 ────────────────────────────────────────────────────────────
print("\nComputing ensemble...")
slide_ids   = []
slide_preds = []
slide_bulks = []

for sid, preds_list in slide_fold_preds.items():
    if not preds_list:
        continue
    slide_ids.append(sid)
    slide_preds.append(np.mean(preds_list, axis=0))
    slide_bulks.append(slide_bulks_dict[sid])

pred_arr = np.array(slide_preds)
bulk_arr = np.array(slide_bulks)
print(f"  Total slides: {len(slide_ids)}")
print(f"  Avg folds per slide: {np.mean([len(v) for v in slide_fold_preds.values()]):.1f}")

# ─── 저장 ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "slide_preds_ensemble_top300.npy"), pred_arr)
np.save(os.path.join(OUTPUT_DIR, "slide_bulks_top300.npy"),          bulk_arr)
np.save(os.path.join(OUTPUT_DIR, "slide_ids.npy"),                   np.array(slide_ids))

# ─── PCC 계산 ─────────────────────────────────────────────────────────────────
print("\nComputing PCC (top 300 genes)...")

# Gene-wise
gene_pccs, valid_genes = [], []
for i in range(pred_arr.shape[1]):
    p, b = pred_arr[:, i], bulk_arr[:, i]
    if p.std() < 1e-8 or b.std() < 1e-8:
        continue
    r, _ = pearsonr(p, b)
    gene_pccs.append(r)
    valid_genes.append(i)

gene_pccs = np.array(gene_pccs)

# Slide-wise
slide_pccs = []
for i in range(len(slide_ids)):
    p, b = pred_arr[i], bulk_arr[i]
    if p.std() < 1e-8 or b.std() < 1e-8:
        slide_pccs.append(np.nan)
        continue
    r, _ = pearsonr(p, b)
    slide_pccs.append(r)

slide_pccs = np.array(slide_pccs)
valid      = ~np.isnan(slide_pccs)

print(f"\n[Fine-tuned Ensemble | Top 300 genes]")
print(f"  Slides:      {len(slide_ids)}")
print(f"  Gene-wise  | mean={gene_pccs.mean():.4f}  median={np.median(gene_pccs):.4f}  "
      f"PCC>0.3: {(gene_pccs>0.3).sum()}  PCC>0.5: {(gene_pccs>0.5).sum()}")
print(f"  Slide-wise | mean={slide_pccs[valid].mean():.4f}  "
      f"median={np.median(slide_pccs[valid]):.4f}")

# ─── CSV 저장 ─────────────────────────────────────────────────────────────────
pd.DataFrame({
    "slide_id": slide_ids, "slide_pcc": slide_pccs
}).to_csv(os.path.join(OUTPUT_DIR, "slide_pcc_ensemble_top300.csv"), index=False)

print(f"\nSaved to {OUTPUT_DIR}")
print("Done!")