"""
TCGA Bulk RNA-seq Prediction - Fine-tuned Loki Ensemble (10 folds) + Sliding Window
=====================================================================================

Zero-shot과 다른 점:
  - 10개 fold의 fine-tuned checkpoint를 각각 로드
  - 각 fold마다 TCGA 슬라이드 예측
  - 10개 fold 예측값 평균 → 최종 예측 (ensemble)
  - Sequoia 논문 방식과 동일

흐름:
  for each fold (1~10):
      1. fine-tuned image encoder 로드
      2. fold의 ST train embeddings 로드 (fine-tuned)
      3. TCGA 슬라이드마다:
         a. tile 이미지 → fine-tuned embedding
         b. sliding window PredEx → slide 예측
  10개 fold 예측 평균 → bulk PCC 계산
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
PATCH_DIR      = "/project_antwerp/TCGA-HNSC/TCGA_patch"
REF_FILE       = "/project_antwerp/hbae/ref_file.csv"
FINETUNE_DIR   = "/project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_"
FINETUNE_EMB   = "/project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding"
GENE_LIST      = "/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt"
LOKI_CKPT      = "/project_antwerp/assets/loki_ckpts/checkpoint.pt"
OUTPUT_DIR     = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/finetuned_ensemble"

# Sliding window 파라미터
WINDOW_SIZE = 10
STRIDE      = 5
MIN_TILES   = 50

FOLDS = ["fold_01"]  # 먼저 1개만 #[f"fold_{i:02d}" for i in range(1, 11)]  # fold_01 ~ fold_10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── GPU 설정 ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── Step 1: Gene list ────────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]
print(f"\n[Step 1] Gene list: {len(gene_list)} genes")

# ─── Step 2: ref_file 로드 및 매칭 ───────────────────────────────────────────
print("\n[Step 2] Loading ref_file.csv...")
ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])

rna_cols     = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes    = [c.replace("rna_", "") for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]

pred_gene_idx  = [gene_list.index(g) for g in common_genes]
bulk_gene_cols = ["rna_" + g for g in common_genes]

print(f"  Slides in ref:  {len(ref_df)}")
print(f"  Common genes:   {len(common_genes)}")

patch_dirs = {p.name: p for p in Path(PATCH_DIR).iterdir() if p.is_dir()}
matched = [
    (row["slide_id"], patch_dirs[row["slide_id"]], row)
    for _, row in ref_df.iterrows()
    if row["slide_id"] in patch_dirs
]
print(f"  Matched slides: {len(matched)} / {len(ref_df)}")

# ─── Helper: checkpoint 경로 자동 탐색 ───────────────────────────────────────
def find_checkpoint(fold_name):
    """
    fold_01/finetune_hvg_fold_01_***/checkpoints/epoch_latest.pt
    날짜 부분이 fold마다 다를 수 있으므로 glob으로 탐색
    """
    pattern = os.path.join(
        FINETUNE_DIR, fold_name,
        "finetune_hvg_*", "checkpoints", "epoch_latest.pt"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found: {pattern}")
    return matches[0]

# ─── Helper: fine-tuned 모델 로드 ────────────────────────────────────────────
def load_finetuned_model(ckpt_path):
    """
    fine-tuned checkpoint로 Loki image encoder 로드
    zero-shot과 동일한 구조, weights만 다름
    """
    model, _, preprocess = open_clip.create_model_and_transforms("coca_ViT-L-14")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # checkpoint 키 구조에 따라 state_dict 추출
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
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
    return torch.cat(all_embs, dim=0)  # (T, 768)

# ─── Helper: HDF5 로드 ────────────────────────────────────────────────────────
def load_tiles_from_hdf5(hdf5_path):
    records, pil_images = [], []
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        for idx, k in enumerate(keys):
            pil_images.append(Image.fromarray(f[k][:]))
            y_px, x_px = map(int, k.split("_"))
            records.append({"tile_idx": idx, "y_px": y_px, "x_px": x_px})

    df = pd.DataFrame(records)
    if len(df) > 1:
        x_vals = sorted(df["x_px"].unique())
        tile_step = int(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 512
    else:
        tile_step = 512

    df["x_grid"] = ((df["x_px"] - df["x_px"].min()) / tile_step).astype(int)
    df["y_grid"] = ((df["y_px"] - df["y_px"].min()) / tile_step).astype(int)
    return pil_images, df

# ─── Helper: PredEx (GPU) ─────────────────────────────────────────────────────
@torch.no_grad()
def predex(emb, train_embs_gpu, train_exprs_gpu):
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    sim     = emb @ train_embs_gpu.T
    sim     = torch.clamp(sim, min=0)
    weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
    pred    = weights @ train_exprs_gpu
    return pred.squeeze(0).cpu().numpy()  # (2000,)

# ─── Helper: Sliding Window ───────────────────────────────────────────────────
def sliding_window_predict(tile_embs_gpu, df_tiles, train_embs_gpu, train_exprs_gpu,
                            window_size=10, stride=5, min_tiles=50):
    max_x = df_tiles["x_grid"].max()
    max_y = df_tiles["y_grid"].max()
    tile_preds = defaultdict(list)
    n_windows  = 0

    for x in range(0, max_x + 1, stride):
        for y in range(0, max_y + 1, stride):
            mask = (
                (df_tiles["x_grid"] >= x) & (df_tiles["x_grid"] < x + window_size) &
                (df_tiles["y_grid"] >= y) & (df_tiles["y_grid"] < y + window_size)
            )
            window_df = df_tiles[mask]
            if len(window_df) < min_tiles:
                continue

            idxs     = window_df["tile_idx"].values
            mean_emb = tile_embs_gpu[idxs].mean(dim=0)
            mean_emb = F.normalize(mean_emb, dim=-1)
            pred     = predex(mean_emb, train_embs_gpu, train_exprs_gpu)

            for idx in idxs:
                tile_preds[idx].append(pred)
            n_windows += 1

    if n_windows == 0:
        return None, 0

    all_tile_preds = [
        np.mean(tile_preds[idx], axis=0)
        for idx in sorted(tile_preds.keys())
    ]
    return np.mean(all_tile_preds, axis=0), n_windows  # (2000,)

# ─── Step 3: Fold별 예측 → Ensemble ──────────────────────────────────────────
print("\n[Step 3] Running fold ensemble predictions...")

# 슬라이드별로 fold 예측값을 누적할 dict
# {slide_id: [fold1_pred, fold2_pred, ...]}
slide_fold_preds = defaultdict(list)
slide_bulks_dict = {}

for fold in FOLDS:
    print(f"\n{'='*50}")
    print(f"  Fold: {fold}")
    print(f"{'='*50}")

    # ── 1. Fine-tuned checkpoint 로드 ──
    try:
        ckpt_path = find_checkpoint(fold)
        print(f"  Checkpoint: {ckpt_path}")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        continue

    fold_model, fold_preprocess = load_finetuned_model(ckpt_path)
    print(f"  Model loaded.")

    # ── 2. Fine-tuned ST train embeddings 로드 ──
    emb_dir = os.path.join(FINETUNE_EMB, fold)
    try:
        train_img_embs = np.load(os.path.join(emb_dir, "train_img_embs.npy"))
        train_exprs    = np.load(os.path.join(emb_dir, "train_exprs.npy"))
    except FileNotFoundError:
        print(f"  [SKIP] Embeddings not found: {emb_dir}")
        del fold_model
        torch.cuda.empty_cache()
        continue

    train_embs_gpu  = F.normalize(
        torch.tensor(train_img_embs, dtype=torch.float32, device=device), dim=-1
    )
    train_exprs_gpu = torch.tensor(train_exprs, dtype=torch.float32, device=device)
    print(f"  Train embs: {train_img_embs.shape}, exprs: {train_exprs.shape}")

    # ── 3. 슬라이드별 예측 ──
    for sid, patch_path, row in tqdm(matched, desc=f"  {fold}"):
        hdf5_files = list(patch_path.glob("*.hdf5"))
        if not hdf5_files:
            continue

        try:
            pil_images, df_tiles = load_tiles_from_hdf5(hdf5_files[0])
        except Exception as e:
            tqdm.write(f"    [SKIP] {sid}: {e}")
            continue

        if not pil_images:
            continue

        # tile → embedding (이 fold의 fine-tuned 모델)
        tile_embs = extract_embeddings(
            pil_images, fold_model, fold_preprocess, batch_size=256
        )

        # sliding window PredEx
        slide_pred, n_windows = sliding_window_predict(
            tile_embs, df_tiles, train_embs_gpu, train_exprs_gpu,
            window_size=WINDOW_SIZE, stride=STRIDE, min_tiles=MIN_TILES
        )

        if slide_pred is None:
            continue

        # fold 예측값 누적
        slide_fold_preds[sid].append(slide_pred[pred_gene_idx])

        # bulk 정답은 한 번만 저장
        if sid not in slide_bulks_dict:
            slide_bulks_dict[sid] = row[bulk_gene_cols].values.astype(float)

    # ── 4. 메모리 해제 ──
    del fold_model, train_embs_gpu, train_exprs_gpu
    torch.cuda.empty_cache()
    print(f"  Done. GPU memory freed.")

# ─── Step 4: 10 fold 평균 → 최종 예측 ───────────────────────────────────────
print("\n[Step 4] Averaging fold predictions (ensemble)...")

slide_ids   = []
slide_preds = []
slide_bulks = []

for sid, fold_preds_list in slide_fold_preds.items():
    if len(fold_preds_list) == 0:
        continue
    # 참여한 fold 수 출력
    ensemble_pred = np.mean(fold_preds_list, axis=0)  # (n_common_genes,)

    slide_ids.append(sid)
    slide_preds.append(ensemble_pred)
    slide_bulks.append(slide_bulks_dict[sid])

print(f"  Total slides with predictions: {len(slide_ids)}")
print(f"  Folds used per slide (avg): {np.mean([len(v) for v in slide_fold_preds.values()]):.1f}")

pred_arr = np.array(slide_preds)   # (S, n_common_genes)
bulk_arr = np.array(slide_bulks)   # (S, n_common_genes)

# ─── Step 5: 저장 ─────────────────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "slide_preds_ensemble.npy"), pred_arr)
np.save(os.path.join(OUTPUT_DIR, "slide_bulks.npy"),          bulk_arr)
np.save(os.path.join(OUTPUT_DIR, "common_genes.npy"),         np.array(common_genes))
np.save(os.path.join(OUTPUT_DIR, "slide_ids.npy"),            np.array(slide_ids))
print(f"  Saved to {OUTPUT_DIR}")

# ─── Step 6: PCC 계산 ─────────────────────────────────────────────────────────
print("\n[Step 6] Computing PCC...")

# Gene-wise PCC
gene_pccs, valid_genes = [], []
for i, gene in enumerate(common_genes):
    p, b = pred_arr[:, i], bulk_arr[:, i]
    if p.std() < 1e-8 or b.std() < 1e-8:
        continue
    r, _ = pearsonr(p, b)
    gene_pccs.append(r)
    valid_genes.append(gene)

gene_pccs = np.array(gene_pccs)

# Slide-wise PCC
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

print(f"\n[Fine-tuned Ensemble Result]")
print(f"  Slides:    {len(slide_ids)}")
print(f"  Gene-wise  | mean={gene_pccs.mean():.4f}  median={np.median(gene_pccs):.4f}  "
      f"PCC>0.3: {(gene_pccs>0.3).sum()}  PCC>0.5: {(gene_pccs>0.5).sum()}")
print(f"  Slide-wise | mean={slide_pccs[valid].mean():.4f}  median={np.median(slide_pccs[valid]):.4f}")

# ─── Step 7: 결과 CSV 저장 ────────────────────────────────────────────────────
pd.DataFrame({
    "slide_id":  slide_ids,
    "slide_pcc": slide_pccs
}).to_csv(os.path.join(OUTPUT_DIR, "slide_pcc_ensemble.csv"), index=False)

pd.DataFrame({
    "gene": valid_genes,
    "pcc":  gene_pccs
}).sort_values("pcc", ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, "gene_pcc_ensemble.csv"), index=False
)

print(f"\nAll results saved to {OUTPUT_DIR}")
print("Done!")