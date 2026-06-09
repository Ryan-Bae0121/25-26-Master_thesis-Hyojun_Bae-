"""
TCGA Bulk Prediction - Zero-shot + Top 300 genes
=================================================
ST 평가와 동일하게 val_exprs 기준 top 300 expressed genes만 PCC 계산
"""

import os
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
ZEROSHOT_DIR = "/project_antwerp/hbae/Loki_output/0228_embeddings_zeroshot/fold_01"
GENE_LIST    = "/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt"
LOKI_CKPT    = "/project_antwerp/assets/loki_ckpts/checkpoint.pt"
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/zeroshot_top300"

WINDOW_SIZE = 10
STRIDE      = 5
MIN_TILES   = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Gene list ────────────────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]
print(f"Gene list: {len(gene_list)} genes")

# ─── Top 300 gene index 추출 (val_exprs 기준) ─────────────────────────────────
# ST 평가와 동일한 방식: val set에서 평균 발현량 높은 top 300
val_exprs  = np.load(os.path.join(ZEROSHOT_DIR, "val_exprs.npy"))  # (N_val, 2000)
mean_expr  = val_exprs.mean(axis=0)                                 # (2000,)
top300_idx = np.argsort(mean_expr)[::-1][:300]                      # top 300 index
top300_genes = [gene_list[i] for i in top300_idx]
print(f"Top 300 genes selected from val set (mean expr range: "
      f"{mean_expr[top300_idx].min():.3f} ~ {mean_expr[top300_idx].max():.3f})")

# ─── ST train embeddings → GPU ───────────────────────────────────────────────
print("\nLoading ST train embeddings...")
train_img_embs = np.load(os.path.join(ZEROSHOT_DIR, "train_img_embs.npy"))
train_exprs    = np.load(os.path.join(ZEROSHOT_DIR, "train_exprs.npy"))

train_embs_gpu  = F.normalize(
    torch.tensor(train_img_embs, dtype=torch.float32, device=device), dim=-1
)
train_exprs_gpu = torch.tensor(train_exprs, dtype=torch.float32, device=device)
print(f"  train_img_embs: {train_img_embs.shape}")

# ─── Loki 모델 로드 ───────────────────────────────────────────────────────────
print("\nLoading Loki zero-shot model...")
model, _, preprocess = open_clip.create_model_and_transforms("coca_ViT-L-14")
ckpt = torch.load(LOKI_CKPT, map_location=device, weights_only=False)
sd   = ckpt.get("state_dict", ckpt.get("model", ckpt))
model.load_state_dict(sd, strict=False)
model = model.to(device).eval()
print("  Model loaded.")

# ─── ref_file 로드 ────────────────────────────────────────────────────────────
ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])

# bulk에서 top 300 gene에 해당하는 컬럼 추출
# top300_genes는 HVG 기준 → bulk rna_ 컬럼과 매칭
rna_cols   = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes  = [c.replace("rna_", "") for c in rna_cols]

# top300 중 bulk에 있는 gene만 사용
top300_in_bulk = [g for g in top300_genes if g in ref_genes]
bulk_top300_cols = ["rna_" + g for g in top300_in_bulk]
# HVG index에서 top300_in_bulk에 해당하는 index
pred_top300_idx = [gene_list.index(g) for g in top300_in_bulk]

print(f"\nTop 300 genes in bulk: {len(top300_in_bulk)} / 300")

patch_dirs = {p.name: p for p in Path(PATCH_DIR).iterdir() if p.is_dir()}
matched = [
    (row["slide_id"], patch_dirs[row["slide_id"]], row)
    for _, row in ref_df.iterrows()
    if row["slide_id"] in patch_dirs
]
print(f"Matched slides: {len(matched)}")

# ─── Helper 함수들 ────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(pil_images, batch_size=256):
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch   = pil_images[i:i+batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        embs    = model.encode_image(tensors)
        embs    = F.normalize(embs, dim=-1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)

@torch.no_grad()
def predex(emb):
    emb     = emb.unsqueeze(0)
    sim     = emb @ train_embs_gpu.T
    sim     = torch.clamp(sim, min=0)
    weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
    pred    = weights @ train_exprs_gpu
    return pred.squeeze(0).cpu().numpy()

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

def sliding_window_predict(tile_embs_gpu, df_tiles):
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
            pred     = predex(mean_emb)
            for idx in idxs:
                tile_preds[idx].append(pred)
            n_windows += 1
    if n_windows == 0:
        return None, 0
    all_preds = [np.mean(tile_preds[idx], axis=0) for idx in sorted(tile_preds.keys())]
    return np.mean(all_preds, axis=0), n_windows  # (2000,)

# ─── 메인 루프 ────────────────────────────────────────────────────────────────
print("\nRunning zero-shot predictions...")
slide_ids   = []
slide_preds = []  # top 300만
slide_bulks = []  # top 300만

for sid, patch_path, row in tqdm(matched, desc="Slides"):
    hdf5_files = list(patch_path.glob("*.hdf5"))
    if not hdf5_files:
        continue
    try:
        pil_images, df_tiles = load_tiles(hdf5_files[0])
    except Exception as e:
        tqdm.write(f"  [SKIP] {sid}: {e}")
        continue
    if not pil_images:
        continue

    tile_embs  = extract_embeddings(pil_images)
    slide_pred, n_windows = sliding_window_predict(tile_embs, df_tiles)
    if slide_pred is None:
        continue

    # top 300 gene만 추출
    pred_top300 = slide_pred[pred_top300_idx]            # (300,)
    bulk_top300 = row[bulk_top300_cols].values.astype(float)  # (300,)

    slide_ids.append(sid)
    slide_preds.append(pred_top300)
    slide_bulks.append(bulk_top300)

pred_arr = np.array(slide_preds)  # (S, 300)
bulk_arr = np.array(slide_bulks)  # (S, 300)

np.save(os.path.join(OUTPUT_DIR, "slide_preds_top300.npy"),  pred_arr)
np.save(os.path.join(OUTPUT_DIR, "slide_bulks_top300.npy"),  bulk_arr)
np.save(os.path.join(OUTPUT_DIR, "top300_genes.npy"),        np.array(top300_in_bulk))
np.save(os.path.join(OUTPUT_DIR, "slide_ids.npy"),           np.array(slide_ids))
print(f"\nTotal predicted: {len(slide_ids)} slides")

# ─── PCC 계산 ─────────────────────────────────────────────────────────────────
print("\nComputing PCC (top 300 genes)...")

# Gene-wise PCC
gene_pccs, valid_genes = [], []
for i, gene in enumerate(top300_in_bulk):
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

print(f"\n[Zero-shot | Top 300 genes]")
print(f"  Slides:      {len(slide_ids)}")
print(f"  Gene-wise  | mean={gene_pccs.mean():.4f}  median={np.median(gene_pccs):.4f}  "
      f"PCC>0.3: {(gene_pccs>0.3).sum()}  PCC>0.5: {(gene_pccs>0.5).sum()}")
print(f"  Slide-wise | mean={slide_pccs[valid].mean():.4f}  "
      f"median={np.median(slide_pccs[valid]):.4f}")

# ─── 저장 ─────────────────────────────────────────────────────────────────────
pd.DataFrame({
    "slide_id": slide_ids, "slide_pcc": slide_pccs
}).to_csv(os.path.join(OUTPUT_DIR, "slide_pcc_zeroshot_top300.csv"), index=False)

pd.DataFrame({
    "gene": valid_genes, "pcc": gene_pccs
}).sort_values("pcc", ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, "gene_pcc_zeroshot_top300.csv"), index=False
)

print(f"\nSaved to {OUTPUT_DIR}")
print("Done!")