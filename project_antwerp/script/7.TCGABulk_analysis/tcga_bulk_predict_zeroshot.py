"""
TCGA Bulk RNA-seq Prediction - Loki PredEx + Sequoia Sliding Window
====================================================================

Sequoia 방식을 Loki에 맞게 변형:
  Sequoia: 10x10 window → transformer 모델 → 예측값
  Loki:    10x10 window → tile embedding 평균 → ST similarity → 예측값

슬라이딩 윈도우 과정:
  1. HDF5 key("y_x")에서 tile 격자 좌표 파싱
  2. 10x10 window를 stride=1로 슬라이딩
  3. 각 window: 포함된 tile embedding 평균 → PredEx → 예측값
  4. 각 tile이 여러 window에 포함 → tile별 예측값 평균
  5. 모든 tile 평균 → slide 1개 예측 → bulk PCC 계산

단순 평균(global avg)과 sliding window 결과를 모두 저장해서 비교 가능
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
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/zeroshot"

# 슬라이딩 윈도우 파라미터 (Sequoia 논문 기본값)
WINDOW_SIZE = 10   # 10x10 tile window
STRIDE      = 1    # 한 tile씩 이동 (정밀도 최대, 계산 오래 걸림)
MIN_TILES   = 50   # window 내 최소 tile 수 (10x10의 절반)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── GPU 설정 ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── Step 1: Gene list 로드 ───────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]
print(f"\n[Step 1] Gene list: {len(gene_list)} genes")

# ─── Step 2: ST train data → GPU ─────────────────────────────────────────────
print("\n[Step 2] Loading ST train embeddings...")
train_img_embs = np.load(os.path.join(ZEROSHOT_DIR, "train_img_embs.npy"))
train_exprs    = np.load(os.path.join(ZEROSHOT_DIR, "train_exprs.npy"))
print(f"  train_img_embs: {train_img_embs.shape}")
print(f"  train_exprs:    {train_exprs.shape}")

train_embs_gpu  = F.normalize(
    torch.tensor(train_img_embs, dtype=torch.float32, device=device), dim=-1
)  # (64157, 768)
train_exprs_gpu = torch.tensor(train_exprs, dtype=torch.float32, device=device)
# (64157, 2000)

print(f"  GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ─── Step 3: Loki 모델 로드 ───────────────────────────────────────────────────
print("\n[Step 3] Loading Loki model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "coca_ViT-L-14",
    pretrained=LOKI_CKPT,
    weights_only=False
)
model = model.to(device)
model.eval()
print("  Loki model loaded.")

# ─── Step 4: ref_file 로드 및 매칭 ───────────────────────────────────────────
print("\n[Step 4] Loading ref_file.csv...")
ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])

rna_cols     = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes    = [c.replace("rna_", "") for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]

pred_gene_idx  = [gene_list.index(g) for g in common_genes]
bulk_gene_cols = ["rna_" + g for g in common_genes]

print(f"  Slides in ref: {len(ref_df)}")
print(f"  Common genes:  {len(common_genes)}")

patch_dirs = {p.name: p for p in Path(PATCH_DIR).iterdir() if p.is_dir()}
matched = [
    (row["slide_id"], patch_dirs[row["slide_id"]], row)
    for _, row in ref_df.iterrows()
    if row["slide_id"] in patch_dirs
]
print(f"  Matched slides: {len(matched)} / {len(ref_df)}")

# ─── Helper: tile 이미지 → embedding 추출 (GPU batch) ────────────────────────
@torch.no_grad()
def extract_embeddings(pil_images, batch_size=256):
    """
    PIL 이미지 리스트 → L2 정규화된 embedding tensor (GPU)
    Returns: (N, 768) on GPU
    """
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch   = pil_images[i:i+batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        embs    = model.encode_image(tensors)
        embs    = F.normalize(embs, dim=-1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)  # (N, 768)

# ─── Helper: PredEx - embedding → gene expression 예측 ───────────────────────
@torch.no_grad()
def predex(emb):
    """
    embedding 1개 (또는 평균 embedding) → gene expression 예측

    emb: (768,) or (1, 768), L2 정규화됨, GPU
    returns: (2000,) numpy
    """
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)  # (1, 768)

    # cosine similarity with all ST train spots
    sim = emb @ train_embs_gpu.T          # (1, 64157)
    sim = torch.clamp(sim, min=0)
    weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)  # (1, 64157)

    pred = weights @ train_exprs_gpu       # (1, 2000)
    return pred.squeeze(0).cpu().numpy()   # (2000,)

# ─── Helper: HDF5에서 tile 데이터 로드 ───────────────────────────────────────
def load_tiles_from_hdf5(hdf5_path):
    """
    HDF5 → tile 이미지 리스트 + 격자 좌표 DataFrame

    HDF5 key 형식: "y_x" (예: "21504_45568")
    격자 좌표: 픽셀 좌표를 tile 단위로 정규화 (0부터 시작하는 정수)

    Returns:
        pil_images: list of PIL.Image
        df: DataFrame with columns [tile_idx, y_px, x_px, y_grid, x_grid]
    """
    records = []
    pil_images = []

    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        for idx, k in enumerate(keys):
            img_arr = f[k][:]  # (256, 256, 3)
            pil_images.append(Image.fromarray(img_arr))

            # key = "y_x" 파싱
            y_px, x_px = map(int, k.split("_"))
            records.append({"tile_idx": idx, "y_px": y_px, "x_px": x_px})

    df = pd.DataFrame(records)

    # 격자 좌표 정규화 (Sequoia 방식)
    # 픽셀 좌표 → tile 단위 정수 인덱스 (0부터 시작)
    tile_step = 512  # 실제 tile step (HDF5 key 간격 확인 필요)
    # 자동으로 step 감지
    if len(df) > 1:
        x_vals = sorted(df["x_px"].unique())
        y_vals = sorted(df["y_px"].unique())
        if len(x_vals) > 1:
            tile_step = int(np.median(np.diff(x_vals)))

    df["x_grid"] = ((df["x_px"] - df["x_px"].min()) / tile_step).astype(int)
    df["y_grid"] = ((df["y_px"] - df["y_px"].min()) / tile_step).astype(int)

    return pil_images, df

# ─── Sliding Window 예측 (Sequoia 방식 → Loki 변형) ──────────────────────────
def sliding_window_predict(tile_embs_gpu, df, window_size=10, stride=1, min_tiles=50):
    """
    Sequoia sliding window를 Loki PredEx에 맞게 변형

    Sequoia:  window 100개 tile → transformer → 예측값
    Loki:     window tile들의 embedding 평균 → PredEx → 예측값

    각 tile이 여러 window에 포함 → tile별 예측값 리스트에 누적 → 평균

    Args:
        tile_embs_gpu: (T, 768) tensor, GPU, L2 정규화됨
        df: DataFrame with [tile_idx, x_grid, y_grid]
        window_size: 10 (10x10 window)
        stride: 1 (한 tile씩 이동)
        min_tiles: 50 (window 최소 tile 수)

    Returns:
        slide_pred: (2000,) numpy - 모든 tile 예측 평균
    """
    max_x = df["x_grid"].max()
    max_y = df["y_grid"].max()

    # tile별 예측값 누적 저장
    # key: tile_idx, value: list of (2000,) predictions
    tile_preds = defaultdict(list)

    n_windows = 0

    for x in range(0, max_x + 1, stride):
        for y in range(0, max_y + 1, stride):
            # 이 window에 포함되는 tile들
            mask = (
                (df["x_grid"] >= x) & (df["x_grid"] < x + window_size) &
                (df["y_grid"] >= y) & (df["y_grid"] < y + window_size)
            )
            window_df = df[mask]

            # tile이 너무 적으면 skip (Sequoia: 50개 이상)
            if len(window_df) < min_tiles:
                continue

            # window 내 tile들의 embedding 추출
            idxs = window_df["tile_idx"].values
            window_embs = tile_embs_gpu[idxs]  # (k, 768)

            # window 대표 embedding = tile embedding 평균
            # (Sequoia의 transformer 입력을 단순화한 버전)
            mean_emb = window_embs.mean(dim=0)         # (768,)
            mean_emb = F.normalize(mean_emb, dim=-1)   # L2 재정규화

            # PredEx: 이 window의 예측값
            pred = predex(mean_emb)  # (2000,)

            # 이 window의 모든 tile에 예측값 누적
            for idx in idxs:
                tile_preds[idx].append(pred)

            n_windows += 1

    if n_windows == 0:
        return None

    # 각 tile별 예측값 평균 (여러 window에서 받은 예측 평균)
    all_tile_preds = []
    for idx in sorted(tile_preds.keys()):
        tile_mean = np.mean(tile_preds[idx], axis=0)  # (2000,)
        all_tile_preds.append(tile_mean)

    # 슬라이드 레벨: 모든 tile 평균
    slide_pred = np.mean(all_tile_preds, axis=0)  # (2000,)
    return slide_pred, n_windows

# ─── Step 5: 단순 평균 예측 (비교용) ─────────────────────────────────────────
@torch.no_grad()
def global_avg_predict(tile_embs_gpu):
    """
    모든 tile embedding 평균 → PredEx → slide 예측
    (Sliding window와 비교하기 위한 baseline)
    """
    sim = tile_embs_gpu @ train_embs_gpu.T          # (T, 64157)
    sim = torch.clamp(sim, min=0)
    weights = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)
    tile_preds = weights @ train_exprs_gpu           # (T, 2000)
    return tile_preds.mean(dim=0).cpu().numpy()      # (2000,)

# ─── Step 6: 메인 루프 ────────────────────────────────────────────────────────
print("\n[Step 6] Running predictions...")

slide_ids      = []
slide_preds_sw = []   # sliding window 예측
slide_preds_ga = []   # global average 예측 (비교용)
slide_bulks    = []

for sid, patch_path, row in tqdm(matched, desc="Slides"):
    hdf5_files = list(patch_path.glob("*.hdf5"))
    if not hdf5_files:
        tqdm.write(f"  [SKIP] No hdf5: {sid}")
        continue

    # HDF5 로드: tile 이미지 + 격자 좌표
    try:
        pil_images, df_tiles = load_tiles_from_hdf5(hdf5_files[0])
    except Exception as e:
        tqdm.write(f"  [SKIP] HDF5 error {sid}: {e}")
        continue

    if len(pil_images) == 0:
        continue

    # 모든 tile → embedding (GPU)
    tile_embs = extract_embeddings(pil_images, batch_size=256)  # (T, 768)

    # ── 방법 1: Sliding Window (Sequoia 방식) ──
    result = sliding_window_predict(
        tile_embs, df_tiles,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        min_tiles=MIN_TILES
    )

    if result is None:
        tqdm.write(f"  [SKIP] No valid windows: {sid}")
        continue

    slide_pred_sw, n_windows = result

    # ── 방법 2: Global Average (비교용) ──
    slide_pred_ga = global_avg_predict(tile_embs)

    # bulk 정답
    bulk_vals = row[bulk_gene_cols].values.astype(float)

    slide_ids.append(sid)
    slide_preds_sw.append(slide_pred_sw[pred_gene_idx])
    slide_preds_ga.append(slide_pred_ga[pred_gene_idx])
    slide_bulks.append(bulk_vals)

    tqdm.write(f"  {sid}: {len(pil_images)} tiles, {n_windows} windows")

print(f"\nTotal predicted: {len(slide_ids)} slides")

# ─── Step 7: 저장 ─────────────────────────────────────────────────────────────
pred_sw = np.array(slide_preds_sw)  # (S, n_common_genes)
pred_ga = np.array(slide_preds_ga)
bulk_arr = np.array(slide_bulks)

np.save(os.path.join(OUTPUT_DIR, "slide_preds_sliding_window.npy"), pred_sw)
np.save(os.path.join(OUTPUT_DIR, "slide_preds_global_avg.npy"),     pred_ga)
np.save(os.path.join(OUTPUT_DIR, "slide_bulks.npy"),                bulk_arr)
np.save(os.path.join(OUTPUT_DIR, "common_genes.npy"),               np.array(common_genes))
np.save(os.path.join(OUTPUT_DIR, "slide_ids.npy"),                  np.array(slide_ids))

# ─── Step 8: PCC 계산 함수 ───────────────────────────────────────────────────
def compute_pccs(pred_arr, bulk_arr, common_genes, label=""):
    """gene-wise PCC + slide-wise PCC 계산 및 출력"""
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
    for i in range(len(pred_arr)):
        p, b = pred_arr[i], bulk_arr[i]
        if p.std() < 1e-8 or b.std() < 1e-8:
            slide_pccs.append(np.nan)
            continue
        r, _ = pearsonr(p, b)
        slide_pccs.append(r)
    slide_pccs = np.array(slide_pccs)
    valid = ~np.isnan(slide_pccs)

    print(f"\n[{label}]")
    print(f"  Gene-wise  | mean={gene_pccs.mean():.4f}  median={np.median(gene_pccs):.4f}  "
          f"PCC>0.3: {(gene_pccs>0.3).sum()}  PCC>0.5: {(gene_pccs>0.5).sum()}")
    print(f"  Slide-wise | mean={slide_pccs[valid].mean():.4f}  median={np.median(slide_pccs[valid]):.4f}")

    return valid_genes, gene_pccs, slide_pccs

# ─── Step 9: 결과 계산 및 저장 ───────────────────────────────────────────────
print("\n[Step 8] Computing PCC...")

valid_genes_sw, gene_pccs_sw, slide_pccs_sw = compute_pccs(
    pred_sw, bulk_arr, common_genes, label="Sliding Window"
)
valid_genes_ga, gene_pccs_ga, slide_pccs_ga = compute_pccs(
    pred_ga, bulk_arr, common_genes, label="Global Average"
)

# Sliding Window 결과 저장
pd.DataFrame({"slide_id": slide_ids, "slide_pcc": slide_pccs_sw}).to_csv(
    os.path.join(OUTPUT_DIR, "slide_pcc_sliding_window.csv"), index=False
)
pd.DataFrame({"gene": valid_genes_sw, "pcc": gene_pccs_sw}).sort_values(
    "pcc", ascending=False
).to_csv(os.path.join(OUTPUT_DIR, "gene_pcc_sliding_window.csv"), index=False)

# Global Average 결과 저장
pd.DataFrame({"slide_id": slide_ids, "slide_pcc": slide_pccs_ga}).to_csv(
    os.path.join(OUTPUT_DIR, "slide_pcc_global_avg.csv"), index=False
)
pd.DataFrame({"gene": valid_genes_ga, "pcc": gene_pccs_ga}).sort_values(
    "pcc", ascending=False
).to_csv(os.path.join(OUTPUT_DIR, "gene_pcc_global_avg.csv"), index=False)

print(f"\nAll results saved to {OUTPUT_DIR}")
print("\nDone!")