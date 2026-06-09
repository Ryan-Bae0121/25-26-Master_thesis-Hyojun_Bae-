"""
Open-ST HNSCC External Validation - Step 1: Patch Extraction (Level 3)
=======================================================================
Visium 학습 파이프라인과 해상도를 맞추기 위해 Pyramid TIFF Level 3 사용.

해상도 매칭 근거:
  Visium hires   : 3.41  µm/px, FOV = 21px crop → 224×224 BILINEAR resize (71 µm)
  Open-ST Level 0: 0.345 µm/px  ← 기존 방식 (너무 선명, tile artifact 있음)
  Open-ST Level 3: 2.761 µm/px, FOV = 26px crop → 224×224 BILINEAR resize (71 µm) ✓

좌표계:
  obsm['spatial'] 은 Level 0 기준 pixel 좌표
  → Level 3 사용 시 scale factor (L3_H / L0_H = 1312/10501 = 0.1249) 적용 필요

Coordinate convention confirmed:
  coords[:, 0] = row (H axis)  [Level 0 기준]
  coords[:, 1] = col (W axis)  [Level 0 기준]
"""

import os
import numpy as np
import scanpy as sc
import tifffile
import h5py
from scipy.ndimage import zoom
from PIL import Image
import scipy.sparse as sp

# ── Paths ──────────────────────────────────────────────────────────────────
H5AD_PATH   = "/project_antwerp/hbae/data/Open_ST/GSM7990099_primary_HNSCC.h5ad"
TIF_PATH    = "/project_antwerp/hbae/data/Open_ST/GSM7990099_primary_HNSCC.tif"
MASK_NPY    = "/project_antwerp/hbae/data/Open_ST/GSM7990099_primary_HNSCC/mask.npy"
HVG_LIST    = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
OUT_H5      = "/project_antwerp/hbae/data/Open_ST/openst_patches_level3.h5"

# ── Resolution parameters ──────────────────────────────────────────────────
PYRAMID_LEVEL   = 3         # Level 3: 2.761 µm/px (Visium hires: 3.41 µm/px)
UM_PER_PX_L3    = 2.761     # µm/px at Level 3
FOV_UM          = 71.0      # 물리적 FOV 크기 (Visium과 동일)
FOV_PX_L3       = round(FOV_UM / UM_PER_PX_L3)  # = 26px
PATCH_SIZE      = 224       # 최종 패치 크기 (ViT 입력)
MIN_COUNTS      = 10        # QC: 최소 raw counts per cell

print(f"FOV at Level 3: {FOV_PX_L3} px × {UM_PER_PX_L3} µm/px = {FOV_PX_L3 * UM_PER_PX_L3:.1f} µm")
print(f"Visium FOV    : 21 px × 3.41 µm/px = 71.6 µm")

# ── 1. Load AnnData ────────────────────────────────────────────────────────
print("\nLoading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
print(f"  Raw shape: {adata.shape}")

raw = adata.layers['raw']
if sp.issparse(raw):
    raw = raw.toarray()
adata.X = raw.copy()

# ── 2. QC filter ───────────────────────────────────────────────────────────
cell_counts = adata.X.sum(axis=1)
adata = adata[cell_counts >= MIN_COUNTS].copy()
print(f"  After QC (>={MIN_COUNTS} counts): {adata.shape[0]} cells")

# ── 3. Filter to HVG 2000 genes ───────────────────────────────────────────
print("Filtering to HVG 2000 genes ...")
with open(HVG_LIST) as f:
    hvg_genes = [line.strip() for line in f if line.strip()]
print(f"  HVG list: {len(hvg_genes)} genes")

common_genes = [g for g in hvg_genes if g in adata.var_names]
print(f"  Common genes (Open-ST ∩ HVG): {len(common_genes)}")
adata = adata[:, common_genes].copy()

# ── 4. Normalize ───────────────────────────────────────────────────────────
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"  Normalized. Expression shape: {adata.X.shape}")

# ── 5. Load Pyramid TIFF Level 3 ──────────────────────────────────────────
print(f"\nLoading Pyramid TIFF Level {PYRAMID_LEVEL} ...")
with tifffile.TiffFile(TIF_PATH) as tif:
    img_l3 = tif.series[0].levels[PYRAMID_LEVEL].asarray()  # (H3, W3, 3)
H3, W3 = img_l3.shape[0], img_l3.shape[1]
print(f"  Level {PYRAMID_LEVEL} image shape: {img_l3.shape}")
print(f"  Resolution: {UM_PER_PX_L3} µm/px")

# Level 0 크기 (좌표 변환용)
with tifffile.TiffFile(TIF_PATH) as tif:
    l0_shape = tif.series[0].levels[0].shape
H0, W0 = l0_shape[0], l0_shape[1]
SCALE_L3 = H3 / H0   # Level 0 → Level 3 좌표 변환 비율
print(f"  Scale factor (L0→L3): {SCALE_L3:.6f} (= {H3}/{H0})")

# ── 6. Load mask, upsample to Level 3 size, apply to image ────────────────
print("Applying tissue mask ...")
mask_low  = np.load(MASK_NPY)   # (82, 130)
mask_l3   = zoom(mask_low, (H3 / mask_low.shape[0], W3 / mask_low.shape[1]), order=0).astype(np.uint8)
print(f"  Mask upsampled to Level 3 size: {mask_l3.shape}")

mask_3ch   = np.stack([mask_l3] * 3, axis=-1)
img_masked = np.where(mask_3ch == 1, img_l3, 255).astype(np.uint8)
print(f"  Masked image shape: {img_masked.shape}")

# ── 7. Convert coordinates from Level 0 → Level 3 ─────────────────────────
coords  = adata.obsm['spatial']          # Level 0 기준
rows_l0 = coords[:, 0]                  # row (H axis), Level 0
cols_l0 = coords[:, 1]                  # col (W axis), Level 0

rows_l3 = (rows_l0 * SCALE_L3).astype(int)   # Level 3 기준
cols_l3 = (cols_l0 * SCALE_L3).astype(int)

# ── 8. Filter cells by tissue mask (Level 3 좌표 기준) ─────────────────────
rows_c    = np.clip(rows_l3, 0, H3 - 1)
cols_c    = np.clip(cols_l3, 0, W3 - 1)
in_tissue = mask_l3[rows_c, cols_c].astype(bool)

adata    = adata[in_tissue].copy()
rows_l3  = rows_l3[in_tissue]
cols_l3  = cols_l3[in_tissue]
print(f"  After tissue mask filter: {adata.shape[0]} cells")

# ── 9. Find valid patches (border check) ──────────────────────────────────
half        = FOV_PX_L3 // 2
valid_idx   = []
skip_border = 0

print(f"\nExtracting {FOV_PX_L3}px crop → {PATCH_SIZE}×{PATCH_SIZE} resize ...")
for i in range(len(adata)):
    r, c = rows_l3[i], cols_l3[i]
    r0, r1 = r - half, r + half
    c0, c1 = c - half, c + half
    if r0 < 0 or r1 > H3 or c0 < 0 or c1 > W3:
        skip_border += 1
        continue
    valid_idx.append(i)

print(f"  Skipped (border): {skip_border}")
print(f"  Valid patches   : {len(valid_idx)} / {len(adata)}")

# ── 10. Save to HDF5 ───────────────────────────────────────────────────────
print(f"\nSaving to {OUT_H5} ...")
adata_valid = adata[valid_idx].copy()

if sp.issparse(adata_valid.X):
    expr_arr = adata_valid.X.toarray()
else:
    expr_arr = np.array(adata_valid.X)

rows_valid = rows_l3[valid_idx]
cols_valid = cols_l3[valid_idx]
N = len(valid_idx)

with h5py.File(OUT_H5, 'w') as f:
    ds_patches = f.create_dataset(
        'patches', shape=(N, PATCH_SIZE, PATCH_SIZE, 3),
        dtype=np.uint8, chunks=(1, PATCH_SIZE, PATCH_SIZE, 3)
    )
    f.create_dataset('expression',  data=expr_arr,   dtype=np.float32)
    f.create_dataset('coords_row',  data=rows_valid)
    f.create_dataset('coords_col',  data=cols_valid)
    f.attrs['n_cells']       = N
    f.attrs['n_genes']       = expr_arr.shape[1]
    f.attrs['patch_size']    = PATCH_SIZE
    f.attrs['fov_px']        = FOV_PX_L3
    f.attrs['pyramid_level'] = PYRAMID_LEVEL
    f.attrs['um_per_px']     = UM_PER_PX_L3
    f.attrs['gene_names']    = np.array(common_genes, dtype='S')

    # 패치 추출: FOV crop → BILINEAR resize (Visium과 동일 방식)
    for out_i, src_i in enumerate(valid_idx):
        r, c   = rows_l3[src_i], cols_l3[src_i]
        r0, r1 = r - half, r + half
        c0, c1 = c - half, c + half
        crop   = img_masked[r0:r1, c0:c1]                          # (26, 26, 3)
        patch  = np.array(
            Image.fromarray(crop).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
        )                                                            # (224, 224, 3)
        ds_patches[out_i] = patch
        if (out_i + 1) % 5000 == 0:
            print(f"  Written {out_i+1}/{N} patches ...")

print(f"\nDone! Saved {N} cells to {OUT_H5}")
print(f"  patches shape   : ({N}, {PATCH_SIZE}, {PATCH_SIZE}, 3)")
print(f"  expression shape: {expr_arr.shape}")
print(f"  FOV: {FOV_PX_L3}px @ {UM_PER_PX_L3}µm/px = {FOV_PX_L3*UM_PER_PX_L3:.1f}µm")