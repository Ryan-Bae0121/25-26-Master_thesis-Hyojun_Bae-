"""
Open-ST HNSCC External Validation - Step 1: Patch Extraction (FOV Aggregation)
===============================================================================
각 패치의 GT expression을 패치 FOV(71µm) 안의 모든 cell expression 합산으로 계산.
→ Visium spot과 동일한 방식: spot 안의 여러 cell expression을 합산

핵심 개선:
  기존: 중심 cell 하나의 expression만 사용 → sparse (nonzero ratio ~0.40)
  개선: FOV 안 모든 cell expression 합산 → dense (Visium spot 방식과 동일)

패치 이미지:
  Pyramid TIFF Level 3 (2.761 µm/px), 26px crop → 224×224 BILINEAR resize
  Visium hires (3.41 µm/px), 21px crop → 224×224 resize 와 동일한 물리 FOV(71µm)

필터링:
  min_cells_per_fov: FOV 안 cell 수 부족한 패치 제거 (경계, sparse 영역)

Usage:
    python openst_step1_patch_extraction.py --min_cells 10
"""

import os
import argparse
import numpy as np
import scanpy as sc
import tifffile
import h5py
from scipy.ndimage import zoom
from sklearn.neighbors import KDTree
from PIL import Image
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--min_cells', type=int, default=10,
                    help='Minimum cells within FOV (default: 10)')
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────
H5AD_PATH = "/project_antwerp/hbae/data/Open_ST/GSM7990099_primary_HNSCC.h5ad"
TIF_PATH  = "/project_antwerp/hbae/data/Open_ST/GSM7990099_primary_HNSCC.tif"
MASK_NPY  = "/project_antwerp/hbae/data/Open_ST/GSM7990099_primary_HNSCC/mask.npy"
HVG_LIST  = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
OUT_H5    = f"/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc{args.min_cells}.h5"

# ── Resolution parameters ──────────────────────────────────────────────────
PYRAMID_LEVEL    = 3
UM_PER_PX_L0     = 0.345
UM_PER_PX_L3     = 2.761
PATCH_SIZE       = 224
FOV_UM           = 71.0
FOV_PX_L3        = round(FOV_UM / UM_PER_PX_L3)       # 26px
FOV_RADIUS_UM    = FOV_UM / 2                           # 35.5µm
FOV_RADIUS_PX_L0 = FOV_RADIUS_UM / UM_PER_PX_L0        # Level 0 기준
MIN_COUNTS       = 10   # QC: cell당 최소 raw counts

print(f"FOV aggregation parameters:")
print(f"  FOV diameter     : {FOV_UM} µm")
print(f"  FOV radius       : {FOV_RADIUS_UM} µm = {FOV_RADIUS_PX_L0:.1f} px (Level 0)")
print(f"  Min cells/FOV    : {args.min_cells}")
print(f"  Patch crop       : {FOV_PX_L3} px (Level 3) → {PATCH_SIZE}×{PATCH_SIZE}")

# ── 1. Load AnnData (raw counts) ──────────────────────────────────────────
print("\nLoading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
print(f"  Raw shape: {adata.shape}")

raw = adata.layers['raw']
if sp.issparse(raw):
    raw = raw.toarray()
else:
    raw = raw.copy()

# QC filter
cell_counts = raw.sum(axis=1)
keep_qc = cell_counts >= MIN_COUNTS
raw     = raw[keep_qc]
coords_all = adata.obsm['spatial'][keep_qc]   # Level 0 기준, (N, 2)
print(f"  After QC (>={MIN_COUNTS} counts): {raw.shape[0]} cells")

# HVG filter
with open(HVG_LIST) as f:
    hvg_genes = [line.strip() for line in f if line.strip()]
gene_names = list(adata.var_names)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}
common_genes = [g for g in hvg_genes if g in gene_to_idx]
hvg_col_idx  = [gene_to_idx[g] for g in common_genes]
raw_hvg = raw[:, hvg_col_idx]   # (N_cells, n_hvg), raw counts
print(f"  Common genes: {len(common_genes)}")

# ── 2. Build KDTree on all cells ──────────────────────────────────────────
print("\nBuilding KDTree ...")
tree = KDTree(coords_all)   # Level 0 좌표 기준

# ── 3. Load Level 3 image + mask ──────────────────────────────────────────
print("Loading Pyramid TIFF Level 3 ...")
with tifffile.TiffFile(TIF_PATH) as tif:
    img_l3   = tif.series[0].levels[PYRAMID_LEVEL].asarray()
    l0_shape = tif.series[0].levels[0].shape
H3, W3 = img_l3.shape[:2]
H0, W0 = l0_shape[:2]
SCALE_L3 = H3 / H0
print(f"  Level 3: {img_l3.shape}, scale: {SCALE_L3:.6f}")

mask_low   = np.load(MASK_NPY)
mask_l3    = zoom(mask_low, (H3/mask_low.shape[0], W3/mask_low.shape[1]), order=0).astype(np.uint8)
mask_3ch   = np.stack([mask_l3]*3, axis=-1)
img_masked = np.where(mask_3ch==1, img_l3, 255).astype(np.uint8)

# ── 4. Filter cells by tissue mask ────────────────────────────────────────
rows_l0 = coords_all[:, 0].astype(int)
cols_l0 = coords_all[:, 1].astype(int)
rows_l3_all = np.clip((rows_l0 * SCALE_L3).astype(int), 0, H3-1)
cols_l3_all = np.clip((cols_l0 * SCALE_L3).astype(int), 0, W3-1)
in_tissue = mask_l3[rows_l3_all, cols_l3_all].astype(bool)

coords_tissue = coords_all[in_tissue]
raw_tissue    = raw_hvg[in_tissue]
print(f"  After tissue mask: {coords_tissue.shape[0]} cells")

# ── 5. For each cell: aggregate FOV expression + check min_cells ──────────
print(f"\nAggregating FOV expression (radius={FOV_RADIUS_UM}µm) ...")

# KDTree는 tissue 내 cell만으로 재구축
tree_tissue = KDTree(coords_tissue)
neighbor_indices = tree_tissue.query_radius(coords_tissue, r=FOV_RADIUS_PX_L0)
cell_counts_per_fov = np.array([len(idx) for idx in neighbor_indices])

print(f"  Cells per FOV: mean={cell_counts_per_fov.mean():.1f}, "
      f"median={np.median(cell_counts_per_fov):.1f}")
for thresh in [5, 10, 15, 20]:
    print(f"  >= {thresh} cells: {(cell_counts_per_fov >= thresh).sum()} "
          f"({(cell_counts_per_fov >= thresh).mean()*100:.1f}%)")

# min_cells 필터
valid_mask = cell_counts_per_fov >= args.min_cells
print(f"\n  After min_cells>={args.min_cells} filter: {valid_mask.sum()} cells")

# ── 6. Border check (Level 3 기준) ────────────────────────────────────────
rows_l3_tissue = (coords_tissue[:, 0] * SCALE_L3).astype(int)
cols_l3_tissue = (coords_tissue[:, 1] * SCALE_L3).astype(int)
half = FOV_PX_L3 // 2

border_ok = (
    (rows_l3_tissue - half >= 0) &
    (rows_l3_tissue + half <= H3) &
    (cols_l3_tissue - half >= 0) &
    (cols_l3_tissue + half <= W3)
)
final_mask = valid_mask & border_ok
print(f"  After border check: {final_mask.sum()} valid patches")

valid_idx = np.where(final_mask)[0]
N = len(valid_idx)

# ── 7. Aggregate expression for valid cells ────────────────────────────────
print(f"\nAggregating expression for {N} valid patches ...")
expr_agg = np.zeros((N, len(common_genes)), dtype=np.float32)

for out_i, src_i in enumerate(valid_idx):
    # FOV 안의 cell indices
    fov_cell_idx = neighbor_indices[src_i]
    # raw counts 합산
    fov_expr = raw_tissue[fov_cell_idx].sum(axis=0)   # (n_hvg,)
    expr_agg[out_i] = fov_expr

    if (out_i + 1) % 5000 == 0:
        print(f"  Aggregated {out_i+1}/{N} ...")

# normalize_total + log1p (합산된 counts에 적용)
print("  Normalizing aggregated expression ...")
row_sums = expr_agg.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1   # 0 division 방지
expr_agg = expr_agg / row_sums * 1e4
expr_agg = np.log1p(expr_agg)

# nonzero ratio 확인
nonzero_ratio = (expr_agg > 0).mean()
print(f"  Nonzero ratio after aggregation: {nonzero_ratio:.3f}")
print(f"  (기존 single-cell: ~0.40, Visium: ~0.74)")

# ── 8. Save to HDF5 ───────────────────────────────────────────────────────
print(f"\nSaving to {OUT_H5} ...")
rows_valid_l3 = rows_l3_tissue[valid_idx]
cols_valid_l3 = cols_l3_tissue[valid_idx]

with h5py.File(OUT_H5, 'w') as f:
    ds_patches = f.create_dataset(
        'patches', shape=(N, PATCH_SIZE, PATCH_SIZE, 3),
        dtype=np.uint8, chunks=(1, PATCH_SIZE, PATCH_SIZE, 3)
    )
    f.create_dataset('expression',       data=expr_agg,        dtype=np.float32)
    f.create_dataset('coords_row',       data=rows_valid_l3)
    f.create_dataset('coords_col',       data=cols_valid_l3)
    f.create_dataset('cells_per_fov',    data=cell_counts_per_fov[valid_idx])
    f.attrs['n_cells']       = N
    f.attrs['n_genes']       = len(common_genes)
    f.attrs['patch_size']    = PATCH_SIZE
    f.attrs['fov_px']        = FOV_PX_L3
    f.attrs['pyramid_level'] = PYRAMID_LEVEL
    f.attrs['um_per_px']     = UM_PER_PX_L3
    f.attrs['min_cells']     = args.min_cells
    f.attrs['gene_names']    = np.array(common_genes, dtype='S')

    # 패치 추출: Level 3에서 26px crop → 224×224 BILINEAR resize
    for out_i, src_i in enumerate(valid_idx):
        r, c   = rows_l3_tissue[src_i], cols_l3_tissue[src_i]
        r0, r1 = r - half, r + half
        c0, c1 = c - half, c + half
        crop   = img_masked[r0:r1, c0:c1]
        patch  = np.array(
            Image.fromarray(crop).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
        )
        ds_patches[out_i] = patch
        if (out_i + 1) % 5000 == 0:
            print(f"  Written {out_i+1}/{N} patches ...")

print(f"\nDone! Saved {N} patches to {OUT_H5}")
print(f"  patches shape   : ({N}, {PATCH_SIZE}, {PATCH_SIZE}, 3)")
print(f"  expression shape: {expr_agg.shape}")
print(f"  nonzero ratio   : {(expr_agg > 0).mean():.3f}")
print(f"  cells_per_fov   : mean={cell_counts_per_fov[valid_idx].mean():.1f}")