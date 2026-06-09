"""
Harmony Batch Effect Correction — Before/After UMAP
====================================================
1. Before correction UMAP (PCA space)
2. Harmony correction 실행
3. After correction UMAP (Harmony space)
4. Before/After 4-panel 비교 figure 저장

사용법:
  python run_harmony_correction.py \
    --h5ad /project_antwerp/hbae/merged_all_st_norm.h5ad \
    --out_dir /project_antwerp/hbae/batch_effect_qc
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc

# harmonypy 설치 확인
try:
    import harmonypy as hm
except ImportError:
    raise ImportError("harmonypy가 없습니다. 아래 명령어로 설치하세요:\n  pip install harmonypy")

# ── 인자 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--h5ad",    default="/project_antwerp/hbae/merged_all_st_norm.h5ad")
parser.add_argument("--out_dir", default="/project_antwerp/hbae/batch_effect_qc")
parser.add_argument("--n_hvg",   type=int, default=2000)
parser.add_argument("--n_pcs",   type=int, default=50)
parser.add_argument("--save_corrected", action="store_true",
                    help="Harmony corrected h5ad 저장 (추후 분석용)")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
sc.settings.verbosity = 1

# ── 설정 ─────────────────────────────────────────────────────────────────────
DATASET_ORDER = [
    "GSE181300", "GSE208253", "GSE220978",
    "GSE252265", "GSE281978", "Queensland", "Zenodo"
]
DATASET_COLORS = {
    "GSE181300":  "#4C72B0",
    "GSE208253":  "#DD8452",
    "GSE220978":  "#55A868",
    "GSE252265":  "#C44E52",
    "GSE281978":  "#8172B3",
    "Queensland": "#937860",
    "Zenodo":     "#DA8BC3",
}

def parse_dataset(pid):
    pid = str(pid)
    if "Queensland" in pid or "queensland" in pid.lower(): return "Queensland"
    if "Zenodo" in pid or "zenodo" in pid.lower():         return "Zenodo"
    part = pid.split("_")[0]
    return part if part in DATASET_ORDER else "Unknown"

def plot_umap(ax, coords, obs, color_by, title, dataset_colors, dot_size=1.5):
    """단일 UMAP 패널 그리기"""
    for ds in DATASET_ORDER:
        mask = (obs[color_by] == ds).values
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=dataset_colors[ds], s=dot_size, alpha=0.5,
            rasterized=True, label=f"{ds} ({mask.sum():,})"
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.axis("off")

# ── 1. 로드 ──────────────────────────────────────────────────────────────────
print("[1/6] Loading data ...")
adata = sc.read_h5ad(args.h5ad)
print(f"      {adata.shape[0]:,} spots x {adata.shape[1]:,} genes")

adata.obs["dataset"]   = adata.obs["patient_id"].apply(parse_dataset)
adata.obs["sample_id"] = adata.obs["patient_id"]

print("      Dataset distribution:")
for ds in DATASET_ORDER:
    n = (adata.obs["dataset"] == ds).sum()
    if n > 0:
        print(f"        {ds:<15}: {n:,} spots")

# ── 2. 전처리 → PCA ───────────────────────────────────────────────────────────
print(f"\n[2/6] HVG selection (n={args.n_hvg}) + PCA (n_pcs={args.n_pcs}) ...")
sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, batch_key="dataset")
sc.tl.pca(adata, n_comps=args.n_pcs, use_highly_variable=True, svd_solver="arpack")

# ── 3. BEFORE: UMAP (PCA space) ───────────────────────────────────────────────
print("[3/6] Computing BEFORE-correction UMAP ...")
sc.pp.neighbors(adata, n_pcs=args.n_pcs, key_added="neighbors_before")
sc.tl.umap(adata, neighbors_key="neighbors_before")
adata.obsm["X_umap_before"] = adata.obsm["X_umap"].copy()

# ── 4. Harmony correction ─────────────────────────────────────────────────────
print("[4/6] Running Harmony correction ...")
print("      (이 단계가 가장 오래 걸립니다 — 수 분 소요)")

pca_matrix = adata.obsm["X_pca"]   # shape: (n_spots, n_pcs)
meta_df    = adata.obs[["dataset"]].copy()

ho = hm.run_harmony(
    pca_matrix,
    meta_df,
    vars_use="dataset",
    max_iter_harmony=30,
    random_state=42,
    verbose=True
)
adata.obsm["X_pca_harmony"] = ho.Z_corr.T if ho.Z_corr.shape[1] == adata.n_obs else ho.Z_corr   # (n_spots, n_pcs)
print(f"      Harmony done. Corrected PCA shape: {adata.obsm['X_pca_harmony'].shape}")

# ── 5. AFTER: UMAP (Harmony space) ───────────────────────────────────────────
print("[5/6] Computing AFTER-correction UMAP ...")
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_pcs=args.n_pcs,
                key_added="neighbors_harmony")
sc.tl.umap(adata, neighbors_key="neighbors_harmony")
adata.obsm["X_umap_after"] = adata.obsm["X_umap"].copy()

# ── 6. Figure: 4-panel before/after ──────────────────────────────────────────
print("[6/6] Plotting before/after comparison ...")

fig, axes = plt.subplots(2, 2, figsize=(22, 16))
fig.suptitle(
    "Harmony Batch Effect Correction\n"
    f"36 HNSCC Samples | {adata.n_obs:,} spots | HVG={args.n_hvg} | PCs={args.n_pcs}",
    fontsize=14, fontweight="bold", y=1.01
)

before_coords = adata.obsm["X_umap_before"]
after_coords  = adata.obsm["X_umap_after"]

# Row 0: Dataset 색상
plot_umap(axes[0, 0], before_coords, adata.obs, "dataset",
          "BEFORE Harmony — colored by Dataset", DATASET_COLORS)
axes[0, 0].legend(markerscale=7, fontsize=8, loc="best",
                  framealpha=0.8, title="Dataset")

plot_umap(axes[0, 1], after_coords, adata.obs, "dataset",
          "AFTER Harmony — colored by Dataset", DATASET_COLORS)
axes[0, 1].legend(markerscale=7, fontsize=8, loc="best",
                  framealpha=0.8, title="Dataset")

# Row 1: Sample 색상 (dataset 순서 유지)
samples_sorted = []
for ds in DATASET_ORDER:
    samps = sorted(adata.obs.loc[adata.obs["dataset"] == ds, "sample_id"].unique())
    samples_sorted.extend(samps)
n_s = len(samples_sorted)
cmap = plt.cm.get_cmap("tab20", n_s)

for ax, coords, title in [
    (axes[1, 0], before_coords, "BEFORE Harmony — colored by Sample"),
    (axes[1, 1], after_coords,  "AFTER Harmony — colored by Sample"),
]:
    for i, samp in enumerate(samples_sorted):
        mask = (adata.obs["sample_id"] == samp).values
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], s=1.5, alpha=0.6, rasterized=True)
    handles_ds = [mpatches.Patch(color=DATASET_COLORS[d], label=d)
                  for d in DATASET_ORDER if d in adata.obs["dataset"].values]
    ax.legend(handles=handles_ds, fontsize=8, loc="best",
              framealpha=0.8, title="Dataset (each shade=sample)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.axis("off")

plt.tight_layout()
out_path = os.path.join(args.out_dir, "harmony_before_after_umap.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"      Saved: {out_path}")

# ── (선택) Corrected h5ad 저장 ─────────────────────────────────────────────
if args.save_corrected:
    save_path = os.path.join(args.out_dir, "merged_all_st_harmony.h5ad")
    print(f"\n      Saving corrected h5ad to: {save_path}")
    # UMAP/neighbor 결과 정리
    adata.obsm["X_umap"] = adata.obsm["X_umap_after"]
    adata.write_h5ad(save_path)
    print(f"      Saved: {save_path}")

print("\n✅ Done!")
print(f"   Output: {out_path}")