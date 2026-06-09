"""
Batch Effect UMAP Visualization for 36 HNSCC ST Samples (Fixed)
================================================================
patient_id 컬럼 형식: GSE181300_GSM5494475
  → dataset = GSE181300 (첫 번째 '_' 앞)
  → sample   = GSE181300_GSM5494475 (전체)
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

parser = argparse.ArgumentParser()
parser.add_argument("--h5ad",    default="/project_antwerp/hbae/merged_all_st_norm.h5ad")
parser.add_argument("--out_dir", default="/project_antwerp/hbae/batch_effect_qc")
parser.add_argument("--n_hvg",   type=int, default=2000)
parser.add_argument("--n_pcs",   type=int, default=50)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
sc.settings.verbosity = 1

# ── 데이터셋 순서 & 색상 ─────────────────────────────────────────────────────
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

# ── 1. 로드 ──────────────────────────────────────────────────────────────────
print(f"[1/6] Loading {args.h5ad} ...")
adata = sc.read_h5ad(args.h5ad)
print(f"      Shape: {adata.shape[0]:,} spots x {adata.shape[1]:,} genes")

# ── 2. dataset 컬럼 생성 ─────────────────────────────────────────────────────
print("[2/6] Parsing patient_id -> dataset ...")

def parse_dataset(pid):
    pid = str(pid)
    if "Queensland" in pid or "queensland" in pid.lower():
        return "Queensland"
    if "Zenodo" in pid or "zenodo" in pid.lower():
        return "Zenodo"
    part = pid.split("_")[0]
    if part in DATASET_ORDER:
        return part
    return "Unknown"

adata.obs["dataset"]   = adata.obs["patient_id"].apply(parse_dataset)
adata.obs["sample_id"] = adata.obs["patient_id"]

print(f"      Dataset distribution:")
print(adata.obs["dataset"].value_counts().to_string())
print(f"      Unique samples: {adata.obs['sample_id'].nunique()}")

# ── 3. QC metrics ────────────────────────────────────────────────────────────
print("[3/6] Computing QC metrics ...")
if "n_counts" not in adata.obs.columns:
    adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()
if "n_genes" not in adata.obs.columns:
    adata.obs["n_genes"]  = np.array((adata.X > 0).sum(axis=1)).flatten()

# ── 4. HVG -> PCA -> UMAP ────────────────────────────────────────────────────
print("[4/6] Running HVG -> PCA -> UMAP ...")
sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, batch_key="dataset")
print(f"      HVGs selected: {adata.var['highly_variable'].sum()}")

sc.tl.pca(adata, n_comps=args.n_pcs, use_highly_variable=True, svd_solver="arpack")
sc.pp.neighbors(adata, n_pcs=args.n_pcs)
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

umap_coords = adata.obsm["X_umap"]

# ── 5. Figure A: Dataset + Sample ────────────────────────────────────────────
print("[5/6] Plotting ...")
fig, axes = plt.subplots(1, 2, figsize=(22, 8))
fig.suptitle("Batch Effect Check — UMAP (before correction)", fontsize=15, fontweight="bold")

# Panel 1: Dataset별 색상
ax = axes[0]
for ds in DATASET_ORDER:
    mask = (adata.obs["dataset"] == ds).values
    if mask.sum() == 0:
        continue
    ax.scatter(
        umap_coords[mask, 0], umap_coords[mask, 1],
        c=DATASET_COLORS[ds], s=1.5, alpha=0.5,
        rasterized=True, label=f"{ds}  (n={mask.sum():,})"
    )
ax.set_title(f"Colored by Dataset  (total {adata.n_obs:,} spots)", fontsize=12)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.legend(markerscale=7, fontsize=9, loc="best", framealpha=0.8, title="Dataset")
ax.axis("off")

# Panel 2: Sample별 색상 (dataset 순서 유지)
ax = axes[1]
samples_sorted = []
for ds in DATASET_ORDER:
    samps = sorted(adata.obs.loc[adata.obs["dataset"] == ds, "sample_id"].unique())
    samples_sorted.extend(samps)

n_samples = len(samples_sorted)
cmap = plt.cm.get_cmap("tab20", n_samples)
for i, samp in enumerate(samples_sorted):
    mask = (adata.obs["sample_id"] == samp).values
    ax.scatter(
        umap_coords[mask, 0], umap_coords[mask, 1],
        c=[cmap(i)], s=1.5, alpha=0.6, rasterized=True
    )

# legend: dataset 블록으로 simplified
handles_ds = [mpatches.Patch(color=DATASET_COLORS[d], label=d)
              for d in DATASET_ORDER if d in adata.obs["dataset"].values]
ax.legend(handles=handles_ds, fontsize=9, loc="best",
          framealpha=0.8, title="Dataset (each shade = sample)")
ax.set_title(f"Colored by Sample  ({n_samples} samples)", fontsize=12)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.axis("off")

plt.tight_layout()
p = os.path.join(args.out_dir, "umap_batch_dataset_sample.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"      Saved: {p}")

# ── Figure B: QC metrics ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("QC Metrics on UMAP", fontsize=14, fontweight="bold")
for ax, metric, label in zip(
    axes,
    ["n_counts", "n_genes"],
    ["log10(n_counts + 1)", "# Expressed genes per spot"]
):
    vals = (np.log10(adata.obs[metric].values + 1)
            if "counts" in metric else adata.obs[metric].values)
    sc_p = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                      c=vals, cmap="viridis", s=1.5, alpha=0.5, rasterized=True)
    plt.colorbar(sc_p, ax=ax, fraction=0.03, pad=0.02, label=label)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.axis("off")

plt.tight_layout()
p = os.path.join(args.out_dir, "umap_qc_metrics.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"      Saved: {p}")

# ── Figure C: Spot count per dataset ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ds_stats = (adata.obs.groupby("dataset")
            .agg(n_spots=("sample_id", "count"),
                 n_samples=("sample_id", "nunique"))
            .reindex(DATASET_ORDER)
            .dropna())

bars = ax.bar(
    ds_stats.index,
    ds_stats["n_spots"],
    color=[DATASET_COLORS[d] for d in ds_stats.index],
    edgecolor="black", linewidth=0.7
)
for bar, (_, row) in zip(bars, ds_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{int(row['n_samples'])} samples\n{int(row['n_spots']):,} spots",
            ha="center", va="bottom", fontsize=8.5)

ax.set_ylabel("# Spots", fontsize=12)
ax.set_xlabel("Dataset", fontsize=12)
ax.set_title("Spot Count per Dataset", fontsize=13)
plt.tight_layout()
p = os.path.join(args.out_dir, "spot_count_per_dataset.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"      Saved: {p}")

# ── Figure D: Spot count per sample ──────────────────────────────────────────
per_sample = (adata.obs.groupby(["dataset", "sample_id"])
              .size().reset_index(name="n_spots"))
per_sample["dataset"] = pd.Categorical(
    per_sample["dataset"], categories=DATASET_ORDER, ordered=True)
per_sample = per_sample.sort_values(["dataset", "sample_id"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 5))
colors_bar = [DATASET_COLORS.get(str(d), "#999") for d in per_sample["dataset"]]
ax.bar(range(len(per_sample)), per_sample["n_spots"],
       color=colors_bar, edgecolor="black", lw=0.5)
ax.set_xticks(range(len(per_sample)))
ax.set_xticklabels(per_sample["sample_id"], rotation=90, fontsize=7)
ax.set_ylabel("# Spots", fontsize=12)
ax.set_title("Spot Count per Sample  (color = dataset)", fontsize=13)
handles = [mpatches.Patch(color=DATASET_COLORS[d], label=d)
           for d in DATASET_ORDER if d in per_sample["dataset"].values]
ax.legend(handles=handles, loc="upper right", fontsize=9)
plt.tight_layout()
p = os.path.join(args.out_dir, "spot_count_per_sample.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"      Saved: {p}")

# ── 6. 요약 ──────────────────────────────────────────────────────────────────
print("\n[6/6] Summary:")
print("=" * 60)
for ds in DATASET_ORDER:
    sub = adata.obs[adata.obs["dataset"] == ds]
    if len(sub) == 0:
        continue
    print(f"  {ds:<15}: {sub['sample_id'].nunique():2d} samples, {len(sub):6,} spots")
print("=" * 60)
print(f"\nAll figures saved to: {args.out_dir}")