"""
Biological Validation of Tile Selection - 331 slides
1. Per-slide top-500 tile score vs bulk RNA-seq correlation (r values)
2. High/low scoring slide stratification + DEG analysis
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from tqdm import tqdm
import os

OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tile_selection_331"
REF_FILE     = "/project_antwerp/hbae/ref_file_331.csv"
GENE_LIST    = "/project_antwerp/hbae/data/0317_hvg_2000_list.txt"

# ─── 로드 ──────────────────────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
rna_cols     = [c for c in ref_df.columns if c.startswith("rna_")]
ref_genes    = [c.replace("rna_", "") for c in rna_cols]
common_genes = [g for g in gene_list if g in ref_genes]
bulk_cols    = ["rna_" + g for g in common_genes]

bulk_arr     = np.load(os.path.join(OUTPUT_DIR, "bulk_arr.npy"))
top500_preds = np.load(os.path.join(OUTPUT_DIR, "top500_preds.npy"))
summary_df   = pd.read_csv(os.path.join(OUTPUT_DIR, "slide_summary.csv"))

print(f"Slides: {len(bulk_arr)}, Genes: {len(common_genes)}")
print(f"Summary shape: {summary_df.shape}")

# ─── 1. Per-gene correlation: tile score vs bulk RNA-seq ───────────────────────
# per-slide mean top-500 tile PCC score
tile_scores = summary_df["top500_mean_pcc"].values  # (331,)

print("\n[1] Correlating tile scores with bulk RNA-seq gene expression...")
gene_corrs = []
for i, g in enumerate(tqdm(common_genes, desc="Gene correlation")):
    b = bulk_arr[:, i]
    if b.std() < 1e-8 or tile_scores.std() < 1e-8:
        gene_corrs.append({"gene": g, "r": np.nan, "p": np.nan})
        continue
    r, p = pearsonr(tile_scores, b)
    gene_corrs.append({"gene": g, "r": r, "p": p})

corr_df = pd.DataFrame(gene_corrs).sort_values("r", ascending=False)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "tile_score_gene_corr.csv"), index=False)

print("\n  Top-10 positive correlations:")
print(f"  {'Gene':>10} | {'r':>8}")
print(f"  {'-'*22}")
for _, row in corr_df.head(10).iterrows():
    print(f"  {row['gene']:>10} | {row['r']:>8.3f}")

print("\n  Top-5 negative correlations:")
for _, row in corr_df.tail(5).iterrows():
    print(f"  {row['gene']:>10} | {row['r']:>8.3f}")

# ─── 2. High/low scoring slide stratification ──────────────────────────────────
print("\n[2] High/low scoring slide stratification...")
n_slides  = len(tile_scores)
n_group   = int(n_slides * 0.25)  # top/bottom 25%
print(f"  Total slides: {n_slides}, n per group: {n_group}")

sorted_idx  = np.argsort(tile_scores)
high_idx    = sorted_idx[-n_group:]   # top 25%
low_idx     = sorted_idx[:n_group]    # bottom 25%

high_bulk   = bulk_arr[high_idx]      # (n_group, n_genes)
low_bulk    = bulk_arr[low_idx]       # (n_group, n_genes)

print(f"  High group (n={n_group}): "
      f"mean tile score = {tile_scores[high_idx].mean():.4f}")
print(f"  Low  group (n={n_group}): "
      f"mean tile score = {tile_scores[low_idx].mean():.4f}")

# ─── 3. DEG analysis ───────────────────────────────────────────────────────────
print("\n[3] Differential expression analysis (t-test)...")
deg_results = []
for i, g in enumerate(tqdm(common_genes, desc="DEG")):
    h = high_bulk[:, i]
    l = low_bulk[:, i]
    if h.std() < 1e-8 and l.std() < 1e-8:
        continue
    t, p = ttest_ind(h, l)
    mean_diff = h.mean() - l.mean()
    deg_results.append({
        "gene":      g,
        "mean_high": h.mean(),
        "mean_low":  l.mean(),
        "mean_diff": mean_diff,
        "t_stat":    t,
        "p_value":   p
    })

deg_df = pd.DataFrame(deg_results)
deg_df.to_csv(os.path.join(OUTPUT_DIR, "deg_results.csv"), index=False)

# p < 0.01, |mean_diff| > 0.5 기준
sig_up   = deg_df[(deg_df["p_value"] < 0.01) &
                   (deg_df["mean_diff"] > 0.5)].sort_values(
                       "mean_diff", ascending=False)
sig_down = deg_df[(deg_df["p_value"] < 0.01) &
                   (deg_df["mean_diff"] < -0.5)].sort_values(
                       "mean_diff", ascending=True)

print(f"\n  Upregulated in high-scoring slides:   {len(sig_up)}")
print(f"  Downregulated in high-scoring slides: {len(sig_down)}")

print("\n  Top-10 upregulated genes:")
print(f"  {'Gene':>10} | {'mean_diff':>10} | {'p_value':>12}")
print(f"  {'-'*38}")
for _, row in sig_up.head(10).iterrows():
    print(f"  {row['gene']:>10} | {row['mean_diff']:>10.3f} | "
          f"{row['p_value']:>12.2e}")

print("\n  Top-10 downregulated genes:")
for _, row in sig_down.head(10).iterrows():
    print(f"  {row['gene']:>10} | {row['mean_diff']:>10.3f} | "
          f"{row['p_value']:>12.2e}")

# ─── 최종 요약 ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Summary for thesis update")
print(f"{'='*55}")
print(f"  Slides: {n_slides}, Group size (25%): {n_group}")
print(f"\n  Tile score ~ bulk RNA-seq correlations:")
for gene in ["KRT6B", "KRT16", "SPRR1B", "S100A9", "S100A8", "VCAN"]:
    row = corr_df[corr_df["gene"] == gene]
    if len(row) > 0:
        print(f"    {gene:>8}: r = {row['r'].values[0]:>6.3f}")

print(f"\n  DEG (p<0.01, |diff|>0.5):")
print(f"    Upregulated:   {len(sig_up)}")
print(f"    Downregulated: {len(sig_down)}")

print(f"\nSaved to {OUTPUT_DIR}")
print("Done!")