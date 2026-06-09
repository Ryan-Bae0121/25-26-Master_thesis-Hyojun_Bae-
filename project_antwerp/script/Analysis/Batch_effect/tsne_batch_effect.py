"""
t-SNE batch effect analysis — UPDATED with full metadata
Plots:
  1. by cohort
  2. by tissue type (oral cavity / tongue / OSF-OSCC / HNSCC-PT / HNSCC-LNMT)
  3. by tumor site (Primary Tumor / Lymph Node Metastasis) — GSE281978 key distinction
  4. by country/institution
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
H5AD_PATH   = "/project_antwerp/hbae/data/0228_HVG_NEW/merged_all_st_norm.h5ad"
OUT_DIR     = "/project_antwerp/hbae/batch_effect_qc"
MAX_SPOTS   = 30000
RANDOM_SEED = 42
N_HVG       = 2000
PERPLEXITY  = 30
os.makedirs(OUT_DIR, exist_ok=True)

# ── COHORT COLORS ─────────────────────────────────────────────────────────────
COHORT_ORDER  = ["GSE181300","GSE208253","GSE220978",
                 "GSE252265","GSE281978","Queensland","Zenodo"]
COHORT_COLORS = {
    "GSE181300":"#4E79A7","GSE208253":"#F28E2B","GSE220978":"#E15759",
    "GSE252265":"#76B7B2","GSE281978":"#59A14F",
    "Queensland":"#EDC948","Zenodo":"#B07AA1",
}

# ── SAMPLE-LEVEL METADATA (from GEO) ─────────────────────────────────────────
# tissue_type: broad biological category
# tumor_site:  Primary Tumor vs Lymph Node Metastasis
# country:     sequencing institution country

SAMPLE_META = {
    # GSE181300 — HNSCC, Taiwan (Taipei Veterans General Hospital)
    # upn:1 = same patient, different tumor regions
    "GSE181300_GSM5494475": {"tissue_type": "HNSCC (Oral)",        "tumor_site": "Primary Tumor",        "country": "Taiwan"},
    "GSE181300_GSM5494476": {"tissue_type": "HNSCC (Oral)",        "tumor_site": "Primary Tumor",        "country": "Taiwan"},
    "GSE181300_GSM5494477": {"tissue_type": "HNSCC (Oral)",        "tumor_site": "Primary Tumor",        "country": "Taiwan"},
    "GSE181300_GSM5494478": {"tissue_type": "HNSCC (Oral)",        "tumor_site": "Primary Tumor",        "country": "Taiwan"},

    # GSE208253 — HPV-negative Oral SCC, Canada (Univ of Calgary)
    "GSE208253_GSM6339631_s1":  {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339632_s2":  {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339633_s3":  {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339634_s4":  {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339635_s5":  {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339637_s7":  {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339638_s8":  {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339640_s10": {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339641_s11": {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},
    "GSE208253_GSM6339642_s12": {"tissue_type": "Oral SCC (OSCC)",    "tumor_site": "Primary Tumor",        "country": "Canada"},

    # GSE220978 — OSF-associated OSCC, China (Central South Univ)
    "GSE220978_Patient1": {"tissue_type": "OSF-associated OSCC",  "tumor_site": "Primary Tumor",        "country": "China"},
    "GSE220978_Patient2": {"tissue_type": "OSF-associated OSCC",  "tumor_site": "Primary Tumor",        "country": "China"},
    "GSE220978_Patient3": {"tissue_type": "OSF-associated OSCC",  "tumor_site": "Primary Tumor",        "country": "China"},
    "GSE220978_Patient4": {"tissue_type": "OSF-associated OSCC",  "tumor_site": "Primary Tumor",        "country": "China"},

    # GSE252265 — Tongue SCC, Finland (Helsinki Univ Hospital)
    "GSE252265_GSM7998252": {"tissue_type": "Tongue SCC",          "tumor_site": "Primary Tumor",        "country": "Finland"},
    "GSE252265_GSM7998253": {"tissue_type": "Tongue SCC",          "tumor_site": "Primary Tumor",        "country": "Finland"},
    "GSE252265_GSM7998254": {"tissue_type": "Tongue SCC",          "tumor_site": "Primary Tumor",        "country": "Finland"},
    "GSE252265_GSM7998255": {"tissue_type": "Tongue SCC",          "tumor_site": "Primary Tumor",        "country": "Finland"},
    "GSE252265_GSM7998256": {"tissue_type": "Tongue SCC",          "tumor_site": "Primary Tumor",        "country": "Finland"},
    "GSE252265_GSM7998257": {"tissue_type": "Tongue SCC",          "tumor_site": "Primary Tumor",        "country": "Finland"},
    "GSE252265_GSM7998258": {"tissue_type": "Tongue SCC",          "tumor_site": "Primary Tumor",        "country": "Finland"},
    "GSE252265_GSM7998259": {"tissue_type": "Tongue SCC",          "tumor_site": "Primary Tumor",        "country": "Finland"},

    # GSE281978 — HPV+/HPV- HNSCC PT + LNMT, Korea (Pusan National Univ)
    "GSE281978_GSM8633891_21_00757_LI_SING": {"tissue_type": "HNSCC (HPV-)",  "tumor_site": "Primary Tumor",          "country": "Korea"},
    "GSE281978_GSM8633892_21_00758_LI_SING": {"tissue_type": "HNSCC (HPV-)",  "tumor_site": "Lymph Node Metastasis",  "country": "Korea"},
    "GSE281978_GSM8633893_21_01569_LI_SING": {"tissue_type": "HNSCC (HPV+)",  "tumor_site": "Primary Tumor",          "country": "Korea"},
    "GSE281978_GSM8633894_21_01570_LI_SING": {"tissue_type": "HNSCC (HPV+)",  "tumor_site": "Lymph Node Metastasis",  "country": "Korea"},
    "GSE281978_GSM8633895_21_01586_LI_SING": {"tissue_type": "HNSCC (HPV-)",  "tumor_site": "Primary Tumor",          "country": "Korea"},
    "GSE281978_GSM8633896_21_01587_LI_SING": {"tissue_type": "HNSCC (HPV-)",  "tumor_site": "Lymph Node Metastasis",  "country": "Korea"},

    # Queensland — Tonsillar cancer, HPV+, T4N2M0, Australia (Univ of Queensland)
    # 2 patients, Visium ST + CODEX spatial proteomics (multi-omics)
    "Queensland_Visium_S01":  {"tissue_type": "Tonsillar SCC (HPV+)",  "tumor_site": "Primary Tumor",  "country": "Australia"},
    "Queensland P5 Data_P5":  {"tissue_type": "Tonsillar SCC (HPV+)",  "tumor_site": "Primary Tumor",  "country": "Australia"},

    # Zenodo — Oropharyngeal SCC, p16+, recurrent/ICI-failure, France
    # 2 patients, recurrent OPSCC after Nivolumab/Pembrolizumab+Lenvatinib failure
    # MAR21: left palatine tonsil local recurrence (ST+SP), SEP21: re-biopsy same site (ST only)
    "Zenodo_17B5776": {"tissue_type": "Oropharyngeal SCC (OPSCC, recurrent)",  "tumor_site": "Primary Tumor",  "country": "France"},
    "Zenodo_19h1257": {"tissue_type": "Oropharyngeal SCC (OPSCC, recurrent)",  "tumor_site": "Primary Tumor",  "country": "France"},
}

# ── TISSUE TYPE COLORS ────────────────────────────────────────────────────────
TISSUE_ORDER = [
    "HNSCC (Oral)", "Oral SCC (OSCC)", "OSF-associated OSCC",
    "Tongue SCC", "HNSCC (HPV-)", "HNSCC (HPV+)",
    "Tonsillar SCC (HPV+)", "Oropharyngeal SCC (OPSCC, recurrent)"
]
TISSUE_COLORS = {
    "HNSCC (Oral)":                          "#4E79A7",
    "Oral SCC (OSCC)":                       "#F28E2B",
    "OSF-associated OSCC":                   "#E15759",
    "Tongue SCC":                            "#76B7B2",
    "HNSCC (HPV-)":                          "#59A14F",
    "HNSCC (HPV+)":                          "#B07AA1",
    "Tonsillar SCC (HPV+)":                  "#EDC948",
    "Oropharyngeal SCC (OPSCC, recurrent)":  "#FF9DA7",
}

TUMOR_SITE_COLORS = {
    "Primary Tumor":         "#E15759",
    "Lymph Node Metastasis": "#4E79A7",
}

COUNTRY_COLORS = {
    "Taiwan":    "#4E79A7",
    "Canada":    "#F28E2B",
    "China":     "#E15759",
    "Finland":   "#76B7B2",
    "Korea":     "#59A14F",
    "Australia": "#EDC948",
    "France":    "#FF9DA7",
    "Unknown":   "#BAB0AC",
}

# ── LOAD ──────────────────────────────────────────────────────────────────────
print(f"Loading {H5AD_PATH} ...")
adata = sc.read_h5ad(H5AD_PATH)
print(f"  Shape: {adata.shape}")

# ── ASSIGN METADATA ───────────────────────────────────────────────────────────
def extract_cohort(pid):
    pid = str(pid)
    if pid.startswith("GSE"):   return pid.split("_")[0]
    if "queensland" in pid.lower() or "P5 Data" in pid: return "Queensland"
    if "zenodo" in pid.lower(): return "Zenodo"
    return "Unknown"

pids = adata.obs['patient_id'].astype(str)
adata.obs['cohort']      = pids.apply(extract_cohort).astype('category')
adata.obs['tissue_type'] = pids.map(lambda p: SAMPLE_META.get(p, {}).get('tissue_type', 'Unknown'))
adata.obs['tumor_site']  = pids.map(lambda p: SAMPLE_META.get(p, {}).get('tumor_site',  'Unknown'))
adata.obs['country']     = pids.map(lambda p: SAMPLE_META.get(p, {}).get('country',     'Unknown'))

print("\nCohort distribution:")
print(adata.obs['cohort'].value_counts().to_string())
print("\nTissue type distribution:")
print(adata.obs['tissue_type'].value_counts().to_string())
print("\nTumor site distribution:")
print(adata.obs['tumor_site'].value_counts().to_string())

# Check unknown
unk = adata.obs[adata.obs['tissue_type'] == 'Unknown']['patient_id'].unique()
if len(unk) > 0:
    print(f"\nWARNING: {len(unk)} samples with Unknown tissue_type:")
    for u in unk: print(f"  {u}")

# ── SUBSAMPLE ─────────────────────────────────────────────────────────────────
if MAX_SPOTS and adata.n_obs > MAX_SPOTS:
    np.random.seed(RANDOM_SEED)
    idx = np.sort(np.random.choice(adata.n_obs, MAX_SPOTS, replace=False))
    adata_sub = adata[idx].copy()
    print(f"\nSubsampled to {adata_sub.n_obs} spots")
else:
    adata_sub = adata.copy()

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
print("\nPreprocessing ...")
X = adata_sub.X
max_val = float(X.max() if not sp.issparse(X) else X.max())
if max_val > 50:
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    print("  normalize_total + log1p applied")
else:
    print(f"  Already log-normalized (max={max_val:.2f})")

sc.pp.highly_variable_genes(adata_sub, n_top_genes=N_HVG, flavor='seurat')
adata_hvg = adata_sub[:, adata_sub.var['highly_variable']].copy()
print(f"  HVG shape: {adata_hvg.shape}")

sc.pp.scale(adata_hvg, max_value=10)
print("  PCA (50 PCs) ...")
sc.tl.pca(adata_hvg, n_comps=50, random_state=RANDOM_SEED)
print(f"  t-SNE (perplexity={PERPLEXITY}) ... [~10 min]")
sc.tl.tsne(adata_hvg, n_pcs=50, perplexity=PERPLEXITY, random_state=RANDOM_SEED)
print("  t-SNE done!")

tx = adata_hvg.obsm['X_tsne'][:, 0]
ty = adata_hvg.obsm['X_tsne'][:, 1]

# ── PLOT FUNCTION ─────────────────────────────────────────────────────────────
def scatter_plot(ax, tx, ty, labels, color_map, order, title, markerscale=4):
    for label in order:
        m = np.array(labels) == label
        if m.sum() == 0: continue
        ax.scatter(tx[m], ty[m], c=color_map.get(label, '#999999'),
                   s=2, alpha=0.5, label=f"{label} (n={m.sum():,})", rasterized=True)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=7, markerscale=markerscale,
              framealpha=0.85, title=None)
    ax.set_aspect('equal', adjustable='datalim')

# ── FIGURE 1: 2x2 overview ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("HNSCC ST Batch Effect Analysis (n=30,000 spots)", fontsize=14, fontweight='bold')

scatter_plot(axes[0,0], tx, ty,
             adata_hvg.obs['cohort'].values, COHORT_COLORS, COHORT_ORDER,
             "① Colored by Cohort")

scatter_plot(axes[0,1], tx, ty,
             adata_hvg.obs['tissue_type'].values, TISSUE_COLORS, TISSUE_ORDER,
             "② Colored by Tissue Type")

scatter_plot(axes[1,0], tx, ty,
             adata_hvg.obs['tumor_site'].values, TUMOR_SITE_COLORS,
             ["Primary Tumor", "Lymph Node Metastasis"],
             "③ Colored by Tumor Site\n(Primary vs Lymph Node Metastasis)")

scatter_plot(axes[1,1], tx, ty,
             adata_hvg.obs['country'].values, COUNTRY_COLORS,
             list(COUNTRY_COLORS.keys()),
             "④ Colored by Country/Institution")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "tsne_4panel_overview.png")
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out1}")
plt.close()

# ── FIGURE 2: cohort only (large) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
scatter_plot(ax, tx, ty,
             adata_hvg.obs['cohort'].values, COHORT_COLORS, COHORT_ORDER,
             "t-SNE — colored by cohort (batch effect)", markerscale=5)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "tsne_by_cohort.png")
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close()

# ── FIGURE 3: tissue type only (large) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
scatter_plot(ax, tx, ty,
             adata_hvg.obs['tissue_type'].values, TISSUE_COLORS, TISSUE_ORDER,
             "t-SNE — colored by tissue/cancer type", markerscale=5)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "tsne_by_tissue_type.png")
plt.savefig(out3, dpi=150, bbox_inches='tight')
print(f"Saved: {out3}")
plt.close()

# ── FIGURE 4: tumor site (large) ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
scatter_plot(ax, tx, ty,
             adata_hvg.obs['tumor_site'].values, TUMOR_SITE_COLORS,
             ["Primary Tumor", "Lymph Node Metastasis"],
             "t-SNE — Primary Tumor vs Lymph Node Metastasis\n(GSE281978 contains both)", markerscale=5)
plt.tight_layout()
out4 = os.path.join(OUT_DIR, "tsne_by_tumor_site.png")
plt.savefig(out4, dpi=150, bbox_inches='tight')
print(f"Saved: {out4}")
plt.close()

print(f"\nAll outputs in: {OUT_DIR}")
print("Done!")