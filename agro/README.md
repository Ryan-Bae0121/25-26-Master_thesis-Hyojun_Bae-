# HNSCC Spatial Transcriptomics — Thesis Code

Code for a thesis on head-and-neck squamous cell carcinoma (HNSCC) spatial
transcriptomics: WSI tiling, Visium/Open-ST preprocessing, contrastive
image–text training (LOKI / OmiCLIP), gene-expression prediction, and figures.

> This repository contains **code only**. Raw/processed data, model
> checkpoints, virtual environments, and credentials are intentionally
> excluded via `.gitignore` and must be obtained separately (see *Data*).

## Repository layout

```
.
├── prep_gse208253.py            # Visium → LOKI training-data preprocessing (entry point)
├── gse208253_pipeline/          # Preprocessing modules (QC, FOV crop, normalize, folds, ...)
├── build_loki_training_data.py  # Assemble contrastive training dataframe
├── train_contrastive.py         # Image–text contrastive training (skeleton)
├── save_embeddings.py           # Export embeddings
├── predict_fast.py              # Fast inference
├── download_loki_checkpoint.py  # Fetch LOKI checkpoint
├── download_omiclip.py          # Fetch OmiCLIP weights
│
├── scripts/                     # WSI tiling + ST figures + gene lists
│   ├── tcga_wsi_tile_144px.py        # TCGA-HNSC WSI → 144px tiles (tumor-mask filtered)
│   ├── export_slide_mask_comparison.py
│   ├── batch_export_st_figures.py
│   ├── export_st_figure_components.py
│   ├── export_appendix_gene_lists.py
│   ├── create_hvg_gene_list.py
│   └── um_per_pixel_hires.py
│
├── data/scripts/                # WSI visualization / patch extraction
│   ├── tcga_wsi_tile_visualization.py
│   ├── tcga_wsi_viz_v3_fixed_slides.py
│   └── extract_patches_mpp_unified.py
│
├── Loki/                        # LOKI/OmiCLIP-based analysis (built on the LOKI library)
│   ├── run_10fold_loki_predex.py     # 10-fold PredEx gene prediction
│   ├── slide_level_metrics.py / slide_level_calibrated_metrics.py
│   ├── analyze_*_gene_*.py           # gene-prediction quality analyses
│   ├── evaluate_omiclip_performance.py
│   ├── retrieval_ood_diagnostics.py
│   ├── hest_hnscc_subset/            # HEST HNSCC subset prep
│   ├── zero_shot_pipeline/           # zero-shot retrieval pipeline
│   └── visium_gse181300_qc/          # GSE181300 Visium QC
│
├── make_figure2_1_*.py          # Figure 2.1 (TCGA WSI + tumor mask, Visium ST)
├── export_figure2_1_components.py
├── openst_comparison_violin_from_csv.py
│
├── analyze_gene_prediction_quality.py
├── export_results_to_csv.py
├── filter_meta_exclude_samples.py
├── gpulab_csv.py
│
└── docs/
    └── GSE208253_PIPELINE.md     # Detailed GSE208253 preprocessing docs
```

## WSI tissue/tumor masking

TCGA-HNSC tiling uses **precomputed tumor-region masks** as the primary
tile-selection mask. These are rasterizations of pathologist GeoJSON
annotations (two non-overlapping sources: Kather lab — *"Tumor Region"*; and
Prof. Koen — *"Invasive front"* / *"Highest percentage"*), stored at the
coarsest OpenSlide pyramid level. When no annotation mask exists, the pipeline
falls back to an **Otsu-threshold tissue mask** computed from a grayscale
thumbnail. Tiles are kept when mask coverage ≥ 0.30 (plus a near-white
rejection filter). See `scripts/tcga_wsi_tile_144px.py`.

## Data (not included)

Obtain externally and point scripts at your local paths:

- **TCGA-HNSC** WSIs + tumor-annotation GeoJSON masks
- **GSE208253** (OSCC Visium)
- **GSE252265** (HNSCC Visium)
- **GSE251926** (Open-ST primary HNSCC)
- LOKI / OmiCLIP checkpoints (`download_*.py`)

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `scanpy`, `anndata`, `h5py`, `openslide-python`, `Pillow`,
`scikit-image`, `scikit-learn`, `pandas`, `numpy`. The LOKI analysis scripts
additionally require the `loki`/OmiCLIP packages and PyTorch.
