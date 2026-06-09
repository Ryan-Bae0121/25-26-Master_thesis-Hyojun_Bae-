# Applying Histology Images to a Visual-Omics Foundation Model for Gene Expression Prediction

> **Master's Dissertation** · Ghent University · MSc Bioinformatics · 2025–2026  
> **Author:** Hyojun Bae · **Supervisors:** Taewoo Jung, Prof. Kathleen Marchal, Prof. Wesley De Neve  
> **Institute:** IDLab, imec, Ghent University

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.1-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-12.2-green?logo=nvidia" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey" />
  <img src="https://img.shields.io/badge/Ghent%20University-IDLab-blue" />
</p>

---

## Overview

This repository contains all code developed for my master's dissertation on spatially-resolved gene expression prediction from H&E histology images in **Head and Neck Squamous Cell Carcinoma (HNSCC)**.

The core idea: fine-tune **Loki** — a visual-omics foundation model built on OmiCLIP (ViT-L-14, CoCa architecture) — on multi-cohort **10x Visium spatial transcriptomics** data, then evaluate whether the learned image-omics alignment generalizes to unseen tissue sections and external datasets.

```
H&E Tile  ──►  Loki (fine-tuned)  ──►  k-NN retrieval (PredEx)  ──►  Gene expression prediction
                 ViT-L-14 encoder            over ST training spots
```

> **Code only.** Raw/processed data, model checkpoints, virtual environments, and credentials are excluded via `.gitignore` and must be obtained separately (see [Data & Checkpoints](#data--checkpoints)).

---

## Key Results

| Evaluation | Metric | Score |
|---|---|---|
| HNSCC Visium (10-fold CV, HEG top-300) | Gene-wise PCC | **0.207** |
| TCGA bulk RNA-seq (331 patients) — K=50 retrieval | Tile-wise PCC | **0.458** |
| TCGA bulk RNA-seq (331 patients) | Slide-wise PCC | **0.747–0.757** |
| Open-ST external validation (36,642 cells) | Cell-wise PCC | **~0.525** |
| Open-ST external validation | Gene-wise PCC | **~0.085** |

**Notable findings:**
- ECM/structural genes (COL1A1, FN1, SPARC) reach gene-wise PCC **0.65–0.81** — directly readable from H&E morphology
- Immune/IFN-response genes remain near zero — morphologically invisible in H&E
- Sparse retrieval (K=30% of training spots) mitigates PredEx variance collapse: **+10.1% gene-wise PCC**, 3.5× variance ratio improvement vs. all-spot retrieval
- TCGA biological validation: KRT6B r=0.773, KRT16 r=0.724, SPRR1B r=0.670 (n=82 per group; 286 up / 290 down DEGs)

---

## Repository Structure

Work was done across multiple servers. Each server's code lives in its own top-level folder to avoid path/import conflicts. Each folder is self-contained (its own scripts, modules, and README).

```
.
├── agro/                          # Code from agro.bioit.labnet server
│   ├── hbae/
│   │   └── script/0208_start/    # Main pipeline scripts
│   │       ├── preprocessing/    # ST data preprocessing, HVG selection, TF-IDF gene sentences
│   │       ├── finetuning/       # Loki contrastive fine-tuning (10-fold LOPO-CV)
│   │       ├── embedding/        # Embedding extraction (image & ST)
│   │       ├── prediction/       # PredEx k-NN gene expression prediction
│   │       ├── tcga/             # TCGA bulk RNA-seq validation pipeline
│   │       ├── openst/           # Open-ST external validation pipeline
│   │       └── figures/          # Figure generation scripts
│   └── README.md                 # Detailed per-script documentation
├── project_antwerp/               # Code from project_antwerp server (A100 GPU compute)
├── .gitignore
└── README.md
```

See the per-folder `README.md` for details, data sources, and usage.

### Adding code from another server

```bash
git clone <repo-url>
cd <repo>
mkdir <other-server-name>
# copy that server's project code into <other-server-name>/
git add <other-server-name>
git commit -m "Add <other-server-name> server code"
git push
```

Then on the other server, run `git pull` to sync.

---

## Dataset

| Cohort | Source | Samples | Spots |
|---|---|---|---|
| HNSCC Visium (training) | GSE181300, GSE208253, GSE220978, GSE252265, GSE281978, Zenodo, Queensland | 36 samples / 7 cohorts | ~71,432 |
| TCGA HNSC (external) | GDC Data Portal | 331 patient slides (primary) / 461 total | — |
| Open-ST (external) | GSM7990099 | 1 sample (primary HNSCC) | 36,642 cells |

**HVG list:** 2,000 highly variable genes selected via Scanpy across 36 Visium samples  
**Evaluation genes:** HEG top-300 within HVG 2,000 (following Loki protocol)

---

## Methods Summary

### Fine-tuning
- Base model: Loki (OmiCLIP ViT-L-14), pre-trained on pan-cancer spatial transcriptomics
- Strategy: **Leave-one-patient-out 10-fold cross-validation** on 36 HNSCC Visium samples
- Loss: Contrastive (image–gene sentence pairs), 10 epochs
- Gene sentences: TF-IDF weighted, expression-ranked gene symbol lists (IDF fitted per fold)

### Prediction (PredEx)
- k-NN retrieval over training spot embeddings using **tile-wise PCC** as similarity metric
- Optimal K = 50 spots (empirically validated; ~30% of training spots)
- Cross-modal similarity: cosine distance in shared Loki latent space

### External Validation
- **TCGA:** 144px crops at MPP=0.5 (≈71µm, matching Visium spot physical scale); ensemble prediction across fine-tuned folds
- **Open-ST:** Pyramid TIFF Level 3 (2.761 µm/px); FOV aggregation (71µm diameter) to match Visium multi-cell integration

---

## Environment

```
Python          3.11.8
PyTorch         2.2.1
CUDA            12.2
open-clip-torch 3.2.0
Scanpy          1.10.3
NumPy           1.26.4
clusterProfiler 4.18.4  # R, biological validation
```

```bash
# Install Python dependencies
pip install torch==2.2.1 open-clip-torch==3.2.0 scanpy==1.10.3 numpy==1.26.4

# For R-based DEG / pathway analysis
# see agro/README.md for bioconda setup
```

---

## Compute

All GPU experiments were run on **NVIDIA A100 80GB PCIe** (GPULab, Ghent University).  
Preprocessing and figure generation were performed on `agro.bioit.labnet`.

---

## Data & Checkpoints

Raw data and fine-tuned checkpoints are **not** included in this repository due to size and licensing constraints.

| Resource | Access |
|---|---|
| Visium ST data | GEO accession numbers listed above |
| TCGA slides | [GDC Data Portal](https://portal.gdc.cancer.gov/) (project: TCGA-HNSC) |
| Open-ST data | [GSM7990099](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7990099) |
| Loki base model | [Chen et al., Nature Methods 2025](https://doi.org/10.1038/s41592-025-02707-1) |
| Fine-tuned checkpoints | Available upon reasonable request (contact below) |

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{bae2026hnscc,
  author  = {Hyojun Bae},
  title   = {Applying Histology Images to a Visual-Omics Foundation Model
             for Gene Expression Prediction},
  school  = {Ghent University},
  year    = {2026},
  type    = {Master's Dissertation},
  address = {Ghent, Belgium}
}
```

Also cite the Loki foundation model:

```bibtex
@article{chen2025loki,
  author  = {Chen, Weixu and Zhao, Pengfei and Xu, Yang and Tang, Tong and
             Chen, Hang and Sathe, Vijay V. and Bhatt, Kirtee W. and
             Liu, Lei and Yang, Kang and Fan, Lei and Chu, Jiangping and
             Yu, Yinyin and Sun, Qingmin and Ma, Qingpeng and Chen, Kun and
             Lim, Ngak-Teng and Ahn, Jun-ichi and Chang, Shu-Hao and
             Lee, Sanghoon and Wang, Guangyu},
  title   = {Loki: a foundation model for visual omics},
  journal = {Nature Methods},
  volume  = {22},
  pages   = {1568--1582},
  year    = {2025},
  doi     = {10.1038/s41592-025-02707-1}
}
```

---

## Contact

**Hyojun Bae**  
MSc Bioinformatics, Ghent University · IDLab, imec  
📧 [hyojun.bae@ugent.be](mailto:hyojun.bae@ugent.be)
