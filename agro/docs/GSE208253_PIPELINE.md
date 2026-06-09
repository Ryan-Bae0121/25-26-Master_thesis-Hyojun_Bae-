# GSE208253 Preprocessing Pipeline

Complete preprocessing pipeline for GSE208253 (General OSCC) Visium spatial transcriptomics data, compatible with LOKI (OmiCLIP/PredEx) format specifications.

## Overview

This pipeline transforms raw Visium data into training-ready format for contrastive learning models. It performs:

- ✅ Quality control (resolution, gene detection filtering)
- ✅ FOV patch cropping (1.3× spot diameter, 224×224 resized)
- ✅ Scanpy-standard normalization (normalize_total → log1p)
- ✅ Highly variable gene selection (HVG)
- ✅ Gene sentence generation (top-50 expressed genes)
- ✅ Cross-validation fold creation (GroupKFold, patient-level)
- ✅ Training dataframe export with metadata

## Installation

### 1. Clone or download this repository

```bash
cd /home/students/hbae
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `scanpy >= 1.9.0` (for expression normalization)
- `anndata >= 0.8.0` (for data structures)
- `Pillow >= 9.0.0` (for image processing)
- `scikit-learn >= 1.0.0` (for cross-validation)
- `pandas`, `numpy`, `h5py`

### 3. Verify installation

```bash
python -c "import scanpy; import anndata; import PIL; print('Dependencies OK')"
```

## Data Structure

### Input (Raw Visium Data)

Place your raw data in a flat directory structure:

```
/home/students/hbae/data/GSE208253_analysis/
├── GSM6339631_s1_filtered_feature_bc_matrix.h5
├── GSM6339631_s1_tissue_hires_image.png
├── GSM6339631_s1_scalefactors_json.json
├── GSM6339631_s1_tissue_positions_list.csv
├── GSM6339632_s2_filtered_feature_bc_matrix.h5
├── GSM6339632_s2_tissue_hires_image.png
├── GSM6339632_s2_scalefactors_json.json
├── GSM6339632_s2_tissue_positions_list.csv
└── ... (more samples)
```

**Required files per sample:**
- `*_filtered_feature_bc_matrix.h5` - Gene expression counts
- `*_tissue_hires_image.png` - High-resolution tissue image
- `*_scalefactors_json.json` - Spatial scale factors
- `*_tissue_positions_list.csv` - Spot coordinates

### Output (Processed Data)

```
/home/students/hbae/processed/GSE208253/
├── patches/
│   ├── GSM6339631_s1/
│   │   ├── AAACAAGTATCTCCCA-1.png  (224×224)
│   │   ├── AAACACCAATAACTGC-1.png
│   │   └── ...
│   └── ... (more samples)
├── expressions/
│   ├── GSM6339631_s1.h5ad  (normalized AnnData)
│   └── ...
├── tables/
│   ├── train_df.csv  (main training dataframe)
│   ├── all_shared_genes.txt  (common genes across samples)
│   ├── combined_expression.npy  (n_spots × n_genes)
│   └── combined_obs.csv  (spot metadata)
├── splits/
│   └── folds.json  (10-fold cross-validation splits)
└── logs/
    ├── pipeline.log  (processing log)
    ├── qc_summary.csv  (QC statistics)
    └── report.md  (detailed report)
```

## Usage

### Basic Usage

```bash
python prep_gse208253.py \
  --raw_root /home/students/hbae/data/GSE208253_analysis \
  --out_root /home/students/hbae/processed/GSE208253
```

### Full Parameter Example

```bash
python prep_gse208253.py \
  --raw_root /home/students/hbae/data/GSE208253_analysis \
  --out_root /home/students/hbae/processed/GSE208253 \
  --min_image_size 2000 \
  --min_genes 200 \
  --k_fov 1.3 \
  --target_size 224 \
  --target_sum 10000 \
  --hvg 1000 \
  --topk_sentence 50 \
  --apply_tfidf_cap true \
  --cap_families IGH,IGK,IGL,RPL,RPS,S100,MT- \
  --cap_n 10 \
  --n_folds 10 \
  --dataset_name GSE208253
```

### Parameters

**Quality Control:**
- `--min_image_size`: Minimum image resolution (default: 2000 px)
- `--min_genes`: Minimum genes detected per spot (default: 200)

**FOV Cropping:**
- `--k_fov`: FOV size multiplier (default: 1.3× spot diameter)
- `--target_size`: Final patch size (default: 224×224)

**Normalization:**
- `--target_sum`: Target UMI count for normalization (default: 10000)
- `--hvg`: Number of highly variable genes (default: 1000)

**Gene Sentences:**
- `--topk_sentence`: Number of top genes per sentence (default: 50)
- `--apply_tfidf_cap`: Cap gene families (default: false)
- `--cap_families`: Families to cap (e.g., IGH,RPL,RPS)
- `--cap_n`: Max genes per family (default: 10)

**Cross-Validation:**
- `--n_folds`: Number of folds (default: 10)

## Pipeline Steps

### 1. Sample Discovery
Automatically detects all samples in raw_root based on h5 file names.

### 2. Quality Control
- **Image resolution check**: Excludes samples < 2000×2000 px
- **Spot filtering**: Keeps only `in_tissue==1` AND `genes_detected > 200`
- **Statistics**: Logs median UMI, genes per spot, pass rate

### 3. FOV Patch Cropping
- Calculates spot diameter: `D_hires = spot_diameter_fullres × tissue_hires_scalef`
- Crops square FOV: side = 1.3 × D_hires (centered on spot)
- Applies mirror padding at boundaries
- Resizes to 224×224 using bilinear interpolation

### 4. Expression Normalization
- **Scanpy pipeline**:
  1. `sc.pp.normalize_total(target_sum=1e4)`
  2. `sc.pp.log1p()`
  3. `sc.pp.highly_variable_genes(n_top_genes=1000, flavor='seurat_v3')`
- **Gene filtering**:
  - Converts Ensembl IDs → Gene Symbols (if available)
  - Removes housekeeping genes (ACTB, GAPDH, B2M, etc.)
  - Removes mitochondrial genes (MT-*)

### 5. Common Gene Set
- Finds intersection of genes across all samples
- Subsets each sample to common genes (same order)
- Vertically concatenates expression matrices

### 6. Gene Sentence Generation
- For each spot, selects top-50 most expressed genes (by log-normalized value)
- Joins gene symbols with spaces: `"GENE1 GENE2 GENE3 ..."`
- **Optional TF-IDF capping**: Limits repetitive gene families (IG*, RPL*, RPS*, etc.)

### 7. Cross-Validation Folds
- **GroupKFold (10-fold)**: Entire samples assigned to folds
- **No leakage**: Same sample never in both train/val
- **Patient-level separation**: Prevents overfitting

### 8. Export Training Dataframe
Creates `train_df.csv` with columns:
- `label`: Gene sentence (50 genes)
- `sample_name`: Spot barcode
- `dataset`: Dataset name (GSE208253)
- `img_path`: Absolute path to 224×224 patch
- `patient_id`: Sample identifier
- `fold_id`: Fold assignment (0-9)
- `split`: 'train' or 'val'

### 9. Report Generation
Creates detailed markdown report (`logs/report.md`) with:
- Configuration summary
- QC statistics table
- FOV cropping details
- Gene statistics
- Fold statistics
- Validation checklist

## Validation

After running the pipeline, verify outputs:

### 1. Check Random Patches

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('/home/students/hbae/processed/GSE208253/patches/GSM6339631_s1/AAACAAGTATCTCCCA-1.png')
print(img.size)  # Should be (224, 224)
plt.imshow(img)
plt.show()
```

### 2. Verify Gene List Consistency

```python
import numpy as np
import pandas as pd

# Load data
genes = open('/home/students/hbae/processed/GSE208253/tables/all_shared_genes.txt').read().strip().split('\n')
expr = np.load('/home/students/hbae/processed/GSE208253/tables/combined_expression.npy')

print(f"Genes: {len(genes)}")
print(f"Expression matrix: {expr.shape}")
assert expr.shape[1] == len(genes), "Mismatch!"
```

### 3. Inspect Training Dataframe

```python
import pandas as pd

train_df = pd.read_csv('/home/students/hbae/processed/GSE208253/tables/train_df.csv')

print(f"Total spots: {len(train_df)}")
print(f"Unique samples: {train_df['patient_id'].nunique()}")
print(f"\nSample label:\n{train_df['label'].iloc[0]}")
print(f"\nGenes in label: {len(train_df['label'].iloc[0].split())}")
```

### 4. Check Fold Leakage

```python
import json

with open('/home/students/hbae/processed/GSE208253/splits/folds.json') as f:
    folds = json.load(f)

# Check no overlap between train/val within folds
for fold_id, splits in folds.items():
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    assert len(train_set & val_set) == 0, f"Leakage in fold {fold_id}!"

print("✓ No data leakage detected")
```

## Training Contrastive Models

Use the provided skeleton to train image-text contrastive models:

```bash
python train_contrastive.py \
  --data_root /home/students/hbae/processed/GSE208253 \
  --fold 0 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-4 \
  --embed_dim 512 \
  --output_dir models/gse208253
```

**Note:** `train_contrastive.py` is a skeleton/template. You need to:
1. Install PyTorch (`pip install torch torchvision`)
2. Implement ImageEncoder (ResNet, ViT, etc.)
3. Implement TextEncoder (Transformer, BioBERT, etc.)
4. Uncomment training code sections
5. Add data augmentation transforms

## Module Reference

### `gse208253_pipeline/`

- **`io_visium.py`**: Load Visium h5/mtx data, spatial metadata, images
- **`qc.py`**: Quality control and filtering (resolution, genes)
- **`crop_fov.py`**: FOV cropping with padding, resizing to 224×224
- **`normalize.py`**: Scanpy normalization, HVG selection, gene filtering
- **`geneset.py`**: Common gene intersection, matrix combination
- **`sentence.py`**: Top-k gene sentence generation, TF-IDF capping
- **`folds.py`**: GroupKFold cross-validation without leakage
- **`export.py`**: Training dataframe creation and validation
- **`logging_utils.py`**: Logging and markdown report generation

## Troubleshooting

### Issue: "No samples found"
**Solution**: Ensure h5 files follow naming: `{sample_id}_filtered_feature_bc_matrix.h5`

### Issue: "Image not found"
**Solution**: Verify `*_tissue_hires_image.png` files exist with matching prefixes

### Issue: "KeyError: 'spot_diameter_fullres'"
**Solution**: Check `scalefactors_json.json` contains required fields

### Issue: "All samples excluded by QC"
**Solution**: Lower `--min_image_size` or `--min_genes` thresholds

### Issue: "Out of memory"
**Solution**: Process fewer samples at once, or use a machine with more RAM

## Citation

If you use this pipeline, please cite:

```
GSE208253 Preprocessing Pipeline for LOKI/OmiCLIP
https://github.com/your-repo (update with actual repo)
```

Original dataset:
```
GSE208253: [Add original paper citation]
```

## License

MIT License - feel free to modify and redistribute.

## Contact

For issues or questions:
- Open an issue on GitHub
- Contact: [your email]

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-13



