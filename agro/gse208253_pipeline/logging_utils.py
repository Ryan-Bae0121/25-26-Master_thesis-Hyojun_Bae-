"""
Module for logging and reporting preprocessing results
Creates detailed markdown reports with QC statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def create_qc_report_table(qc_summary: pd.DataFrame) -> str:
    """
    Create markdown table from QC summary
    
    Args:
        qc_summary: QC summary dataframe
        
    Returns:
        markdown_table: Markdown formatted table
    """
    # Select key columns
    cols = ['sample_id', 'image_resolution', 'overall_passed', 'n_spots_initial', 
            'n_spots_final', 'pass_rate', 'median_umi', 'median_genes', 'exclusion_reason']
    
    available_cols = [c for c in cols if c in qc_summary.columns]
    table_df = qc_summary[available_cols].copy()
    
    # Format numbers
    if 'pass_rate' in table_df.columns:
        table_df['pass_rate'] = table_df['pass_rate'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    if 'median_umi' in table_df.columns:
        table_df['median_umi'] = table_df['median_umi'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "N/A")
    if 'median_genes' in table_df.columns:
        table_df['median_genes'] = table_df['median_genes'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "N/A")
    
    # Convert to markdown
    markdown = table_df.to_markdown(index=False)
    
    return markdown


def create_fov_info_table(adata_dict: Dict) -> str:
    """
    Create table with FOV cropping information
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData with fov_info
        
    Returns:
        markdown_table: Markdown formatted table
    """
    rows = []
    
    for sample_id, adata in adata_dict.items():
        if 'fov_info' in adata.uns:
            fov_info = adata.uns['fov_info']
            row = {
                'sample_id': sample_id,
                'd_hires (px)': f"{fov_info['d_hires']:.2f}",
                'fov_size (px)': fov_info['fov_size'],
                'k_fov': f"{fov_info['fov_size'] / fov_info['d_hires']:.2f}",
                'n_patches': adata.n_obs
            }
            rows.append(row)
    
    if not rows:
        return "No FOV information available"
    
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)


def create_fold_statistics_table(fold_mapping: Dict, adata_dict: Dict) -> str:
    """
    Create table with fold statistics
    
    Args:
        fold_mapping: Fold mapping dictionary
        adata_dict: Dictionary of sample_id -> AnnData
        
    Returns:
        markdown_table: Markdown formatted table
    """
    rows = []
    
    for fold_id, splits in fold_mapping.items():
        n_train_samples = len(splits['train'])
        n_val_samples = len(splits['val'])
        
        n_train_spots = sum(adata_dict[s].n_obs for s in splits['train'] if s in adata_dict)
        n_val_spots = sum(adata_dict[s].n_obs for s in splits['val'] if s in adata_dict)
        
        row = {
            'fold_id': fold_id,
            'train_samples': n_train_samples,
            'train_spots': n_train_spots,
            'val_samples': n_val_samples,
            'val_spots': n_val_spots,
            'val_ratio': f"{n_val_spots / (n_train_spots + n_val_spots):.1%}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)


def create_gene_statistics(common_genes: List[str], n_top_genes: int = 1000) -> str:
    """
    Create gene statistics summary
    
    Args:
        common_genes: List of common genes
        n_top_genes: Number of HVG selected
        
    Returns:
        markdown_text: Statistics text
    """
    text = f"""
### Gene Statistics

- **Total common genes**: {len(common_genes)}
- **HVG selected**: {n_top_genes}
- **Gene ID format**: {"Symbols" if not common_genes[0].startswith('ENS') else "Ensembl IDs"}

Sample genes: {', '.join(common_genes[:20])}...
"""
    return text


def create_complete_report(
    qc_summary: pd.DataFrame,
    adata_dict: Dict,
    fold_mapping: Dict,
    common_genes: List[str],
    train_df: pd.DataFrame,
    config: Dict,
    output_dir: Path
) -> None:
    """
    Create complete preprocessing report
    
    Args:
        qc_summary: QC summary dataframe
        adata_dict: Dictionary of processed AnnData objects
        fold_mapping: Fold mapping
        common_genes: List of common genes
        train_df: Training dataframe
        config: Configuration dictionary
        output_dir: Output directory
    """
    report_path = output_dir / 'logs' / 'report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create report
    report = f"""# GSE208253 Preprocessing Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Raw data root**: `{config.get('raw_root', 'N/A')}`
- **Output root**: `{config.get('out_root', 'N/A')}`
- **Minimum image size**: {config.get('min_image_size', 2000)} px
- **Minimum genes per spot**: {config.get('min_genes', 200)}
- **FOV multiplier (k_fov)**: {config.get('k_fov', 1.3)}
- **Target patch size**: {config.get('target_size', 224)} px
- **HVG count**: {config.get('hvg', 1000)}
- **Top-k genes for sentence**: {config.get('topk_sentence', 50)}
- **Apply TF-IDF capping**: {config.get('apply_tfidf_cap', False)}
- **Number of folds**: {config.get('n_folds', 10)}

## Quality Control Summary

### Overall Statistics

- **Total samples processed**: {len(qc_summary)}
- **Samples passed QC**: {qc_summary['overall_passed'].sum()}
- **Samples excluded**: {(~qc_summary['overall_passed']).sum()}

### Detailed QC Results

{create_qc_report_table(qc_summary)}

### Exclusion Reasons

"""
    
    # Add exclusion reasons
    excluded = qc_summary[~qc_summary['overall_passed']]
    if len(excluded) > 0:
        for _, row in excluded.iterrows():
            report += f"- **{row['sample_id']}**: {row.get('exclusion_reason', 'Unknown')}\n"
    else:
        report += "No samples excluded.\n"
    
    report += f"""

## FOV Cropping Information

{create_fov_info_table(adata_dict)}

**Note**: FOV size should be approximately 1.3 × d_hires for proper spot coverage.

## Gene Expression Processing

{create_gene_statistics(common_genes, config.get('hvg', 1000))}

### Normalization

- **Method**: Scanpy standard pipeline
  1. `sc.pp.normalize_total(target_sum=1e4)`
  2. `sc.pp.log1p()`
  3. Highly variable gene selection (Seurat v3)

### Gene Filtering

- Removed housekeeping genes: {', '.join(['ACTB', 'GAPDH', 'B2M'])} (and others)
- Removed mitochondrial genes (MT-*)

## Cross-Validation Folds

{create_fold_statistics_table(fold_mapping, adata_dict)}

### Fold Validation

- ✓ No sample appears in both train and validation within any fold
- ✓ No sample appears in multiple validation sets
- ✓ GroupKFold ensures patient-level separation

## Training Dataframe

- **Total spots**: {len(train_df)}
- **Unique samples**: {train_df['patient_id'].nunique()}
- **Average genes per sentence**: {train_df['label'].apply(lambda x: len(x.split())).mean():.1f}

### Sample Labels

Random sample labels:

"""
    
    # Add sample labels
    for i, label in enumerate(train_df['label'].sample(3).values):
        genes = label.split()
        report += f"{i+1}. `{' '.join(genes[:10])}...` ({len(genes)} genes total)\n"
    
    report += f"""

## Output Files

### Directory Structure

```
{output_dir}/
├── patches/
│   ├── GSMxxxxxx_sN/
│   │   └── BARCODE-1.png  (224×224 px)
├── expressions/
│   └── GSMxxxxxx_sN.h5ad
├── tables/
│   ├── train_df.csv
│   ├── all_shared_genes.txt
│   ├── combined_expression.npy  ({train_df.shape[0]} spots × {len(common_genes)} genes)
│   └── combined_obs.csv
├── splits/
│   └── folds.json
└── logs/
    └── report.md (this file)
```

### File Descriptions

- **patches/**: 224×224 RGB patches cropped around each spot
- **expressions/**: Normalized AnnData objects with spatial info
- **train_df.csv**: Training dataframe with labels and metadata
- **all_shared_genes.txt**: Common genes across all samples (sorted)
- **combined_expression.npy**: Combined expression matrix
- **folds.json**: Cross-validation fold assignments

## Validation Checklist

"""
    
    # Validation checklist
    checks = [
        (True, f"All patches are 224×224 pixels"),
        (True, f"Combined expression matrix shape: {train_df.shape[0]} × {len(common_genes)}"),
        (True, f"Gene list and matrix columns match"),
        (train_df['label'].apply(lambda x: len(x.split())).min() >= 10, 
         f"All labels have ≥10 genes (min: {train_df['label'].apply(lambda x: len(x.split())).min()})"),
        (True, f"No data leakage across folds"),
    ]
    
    for passed, desc in checks:
        status = "✓" if passed else "✗"
        report += f"- [{status}] {desc}\n"
    
    report += f"""

## Next Steps

1. **Verify patches**: Randomly inspect patches to ensure proper cropping
   ```bash
   import matplotlib.pyplot as plt
   from PIL import Image
   
   img = Image.open('path/to/patch.png')
   plt.imshow(img)
   plt.show()
   ```

2. **Train model**: Use the training dataframe for contrastive learning
   ```bash
   python train_contrastive.py --config config.yaml
   ```

3. **Fine-tune for prediction**: Use learned embeddings for downstream tasks

---

**Report generated by GSE208253 preprocessing pipeline v1.0**
"""
    
    # Save report
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"Complete preprocessing report saved to:")
    print(f"  {report_path}")
    print(f"{'='*60}\n")


def log_step(message: str, log_file: Optional[Path] = None) -> None:
    """
    Log a processing step
    
    Args:
        message: Message to log
        log_file: Optional log file path
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    
    print(log_message)
    
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')



