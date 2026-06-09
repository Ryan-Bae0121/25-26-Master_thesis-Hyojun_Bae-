#!/usr/bin/env python3
"""
Main CLI script for GSE208253 preprocessing pipeline
Orchestrates the complete workflow from raw data to training-ready format
"""

import argparse
import sys
from pathlib import Path
import glob

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from gse208253_pipeline import (
    io_visium, qc, crop_fov, normalize, geneset, 
    sentence, folds, export, logging_utils
)


def discover_samples(raw_root: Path) -> list:
    """
    Discover all sample IDs from file names in raw_root
    
    Args:
        raw_root: Root directory with raw data
        
    Returns:
        sample_ids: List of sample identifiers
    """
    # Find all h5 files
    h5_files = glob.glob(str(raw_root / "*_filtered_feature_bc_matrix.h5"))
    
    sample_ids = []
    for h5_file in h5_files:
        # Extract sample ID from filename
        filename = Path(h5_file).name
        sample_id = filename.replace('_filtered_feature_bc_matrix.h5', '')
        sample_ids.append(sample_id)
    
    sample_ids.sort()
    
    print(f"Discovered {len(sample_ids)} samples: {sample_ids}")
    
    return sample_ids


def main():
    parser = argparse.ArgumentParser(
        description="GSE208253 Visium Preprocessing Pipeline for LOKI/OmiCLIP"
    )
    
    # Input/Output
    parser.add_argument('--raw_root', type=str, required=True,
                        help='Root directory with raw Visium data')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Output root directory')
    
    # QC parameters
    parser.add_argument('--min_image_size', type=int, default=2000,
                        help='Minimum image size (default: 2000)')
    parser.add_argument('--min_genes', type=int, default=200,
                        help='Minimum genes per spot (default: 200)')
    
    # FOV cropping parameters
    parser.add_argument('--k_fov', type=float, default=1.3,
                        help='FOV size multiplier (default: 1.3)')
    parser.add_argument('--target_size', type=int, default=224,
                        help='Target patch size (default: 224)')
    
    # Normalization parameters
    parser.add_argument('--target_sum', type=float, default=1e4,
                        help='Target sum for normalization (default: 10000)')
    parser.add_argument('--hvg', type=int, default=1000,
                        help='Number of highly variable genes (default: 1000)')
    
    # Sentence generation parameters
    parser.add_argument('--topk_sentence', type=int, default=50,
                        help='Number of top genes for sentence (default: 50)')
    parser.add_argument('--apply_tfidf_cap', type=str, default='false',
                        choices=['true', 'false'],
                        help='Apply TF-IDF style family capping (default: false)')
    parser.add_argument('--cap_families', type=str, default='IGH,IGK,IGL,RPL,RPS,S100,MT-',
                        help='Comma-separated gene families to cap')
    parser.add_argument('--cap_n', type=int, default=10,
                        help='Maximum genes per family (default: 10)')
    
    # Fold parameters
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of cross-validation folds (default: 10)')
    
    # Dataset name
    parser.add_argument('--dataset_name', type=str, default='GSE208253',
                        help='Dataset name for labeling (default: GSE208253)')
    
    args = parser.parse_args()
    
    # Convert paths
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    
    # Convert boolean
    apply_tfidf_cap = (args.apply_tfidf_cap.lower() == 'true')
    cap_families = args.cap_families.split(',') if apply_tfidf_cap else None
    
    # Create output directories
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'patches').mkdir(exist_ok=True)
    (out_root / 'expressions').mkdir(exist_ok=True)
    (out_root / 'tables').mkdir(exist_ok=True)
    (out_root / 'splits').mkdir(exist_ok=True)
    (out_root / 'logs').mkdir(exist_ok=True)
    
    # Setup logging
    log_file = out_root / 'logs' / 'pipeline.log'
    
    logging_utils.log_step("="*60, log_file)
    logging_utils.log_step("GSE208253 Preprocessing Pipeline", log_file)
    logging_utils.log_step("="*60, log_file)
    
    # Store configuration
    config = vars(args)
    
    # ===========================
    # Step 1: Discover samples
    # ===========================
    logging_utils.log_step("\n[Step 1] Discovering samples...", log_file)
    sample_ids = discover_samples(raw_root)
    
    if not sample_ids:
        logging_utils.log_step("ERROR: No samples found!", log_file)
        return 1
    
    # ===========================
    # Step 2: Load and QC samples
    # ===========================
    logging_utils.log_step(f"\n[Step 2] Loading and performing QC on {len(sample_ids)} samples...", log_file)
    
    passed_samples, qc_summary = qc.batch_qc_samples(
        sample_ids=sample_ids,
        raw_root=str(raw_root),
        min_image_size=args.min_image_size,
        min_genes=args.min_genes,
        in_tissue_only=True
    )
    
    logging_utils.log_step(f"  QC passed: {len(passed_samples)}/{len(sample_ids)} samples", log_file)
    
    if not passed_samples:
        logging_utils.log_step("ERROR: No samples passed QC!", log_file)
        return 1
    
    # Save QC summary
    qc_summary.to_csv(out_root / 'logs' / 'qc_summary.csv', index=False)
    
    # ===========================
    # Step 3: Crop FOV patches
    # ===========================
    logging_utils.log_step("\n[Step 3] Cropping FOV patches...", log_file)
    
    for sample_id, adata in passed_samples.items():
        logging_utils.log_step(f"  Processing {sample_id}...", log_file)
        
        image = adata.uns['spatial']['hires_image']
        scalefactors = adata.uns['spatial']['scalefactors']
        
        patch_paths, d_hires, fov_size = crop_fov.process_sample_patches(
            adata=adata,
            image=image,
            scalefactors=scalefactors,
            out_dir=out_root / 'patches',
            sample_id=sample_id,
            k_fov=args.k_fov,
            target_size=args.target_size
        )
        
        # Add patch info to adata
        adata = crop_fov.add_patch_info_to_adata(adata, patch_paths, d_hires, fov_size)
        passed_samples[sample_id] = adata
        
        logging_utils.log_step(f"    Created {len(patch_paths)} patches "
                              f"(d_hires={d_hires:.2f}px, fov={fov_size}px)", log_file)
    
    # ===========================
    # Step 4: Normalize expression
    # ===========================
    logging_utils.log_step("\n[Step 4] Normalizing expression...", log_file)
    
    normalized_samples = normalize.batch_normalize_samples(
        samples_dict=passed_samples,
        target_sum=args.target_sum,
        n_top_genes=args.hvg,
        remove_housekeeping=True
    )
    
    # Save normalized samples
    for sample_id, adata in normalized_samples.items():
        output_path = out_root / 'expressions' / f'{sample_id}.h5ad'
        normalize.save_normalized_sample(adata, output_path)
    
    logging_utils.log_step(f"  Saved {len(normalized_samples)} normalized samples", log_file)
    
    # ===========================
    # Step 5: Create common gene set and combine
    # ===========================
    logging_utils.log_step("\n[Step 5] Creating common gene set and combining expression...", log_file)
    
    common_genes, combined_expression, combined_obs = geneset.process_all_samples(
        adata_dict=normalized_samples,
        output_dir=out_root / 'tables',
        layer=None  # Use normalized .X
    )
    
    # ===========================
    # Step 6: Generate gene sentences
    # ===========================
    logging_utils.log_step("\n[Step 6] Generating gene sentences...", log_file)
    
    sentences_dict = sentence.batch_create_sentences(
        adata_dict=normalized_samples,
        k=args.topk_sentence,
        layer=None,
        apply_tfidf_cap=apply_tfidf_cap,
        cap_families=cap_families,
        max_per_family=args.cap_n
    )
    
    # Add sentences to adata
    for sample_id, adata in normalized_samples.items():
        adata = sentence.add_sentences_to_adata(adata, sentences_dict[sample_id])
        normalized_samples[sample_id] = adata
    
    # Verify sentence format
    all_sentences = []
    for s in sentences_dict.values():
        all_sentences.extend(s.tolist())
    sentence.verify_sentence_format(
        pd.Series(all_sentences), 
        expected_k=args.topk_sentence
    )
    
    # ===========================
    # Step 7: Create cross-validation folds
    # ===========================
    logging_utils.log_step("\n[Step 7] Creating cross-validation folds...", log_file)
    
    fold_mapping = folds.create_fold_mapping(
        adata_dict=normalized_samples,
        n_splits=args.n_folds,
        random_state=42
    )
    
    # Save folds
    folds.save_folds(fold_mapping, out_root / 'splits' / 'folds.json')
    
    # ===========================
    # Step 8: Export training dataframe
    # ===========================
    logging_utils.log_step("\n[Step 8] Exporting training dataframe...", log_file)
    
    train_df = export.export_complete_dataset(
        adata_dict=normalized_samples,
        sentences_dict=sentences_dict,
        fold_mapping=fold_mapping,
        combined_expression=combined_expression,
        common_genes=common_genes,
        output_dir=out_root,
        dataset_name=args.dataset_name
    )
    
    # ===========================
    # Step 9: Generate final report
    # ===========================
    logging_utils.log_step("\n[Step 9] Generating final report...", log_file)
    
    logging_utils.create_complete_report(
        qc_summary=qc_summary,
        adata_dict=normalized_samples,
        fold_mapping=fold_mapping,
        common_genes=common_genes,
        train_df=train_df,
        config=config,
        output_dir=out_root
    )
    
    # ===========================
    # Final summary
    # ===========================
    logging_utils.log_step("\n" + "="*60, log_file)
    logging_utils.log_step("PREPROCESSING COMPLETE!", log_file)
    logging_utils.log_step("="*60, log_file)
    logging_utils.log_step(f"\nOutputs saved to: {out_root}", log_file)
    logging_utils.log_step(f"  - {len(passed_samples)} samples processed", log_file)
    logging_utils.log_step(f"  - {len(train_df)} total spots", log_file)
    logging_utils.log_step(f"  - {len(common_genes)} common genes", log_file)
    logging_utils.log_step(f"  - {args.n_folds} cross-validation folds", log_file)
    logging_utils.log_step(f"\nNext steps:", log_file)
    logging_utils.log_step(f"  1. Review QC report: {out_root}/logs/report.md", log_file)
    logging_utils.log_step(f"  2. Inspect sample patches", log_file)
    logging_utils.log_step(f"  3. Train contrastive model using train_df.csv", log_file)
    logging_utils.log_step("", log_file)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())



